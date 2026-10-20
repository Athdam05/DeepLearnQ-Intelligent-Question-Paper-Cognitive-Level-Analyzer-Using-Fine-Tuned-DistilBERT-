import re
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple

# Common regex patterns to detect question starts in examination papers
QUESTION_PATTERNS = [
    # Q1., Q. 1., Q1), Question 1:, Question 1., Question - 1
    re.compile(r"^(?:(?:Q|Question)\.?\s*[-:]?\s*(\d+))[\.\:\)\-]\s*(.*)", re.IGNORECASE),
    # 1., 2., 12. at start of line
    re.compile(r"^(\d{1,3})\s*[\.\:\)]\s*(.*)", re.IGNORECASE),
    # [1], (1)
    re.compile(r"^(?:\[|\()(\d{1,3})(?:\]|\))\s*(.*)", re.IGNORECASE),
]

# Section headers to skip (e.g., Section A, Part 1, Instructions)
SECTION_HEADER_PATTERN = re.compile(
    r"^(?:section|part|unit|module|group)\s+([a-z0-9ivx]+)|instructions?|marks|time\s*allowed|maximum\s*marks|note\s*:",
    re.IGNORECASE
)

def extract_text_from_pdf(pdf_stream_or_path) -> Tuple[str, List[Dict[str, Any]], bool]:
    """
    Extracts text from each page of a PDF using PyMuPDF.
    Returns:
        full_text: entire string
        pages_info: list of dicts with page_number and text
        is_scanned: boolean indicating if PDF is likely scanned image
    """
    if isinstance(pdf_stream_or_path, (bytes, bytearray)):
        doc = fitz.open(stream=pdf_stream_or_path, filetype="pdf")
    else:
        doc = fitz.open(pdf_stream_or_path)

    total_pages = len(doc)
    pages_info = []
    full_text_list = []
    total_text_length = 0
    total_images = 0

    for page_idx in range(total_pages):
        page = doc[page_idx]
        page_text = page.get_text("text").strip()
        image_list = page.get_images()
        total_images += len(image_list)
        total_text_length += len(page_text)

        pages_info.append({
            "page_number": page_idx + 1,
            "text": page_text,
            "has_images": len(image_list) > 0
        })
        full_text_list.append(page_text)

    doc.close()

    full_text = "\n\n".join(full_text_list)

    # Scanned PDF heuristic: very few text characters per page, but contains pages or images
    avg_chars_per_page = total_text_length / max(1, total_pages)
    is_scanned = (total_text_length < 40 and total_pages > 0) or (avg_chars_per_page < 25 and total_images > 0)

    return full_text, pages_info, is_scanned

def segment_questions_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Segments extracted text into individual questions using regular expressions
    and heuristic boundary detection.
    """
    lines = text.splitlines()
    raw_questions = []
    current_q_num = None
    current_q_text = []

    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue

        # Skip typical exam metadata headers
        if SECTION_HEADER_PATTERN.match(cleaned_line):
            continue

        matched_q = False
        for pattern in QUESTION_PATTERNS:
            match = pattern.match(cleaned_line)
            if match:
                # Save previous question if exists
                if current_q_text:
                    q_body = " ".join(current_q_text).strip()
                    if len(q_body) >= 5:
                        raw_questions.append({
                            "number": current_q_num or (len(raw_questions) + 1),
                            "text": q_body
                        })
                
                # Start new question
                q_num_str = match.group(1)
                try:
                    current_q_num = int(q_num_str)
                except ValueError:
                    current_q_num = len(raw_questions) + 1

                remainder = match.group(2).strip()
                current_q_text = [remainder] if remainder else []
                matched_q = True
                break

        if not matched_q:
            if current_q_text is not None and len(current_q_text) > 0:
                current_q_text.append(cleaned_line)
            elif not current_q_text:
                # If no question number was matched yet, check if this line looks like a question
                if cleaned_line.endswith("?") or any(cleaned_line.lower().startswith(w) for w in ["what", "why", "how", "explain", "define", "calculate", "compare", "design"]):
                    current_q_num = len(raw_questions) + 1
                    current_q_text = [cleaned_line]

    # Append the last accumulated question
    if current_q_text:
        q_body = " ".join(current_q_text).strip()
        if len(q_body) >= 5:
            raw_questions.append({
                "number": current_q_num or (len(raw_questions) + 1),
                "text": q_body
            })

    # Fallback: if no question numbers matched, split by question mark or double newline
    if not raw_questions:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\?(?=\s+[A-Z])", text) if len(p.strip()) > 8]
        for i, p in enumerate(paragraphs):
            cleaned = p + ("?" if not p.endswith((".", "?", "!")) else "")
            raw_questions.append({
                "number": i + 1,
                "text": cleaned
            })

    # Clean up formatting in question texts
    final_questions = []
    for i, q in enumerate(raw_questions):
        txt = re.sub(r"\s+", " ", q["text"]).strip()
        # Remove trailing marks indicators like [5 Marks], (10M), [CO1], etc.
        txt = re.sub(r"\[\s*\d+\s*(?:marks?|m)\s*\]|\(\s*\d+\s*(?:marks?|m)\s*\)", "", txt, flags=re.IGNORECASE).strip()
        if len(txt) >= 5:
            final_questions.append({
                "id": i + 1,
                "number": q.get("number", i + 1),
                "text": txt
            })

    return final_questions

def parse_manual_questions(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parses manually pasted questions from the user interface.
    Accepts questions separated by newlines, numbers, or bullet points.
    """
    if not raw_text or not raw_text.strip():
        return []

    lines = raw_text.strip().splitlines()
    questions = []
    current_q = []

    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            if current_q:
                q_text = " ".join(current_q).strip()
                if len(q_text) >= 4:
                    questions.append(q_text)
                current_q = []
            continue

        # Check if line begins with a number/bullet
        if re.match(r"^(?:Q\d+[\.:\)]|\d+[\.:\)]|[-*•])\s+", cleaned):
            if current_q:
                q_text = " ".join(current_q).strip()
                if len(q_text) >= 4:
                    questions.append(q_text)
                current_q = []
            # Strip the leading numbering/bullet
            body = re.sub(r"^(?:Q\d+[\.:\)]|\d+[\.:\)]|[-*•])\s+", "", cleaned)
            current_q.append(body)
        else:
            current_q.append(cleaned)

    if current_q:
        q_text = " ".join(current_q).strip()
        if len(q_text) >= 4:
            questions.append(q_text)

    return [{"id": i + 1, "number": i + 1, "text": q} for i, q in enumerate(questions)]
