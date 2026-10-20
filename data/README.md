# DeepLearnQ Dataset Documentation

## Overview
This directory contains the training and evaluation dataset for **DeepLearnQ: Intelligent Question Paper Analyzer**.

## Schema Requirements
Every dataset imported or used by DeepLearnQ must adhere to the following schema:
- **Columns**: `question,bloom_level`
- **Encoding**: UTF-8
- **Permitted Labels (`bloom_level`)**:
  1. `Remember` — Recall facts, definitions, terminology, and basic concepts (BT1).
  2. `Understand` — Explain ideas or concepts, interpret, summarize, paraphrase (BT2).
  3. `Apply` — Use information in new situations, execute, implement, solve problems (BT3).
  4. `Analyze` — Draw connections among ideas, differentiate, organize, attribute (BT4).
  5. `Evaluate` — Justify a stand or decision, appraise, argue, defend, judge, critique (BT5).
  6. `Create` — Produce new or original work, design, assemble, construct, formulate (BT6).

## Active Dataset: `blooms_taxonomy_dataset.csv`
- **Source**: Academic benchmark collection aggregated from published educational data-mining studies (Mendeley Data, Kaggle Educational Question Repository, and published Bloom taxonomy NLP benchmarks).
- **Total Samples**: 8,758 cleaned examination questions.
- **Class Distribution**:
  - `Remember`: 2,581 (29.5%)
  - `Understand`: 1,800 (20.6%)
  - `Apply`: 1,506 (17.2%)
  - `Analyze`: 1,291 (14.7%)
  - `Create`: 800 (9.1%)
  - `Evaluate`: 780 (8.9%)
- **Data Cleansing Process**:
  - Removed duplicate queries.
  - Stripped leading/trailing whitespace and malformed control characters.
  - Normalized BT1-BT6 identifier strings to canonical title-case category names.
  - Filtered out uninformative snippets (< 5 characters).

## Template: `dataset_template.csv`
If you have your own college examination questions or department question bank, format it matching `dataset_template.csv`:
```csv
question,bloom_level
"Define Ohm's law and state its SI unit.",Remember
"Explain the working principle of a transformer.",Understand
"Calculate the total resistance in a parallel circuit with 10 ohm resistors.",Apply
"Differentiate between synchronous and asynchronous counters.",Analyze
"Assess the performance trade-offs of microservices vs monolithic architecture.",Evaluate
"Develop an algorithm to detect cycles in a directed graph.",Create
```
