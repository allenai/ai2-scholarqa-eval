# Rubric Building for AI2 ScholarQA Evaluation

This directory includes two-stage pipeline for automatically generating evaluation rubrics from system reports using Claude's API. The pipeline extracts key requirements ("ingredients") from reports and unifies them into comprehensive evaluation rubrics with weighted criteria.

## Overview

The rubric building process consists of two main stages:

1. **Ingredient Extraction** - Analyzes system reports to identify key requirements and criteria
2. **Ingredient Unification** - Consolidates ingredients from multiple sources into coherent, non-overlapping rubrics

## Project Structure

```
rubric_building/
├── 1_generate_ingredients.py      # Stage 1: Extract ingredients from reports
├── 2_unify_ingredients.py          # Stage 2: Unify ingredients into rubrics
├── claude_process_batch.py        # Process batch API results
├── utils_rubric_building.py       # Utility functions (JSON parsing, batch requests)
├── config_extract_ingredients.yaml # Configuration for extraction stage
├── config_unify_ingredients.yaml   # Configuration for unification stage
├── input_files/
│   └── all_report_relevant_texts.jsonl  # Input reports
├── logs_extraction/               # Extraction stage outputs
├── logs_unification/              # Unification stage outputs
└── outputs/                       # Final rubrics
```

## Requirements

- Python 3.x
- Required packages:
  - anthropic
  - pandas
  - pyyaml
  - tqdm

Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

## Usage

### Stage 1: Extract Ingredients

Extract key requirements from system reports:

```bash
# From the rubric_building directory:
python 1_generate_ingredients.py

# Or from the repo root:
python rubric_building/1_generate_ingredients.py
```

Configuration options in `config_extract_ingredients.yaml`:
- `batch_run`: Use batch processing (recommended)
- `sanity_check`: Preview requests without sending API call
- `specific_solvers`: Filter by specific solver names (optional)
- `query_constraints`: Filter by specific queries (optional)
- `extractor_model`: Claude model to use (default: claude-opus-4)
- `all_report_relevant_text_file`: Input JSONL file with reports
- `log_dir`: Output directory for logs and results

**Output**: Generated ingredients saved to `logs_extraction/` with timestamps

### Stage 2: Unify Ingredients

Consolidate ingredients from multiple sources into unified rubrics:

```bash
# From the rubric_building directory:
python 2_unify_ingredients.py

# Or from the repo root:
python rubric_building/2_unify_ingredients.py
```

Configuration options in `config_unify_ingredients.yaml`:
- `batch_run`: Use batch processing (recommended)
- `sanity_check`: Preview requests without sending API call
- `all_ingredients_file`: Path to extraction output from Stage 1. **Update this** to point to your actual Stage 1 output (e.g., `logs_extraction/YYYYMMDD-HHMM-generation_output.jsonl`)
- `unifier_models`: List of models to use for unification
- `log_dir`: Output directory for results

**Output**: Unified rubrics saved to `logs_unification/`

### Processing Batch Results

After batch jobs complete, process the results:

```bash
# Check batch status
python claude_process_batch.py -r check_status -l logs_extraction -p YYYYMMDD-HHMM

# Process extraction results
python claude_process_batch.py -r process_extraction_results -l logs_extraction -p YYYYMMDD-HHMM

# Process unification results
python claude_process_batch.py -r process_unification_results -l logs_unification -p YYYYMMDD-HHMM

# Cancel a batch
python claude_process_batch.py -r cancel_batch -l logs_extraction -p YYYYMMDD-HHMM
```

Arguments:
- `-r, --run`: Operation to perform
- `-l, --log_dir`: Log directory containing run files
- `-p, --run_prefix`: Timestamp prefix from run_info.json file
- `-s, --saved_generation`: (Optional) Use cached generation results

## Quick Start: End-to-End Walkthrough

This walkthrough runs the full pipeline. Stage 1 can be tested in sanity-check mode (no API calls, no API key needed).

### 1. Extract ingredients (dry run)
Verify that `config_extract_ingredients.yaml` has `sanity_check: true` (the default), then run:
```bash
python rubric_building/1_generate_ingredients.py
```
This prints the prompts that would be sent to the API without making any calls.

### 2. Extract ingredients (live run)
Set `sanity_check: false` in `config_extract_ingredients.yaml`, then:
```bash
export ANTHROPIC_API_KEY='your-key'
python rubric_building/1_generate_ingredients.py
```
The batch job ID and metadata are saved to `logs_extraction/`.

### 3. Process Stage 1 batch results
Once the batch completes:
```bash
python rubric_building/claude_process_batch.py -r check_status -l logs_extraction -p YYYYMMDD-HHMM
python rubric_building/claude_process_batch.py -r process_extraction_results -l logs_extraction -p YYYYMMDD-HHMM
```
This writes `logs_extraction/YYYYMMDD-HHMM-generation_output.jsonl`.

### 4. Unify ingredients
Update `config_unify_ingredients.yaml`:
- Set `all_ingredients_file` to the output from Step 3 (e.g., `logs_extraction/YYYYMMDD-HHMM-generation_output.jsonl`)
- Set `sanity_check: false`

```bash
python rubric_building/2_unify_ingredients.py
```

### 5. Process Stage 2 batch results
```bash
python rubric_building/claude_process_batch.py -r check_status -l logs_unification -p YYYYMMDD-HHMM
python rubric_building/claude_process_batch.py -r process_unification_results -l logs_unification -p YYYYMMDD-HHMM
```
Final rubrics are written to `outputs/`.

## Pipeline Details

### Stage 1: Ingredient Extraction

The extraction stage analyzes system reports to identify:
- **High-level requirements**: What concepts should be covered
- **Specific details**: Examples and citations supporting each requirement
- **Criticality**: Whether requirements are answer-critical (SHOULD) or valuable (MIGHT)

Key features:
- Uses Claude's extended thinking mode (3072 token budget)
- Prompt caching for efficiency
- Sequential processing through reports
- Outputs structured JSON with ingredient IDs, descriptions, examples, and citations

### Stage 2: Ingredient Unification

The unification stage:
1. Identifies key concepts and entities from the query
2. Groups overlapping or duplicate ingredients
3. Creates minimal set of non-overlapping requirements
4. Distributes examples across unified requirements
5. Prunes duplicate or irrelevant examples
6. Assigns weights: 2x for answer-critical, 1x for valuable

**Output format**:
```json
{
  "question": "query text",
  "ingredients": [
    {
      "name": "answer_critical_0",
      "criterion": "requirement description",
      "weight": 0.25,
      "examples": ["example1", "example2"]
    }
  ],
  "case_id": "unique_id",
  "annotator": "auto",
  "exp": "experiment_name"
}
```

