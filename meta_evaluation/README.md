# Meta-Evaluation: Human-Model Agreement Analysis

This directory contains tools for evaluating agreement between human and automated model evaluations in pairwise comparisons of ScholarQA-CS2 eval system outputs.

## Project Structure

```
meta_evaluation/
├── calculate_agreement.py          # Main analysis script
└── annotation/
    ├── pairwise_authored.json      # Authored (Deep-Expert) evaluation set
    ├── pairwise_chosen.json        # Chosen (Near-Expert) evaluation set
    ├── pairwise_dev_assigned_ann1.json   # Dev set annotator 1
    ├── pairwise_dev_assigned_ann2.json   # Dev set annotator 2
    ├── pairwise_test_assigned_ann1.json  # Test set annotator 1
    ├── pairwise_test_assigned_ann2.json  # Test set annotator 2
    ├── pairwise_dev_assigned_agreement_only.json   # Dev set (agreed only)
    └── pairwise_test_assigned_agreement_only.json  # Test set (agreed only)
```

## Requirements

- Python 3.x
- Required packages:
  - numpy
  - pandas
  - scipy

## Data Format

Each annotation entry contains pairwise comparison data:
```json
{
  "question": "The research question text",
  "models": ["model_a", "model_b"],
  "question_intent": "Category of question",
  "human_overall": [0.85, 0.875],
  "model_overall": [0.567, 0.564],
  "human_answer_precision": [...],
  "model_answer_precision": [...],
  ...
}
```

## Usage

Run the agreement analysis script:

```bash
python calculate_agreement.py [OPTIONS]
```

Configuration options:
- `--tie_strategy`: How to handle ties (`threshold`, `exclude`, or `partial`, default: `threshold`)
- `--dont_use_thresholds`: Disable threshold-based tie detection
- `--subselect_models`: Comma-delimited list of models to analyze (optional)
- `--human_agreed`: Only use instances where human annotators agreed
- `--drop_elicit`: Exclude 'elicit' model from system ranking
- `--source`: Filter by data source (`authored`, `chosen`, or `assigned`, default: `assigned`)

### Examples

Basic usage with default settings:
```bash
python calculate_agreement.py
```

Calculate agreement only on human-agreed instances:
```bash
python calculate_agreement.py --human_agreed
```

Exclude ties from analysis:
```bash
python calculate_agreement.py --tie_strategy exclude
```

Analyze specific models:
```bash
python calculate_agreement.py --subselect_models sqa,openai-dr,perplexity
```

Use authored data source:
```bash
python calculate_agreement.py --source authored
```

## Output

The script generates three types of analysis:

1. **Inter-Annotator Agreement (IAA)**: When using doubly-annotated data
   - Strict agreement rate
   - Half-credit agreement rate (gives credit for one annotator calling a tie)

2. **Agreement Metrics Table**: For each human-model metric pair
   - Agreement rate
   - Kendall's tau correlation
   - Total number of comparisons
   - Optimal threshold used

3. **System Rankings**: Model performance rankings based on:
   - Human preference rankings
   - Automated overall scores
   - Kendall's tau correlation between rankings

