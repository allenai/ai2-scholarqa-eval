# AI2 ScholarQA Evaluation

Tools for evaluating AI-generated responses to scholarly questions, including automated rubric generation and agreement analysis between human and model evaluations.

## Components

### Rubric Building
Automated pipeline for generating evaluation rubrics from system reports using Claude's API. Extracts key requirements from reports and unifies them into comprehensive, weighted evaluation criteria.

See [`rubric_building/README.md`](rubric_building/README.md) for details.

### Meta-Evaluation
Analysis of agreement between human and automated model evaluations in pairwise comparisons of ScholarQA-CS2 Eval system outputs. Supports multiple evaluation strategies, inter-annotator agreement calculation, and optimal threshold tuning.

See [`meta_evaluation/README.md`](meta_evaluation/README.md) for details.

## Requirements

Python 3.10+ is recommended.

Set up a conda environment and install dependencies:
```bash
conda create -n ai2-scholarqa-eval python=3.10 -y
conda activate ai2-scholarqa-eval
pip install -r requirements.txt
```

For rubric building with Claude API:
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

**Note:** The `ANTHROPIC_API_KEY` is only required when running the rubric building scripts with `sanity_check: false` in the config. You can do a dry run with `sanity_check: true` (the default) without setting the key.