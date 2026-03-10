# Ai2 ScholarQA Evaluation


## Components

This repository contains code and data from **ScholarQA-CS2 Evaluations**, and is associated with two publications:

**[1]** Hwang et al. 2026, [*Deep Research, Shallow Evaluation:A Case Study in Meta-Evaluation for Long-Form QA Benchmarks*](https://arxiv.org/abs/2603.06942)\
**[2]** Bragg et al. 2025, [*AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite*](https://arxiv.org/abs/2510.21652)

This repository includes:
1. Code and annotated data from in the human meta-evaluation conducted over of ScholarQA-CS2 **[1]**. 
2. The implementation of the automated rubric building pipeline supporting ScholarQA-CS2 discussed and released with AstaBench **[2]**. ScholarQA-CS2's [rubrics](https://huggingface.co/datasets/allenai/asta-bench/tree/main/tasks/sqa) are generated via this pipeline. 

**Meta-Evaluation**:
Analysis of agreement between human and automated model evaluations in pairwise comparisons of ScholarQA-CS2 Eval system outputs. Supports multiple evaluation strategies, inter-annotator agreement calculation, and optimal threshold tuning.


**Rubric Building**:
Automated pipeline for generating evaluation rubrics from system reports using Claude's API. Extracts key requirements from reports and unifies them into comprehensive, weighted evaluation criteria.


## Requirements
- Python 3.10+
- Rubric building scripts require Claude API.

## Quick Set Up
Clone the repo and set up the environment as follows:
```
git clone https://github.com/allenai/ai2-scholarqa-eval.git
cd ai2-scholarqa-eval
conda create -n sqaeval python=3.10
conda activate sqaeval
```
<br>

To run **meta-evaluation analyses**:
```
cd meta_evaluation/ 
```
See [`rubric_building/README.md`](rubric_building/README.md) for details.

<br>

To run **rubric building**:

- For generating ingredients (step 1 of the pipeline):
    
    ```
    python rubric_building/1_generate_ingredients.py
    ```
- For unifying ingredients into rubric (step 2):
    ```
    python rubric_building/2_unify_ingredients.py
    ```

See [`meta_evaluation/README.md`](meta_evaluation/README.md) for details.

Both scripts uses API key to run Claude. However, sanity-check test runs 
(default set up; quick run without API call) will let you execute the script without it.
To specify your key run:
```
export ANTHROPIC_API_KEY='your-api-key-here'
```


## License

The code is licensed under Apache 2.0. The data is licensed under ODC-BY 1.0. The artifacts are intended for research and educational use in accordance with Ai2's Responsible Use Guidelines. 


## How to Cite

Citations are supplied in the README files within the respective directory.