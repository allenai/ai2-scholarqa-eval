# Input Files

This directory contains input data for the rubric building pipeline.

## Required Input Format

`all_report_relevant_texts.jsonl`

A JSONL file where each line contains a system report to extract ingredients from.

**Format:**
```json
{
  "query": "The research question or query text",
  "solver": "name-of-the-system",
  "answer": "Full system report",
  "answer_edited": "Full system report (with irrelevant passages removed)"
}
```


### Sample Data

This directory includes sample data.