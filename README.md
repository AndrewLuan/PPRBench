# PPRBench

**PPRBench: A Process-level Benchmark for LLMs' Physical Reasoning**

PPRBench evaluates LLMs on physics problem solving using a process-level grading methodology — rewarding correct intermediate derivations and formulas, not just final answers. The benchmark covers Chinese physics olympiad and undergraduate-level problems across six categories: Electromagnetism, Mechanics, Optics, Quantum Physics, Thermodynamics, and Special Relativity.

## Dataset

`dataset/training.json` contains the benchmark problems. Each entry has the following fields:

| Field | Description |
|---|---|
| `question_id` | Unique integer ID |
| `question` | Problem statement (LaTeX) |
| `answer` | Reference solution with numbered formulas |
| `scoring_criteria` | Per-formula score breakdown |
| `total_score` | Maximum possible score for the problem |
| `formula_scoring` | Structured rubric listing each formula and its point value |

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

Then edit `.env`:

```
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1   # or your provider's endpoint
```

The scripts load these automatically via `os.environ`.

## Grading Pipeline

The grading pipeline has two steps:

### Step 1 — Extract scoring rubric from reference answers

`grading/extract_rubric.py` calls an LLM to parse a reference solution and produce a structured formula-level rubric.

**JSON input** (e.g. from `dataset/training.json`):
```bash
python grading/extract_rubric.py dataset/training.json output_with_rubrics.json
```

**Excel input** (column `solution_grading`):
```bash
python grading/extract_rubric.py input.xlsx output.xlsx
```

Output adds a `formulas` field to each entry containing the structured rubric.

### Step 2 — Score student answers against rubric

`grading/score.py` takes an Excel file with a student answer column and a rubric column, grades each answer using an LLM, and writes the result with a `scoreresult` column containing the per-formula breakdown and a `\boxed{N}` total.

```bash
python grading/score.py input.xlsx output.xlsx
```

Required Excel columns:
- `answer` — student's response
- `score-standard` — scoring rubric (output of Step 1)

### Dataset utilities

`grading/dataset_utils.py` provides helpers for working with the JSON dataset:

```bash
# Add formula_scoring field extracted from $$ ... $$ in answers
python grading/dataset_utils.py dataset/training.json output.json
```

## Repository Structure

```
physics_bench_repo/
├── dataset/
│   └── test.md        # Example of test set
│   └── training.json        # Benchmark problems
├── grading/
│   ├── score.py             # Step 2: LLM-based grading
│   ├── extract_rubric.py    # Step 1: Extract formula rubric from reference solution
│   └── dataset_utils.py     # Dataset inspection and preprocessing utilities
├── requirements.txt
├── .env.example
└── .gitignore
```


