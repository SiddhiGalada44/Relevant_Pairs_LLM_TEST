# Relevant Priors

A Django REST API that predicts which previous radiology exams are relevant comparisons for a current study. It uses a fast keyword heuristic for clear cases and falls back to Gemini 2.0 Flash for ambiguous ones.

## How it works

1. **Heuristic pass** — maps study descriptions to body regions (head, chest, spine, etc.) via keyword matching. If regions overlap → relevant. If no overlap → not relevant.
2. **LLM pass** — only ambiguous cases (unrecognized anatomy or unclear overlap) are batched into a single Gemini prompt.
3. **Cache** — SHA-256 keyed in-memory cache prevents duplicate LLM calls for identical study pairs.

## API

### `POST /predict`

**Request**
```json
{
  "cases": [
    {
      "case_id": "abc123",
      "current_study": {
        "study_description": "CT CHEST WITH CONTRAST",
        "study_date": "2024-01-15"
      },
      "previous_examinations": [
        {
          "study_id": "s001",
          "study_description": "CT CHEST",
          "study_date": "2023-06-10",
          "report_text": "No acute findings..."
        },
        {
          "study_id": "s002",
          "study_description": "MRI BRAIN",
          "study_date": "2023-03-01",
          "report_text": "Normal study..."
        }
      ]
    }
  ]
}
```

**Response**
```json
{
  "predictions": [
    { "case_id": "abc123", "study_id": "s001", "predicted_is_relevant": true },
    { "case_id": "abc123", "study_id": "s002", "predicted_is_relevant": false }
  ]
}
```

## Setup

**Requirements:** Python 3.13+, a Gemini API key

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
```

```bash
python manage.py runserver
```

## Docker

```bash
docker build -t relevant-priors .
docker run -p 8080:8080 -e GEMINI_API_KEY=your_key_here relevant-priors
```

## Deploy (Render)

Configured via `render.yaml`. Set `GEMINI_API_KEY` as an environment variable in the Render dashboard — it is intentionally not synced in the config file.
