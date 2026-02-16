# StereoWipe Project Summary

A concise overview of the StereoWipe stereotyping evaluation benchmark, combining key information from the main documentation, CLI guidance, and deployment instructions.

## 1. Introduction
StereoWipe is a comprehensive evaluation suite for assessing stereotyping in Large Language Models (LLMs). It uses an LLM-as-a-Judge paradigm to detect and score stereotypical content in model responses, producing structured JSON reports and offering a web-based viewer for interactive analysis.

## 2. Project Structure
- **biaswipe/**: Core Python package
  - `cli.py` – Command-line interface for running evaluations
  - `data_loader.py` – Loads prompts, annotations, and model responses
  - `judge.py` – Abstract Judge classes (Mock, OpenAI, Anthropic) with caching
  - `scoring.py` – Orchestrates evaluation pipeline
  - `metrics.py` – Calculates SR, SSS, CSSS, and WOSI metrics
  - `report.py` – Generates JSON report files
- **biaswipe_viewer/**: Flask web application
  - `webserver.py` – Starts the viewer on port 5000
  - `templates/` – HTML templates (Bootstrap)
  - `static/` – CSS and assets
- **sample_data/**: Example JSON data for prompts, annotations, responses, and weights
- **tests/**: Pytest unit tests for core modules
- **config & schema**: `judge_prompt.txt`, `json_output_schema.json`

## 3. Getting Started
1. Clone the repository and `cd stereowipe`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Set API keys (if using OpenAI or Anthropic judges):
   ```bash
   export OPENAI_API_KEY="<key>"
   export ANTHROPIC_API_KEY="<key>"
   ```

## 4. Evaluation CLI
```bash
python biaswipe/cli.py \
  --prompts sample_data/prompts.json \
  --annotations sample_data/annotations.json \
  --model-responses-dir sample_data/model_responses \
  --category-weights sample_data/category_weights.json \
  --judges openai,mock \
  --report-output report.json
```
- **SR**: Stereotype Rate
- **SSS**: Stereotype Severity Score
- **CSSS**: Category-Specific Severity
- **WOSI**: Weighted Overall Index

## 5. Web Viewer
```bash
python biaswipe_viewer/webserver.py
``` 
Visit `http://127.0.0.1:5000/` to upload or view `report.json` interactively.

## 6. Deployment (Google Cloud Run)
### Prerequisites
- Google Cloud project with billing enabled
- Authenticated `gcloud` CLI
- APIs enabled: Cloud Build, Run, Container Registry

### Automatic
```bash
export PROJECT_ID=<project-id>
cd stereowipe
gcloud builds submit --config cloudbuild.yaml
``` 
Builds, pushes, and deploys to Cloud Run in `us-central1`.

### Manual
```bash
docker build -t gcr.io/$PROJECT_ID/biaswipe-viewer biaswipe_viewer
  && docker push gcr.io/$PROJECT_ID/biaswipe-viewer

gcloud run deploy biaswipe-viewer \
  --image gcr.io/$PROJECT_ID/biaswipe-viewer \
  --platform managed --region us-central1 \
  --allow-unauthenticated --memory 1Gi --cpu 1 --timeout 300 --max-instances 10 --port 8080
```

### Environment Variables & Logs
```bash
# Set keys
gcloud run services update biaswipe-viewer --set-env-vars OPENAI_API_KEY=<key>,ANTHROPIC_API_KEY=<key>
# View logs
gcloud run services logs read biaswipe-viewer --region us-central1
# Get URL:
gcloud run services describe biaswipe-viewer --region us-central1 --format 'value(status.url)'
```

## 7. Continuous Deployment
Connect GitHub repo to Cloud Build and trigger on `main` branch pushes using `cloudbuild.yaml`.

---
*For detailed guidance, refer to `README.md`, `CLAUDE.md`, and `DEPLOYMENT.md` in the project root.*
