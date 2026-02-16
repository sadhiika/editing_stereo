# Google Cloud Run Deployment Guide

This guide explains how to deploy the BiasWipe Viewer to Google Cloud Run.

## Prerequisites

1. Google Cloud Project with billing enabled
2. gcloud CLI installed and authenticated
3. Enable required APIs:
   ```bash
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable run.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```

## Automatic Deployment (Using Cloud Build)

1. Set your project ID:
   ```bash
   export PROJECT_ID=your-project-id
   gcloud config set project $PROJECT_ID
   ```

2. Submit the build:
   ```bash
   gcloud builds submit --config cloudbuild.yaml
   ```

This will automatically:
- Build the Docker image
- Push it to Google Container Registry
- Deploy to Cloud Run in us-central1

## Manual Deployment

1. Build and push the Docker image:
   ```bash
   docker build -t gcr.io/$PROJECT_ID/biaswipe-viewer .
   docker push gcr.io/$PROJECT_ID/biaswipe-viewer
   ```

2. Deploy to Cloud Run:
   ```bash
   gcloud run deploy biaswipe-viewer \
     --image gcr.io/$PROJECT_ID/biaswipe-viewer \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 1Gi \
     --cpu 1 \
     --timeout 300 \
     --max-instances 10 \
     --port 8080
   ```

## Configuration

The deployment is configured with:
- **Memory**: 1GB
- **CPU**: 1 vCPU
- **Timeout**: 300 seconds
- **Max Instances**: 10
- **Min Instances**: 0 (scales to zero)
- **Port**: 8080

## Environment Variables

You can set environment variables for API keys:

```bash
gcloud run services update biaswipe-viewer \
  --set-env-vars OPENAI_API_KEY=your-key,ANTHROPIC_API_KEY=your-key \
  --region us-central1
```

## Viewing Logs

```bash
gcloud run services logs read biaswipe-viewer --region us-central1
```

## Getting the Service URL

```bash
gcloud run services describe biaswipe-viewer --region us-central1 --format 'value(status.url)'
```

## Continuous Deployment

To set up continuous deployment from GitHub:

1. Connect your GitHub repository to Cloud Build
2. Create a trigger that runs on push to main branch
3. The cloudbuild.yaml will handle the rest

## Notes

- The viewer expects a `report.json` file to be present. You'll need to generate this using the BiasWipe CLI first.
- For production use, consider:
  - Adding authentication
  - Setting up a custom domain
  - Configuring CORS if needed
  - Adding monitoring and alerting