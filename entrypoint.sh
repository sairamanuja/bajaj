#!/bin/bash
set -e

# Handle Google Cloud credentials in container
if [ ! -z "$GOOGLE_APPLICATION_CREDENTIALS_JSON" ]; then
    echo "Creating Google Cloud credentials file from environment variable..."
    mkdir -p /app/credentials
    echo "$GOOGLE_APPLICATION_CREDENTIALS_JSON" | base64 -d > /app/credentials/service-account-key.json
    export GOOGLE_APPLICATION_CREDENTIALS="/app/credentials/service-account-key.json"
    echo "Google Cloud credentials file created successfully"
elif [ ! -z "$GOOGLE_APPLICATION_CREDENTIALS" ] && [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "Using existing Google Cloud credentials file: $GOOGLE_APPLICATION_CREDENTIALS"
else
    echo "Warning: No Google Cloud credentials found. Vertex AI functionality may not work."
fi

# Validate required environment variables
if [ -z "$API_TOKEN" ]; then
    echo "Error: API_TOKEN environment variable is required"
    exit 1
fi

if [ -z "$GEMINI_PROJECT_ID" ]; then
    echo "Error: GEMINI_PROJECT_ID environment variable is required"
    exit 1
fi

echo "Starting Bajaj Document Q&A API..."
echo "Project ID: $GEMINI_PROJECT_ID"
echo "Region: $GEMINI_REGION"
echo "Max Tokens: $MAX_TOKENS"

# Start the application
exec "$@"
