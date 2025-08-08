# Bajaj Document Q&A API

A FastAPI-based document processing and question-answering system powered by Google Cloud Vertex AI (Gemini).

## Features

- 📄 **Document Processing**: Supports PDF, DOCX, XLSX, CSV, HTML, PPT, and email formats
- 🤖 **AI-Powered Q&A**: Uses Google Gemini 1.5 Flash for intelligent document analysis
- 🔒 **Secure Authentication**: Bearer token-based API security
- 📊 **Structured Responses**: Returns detailed decisions with confidence scores and evidence
- ⚡ **Fast Processing**: Optimized document parsing and token management
- 🌐 **Production Ready**: Docker support and cloud deployment ready

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /test_gemini` - Test Gemini AI connectivity
- `POST /test_without_gemini` - Test API structure with mock data
- `POST /structured_qa` - Main document Q&A with structured responses
- `POST /hackrx/run` - Alternative Q&A format with detailed explanations

## Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (see `.env.example`)
4. Configure Google Cloud credentials
5. Run: `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`

## Environment Variables

```
GEMINI_PROJECT_ID=your-gcp-project-id
GEMINI_REGION=asia-south1
API_TOKEN=your-api-token
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
MAX_TOKENS=32000
```

## Documentation

API documentation available at: `http://localhost:8000/docs`

## License

MIT License
