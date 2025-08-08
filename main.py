import os
import time
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up Google Cloud credentials
google_creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
google_creds_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')

if google_creds_json:
    # Create credentials file from JSON environment variable
    import tempfile
    import json as json_lib
    
    try:
        # Parse the JSON string to validate it
        if isinstance(google_creds_json, str):
            creds_data = json_lib.loads(google_creds_json)
        else:
            creds_data = google_creds_json
            
        # Create a temporary file for credentials
        temp_creds_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json_lib.dump(creds_data, temp_creds_file)
        temp_creds_file.close()
        
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_creds_file.name
        print(f"Created Google Cloud credentials file from environment variable")
        
    except Exception as e:
        print(f"Warning: Failed to process GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")
        
elif google_creds_path:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_creds_path

import aiohttp
from aiolimiter import AsyncLimiter
import tiktoken
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import vertexai
from vertexai.generative_models import GenerativeModel

# Import document processing libraries with error handling
try:
    # Only import what's absolutely necessary to reduce memory
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.docx import partition_docx
    from unstructured.partition.text import partition_text
    from unstructured.partition.html import partition_html
    # Import other formats on-demand to save memory
    UNSTRUCTURED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Unstructured library not fully available: {e}")
    UNSTRUCTURED_AVAILABLE = False
    # Define fallback functions
    def partition_pdf(*args, **kwargs): raise ImportError("PDF processing not available")
    def partition_docx(*args, **kwargs): raise ImportError("DOCX processing not available")
    def partition_text(*args, **kwargs): raise ImportError("Text processing not available")
    def partition_html(*args, **kwargs): raise ImportError("HTML processing not available")

# Init Vertex AI
PROJECT_ID = os.getenv("GEMINI_PROJECT_ID")
REGION = os.getenv("GEMINI_REGION", "asia-south1")

if not PROJECT_ID:
    print("Warning: GEMINI_PROJECT_ID not set. Vertex AI functionality will be limited.")
    
try:
    vertexai.init(project=PROJECT_ID, location=REGION)
    print(f"Vertex AI initialized with project: {PROJECT_ID}, region: {REGION}")
except Exception as e:
    print(f"Warning: Vertex AI initialization failed: {e}")

ENCODING = "cl100k_base"
tokenizer = tiktoken.get_encoding(ENCODING)
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "32000"))

# Auth setup
security = HTTPBearer(auto_error=True)
API_TOKEN = os.getenv("API_TOKEN")

if not API_TOKEN:
    raise ValueError("API_TOKEN environment variable is required")

async def check_token(
    creds: HTTPAuthorizationCredentials = Depends(security)
):
    if creds.scheme.lower() != "bearer" or creds.credentials != API_TOKEN:
        raise HTTPException(401, "Invalid or missing token")

app = FastAPI(title="Document Q&A API", version="2.0.0")

# Add startup event to debug issues
@app.on_event("startup")
async def startup_event():
    print("=== BAJAJ API STARTUP ===")
    print(f"Project ID: {PROJECT_ID}")
    print(f"Region: {REGION}")
    print(f"API Token set: {bool(API_TOKEN)}")
    print(f"Max Tokens: {MAX_TOKENS}")
    print(f"Google Credentials: {bool(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))}")
    print("=== STARTUP COMPLETE ===")

rate_limiter = AsyncLimiter(max_rate=10, time_period=1.0)

class QARequest(BaseModel):
    documents: str
    questions: List[str]

class Decision(BaseModel):
    decision: str
    amount: str
    type: str
    coverage_section: str
    verbatim: str
    confidence: str

class StructuredQAResponse(BaseModel):
    decisions: List[Decision]
    document_info: Dict[str, Any]
    processing_time: float
    questions_processed: int

class HealthResponse(BaseModel):
    status: str
    strategy: str
    top_k: int
    timestamp: float

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    toks = tokenizer.encode(text)
    if len(toks) <= max_tokens:
        return text
    return tokenizer.decode(toks[:max_tokens])

def get_ext(path: str) -> str:
    return Path(path.split("?",1)[0]).suffix.lower()

async def download_file(url: str, max_mb: int = 100) -> str:
    async with rate_limiter:
        ext = get_ext(url)
        dst = f"/tmp/doc_{int(time.time())}{ext}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status!=200:
                    raise HTTPException(400,f"Download failed {resp.status}")
                data = await resp.read()
        with open(dst,"wb") as f:
            f.write(data)
        return dst

def parse_doc(path: str) -> str:
    ext = get_ext(path)
    
    try:
        if ext==".pdf": 
            elems=partition_pdf(path,strategy="fast")
        elif ext in (".docx",".doc"): 
            elems=partition_docx(path)
        elif ext in (".pptx",".ppt"): 
            # Lazy import to save memory
            from unstructured.partition.pptx import partition_pptx
            elems=partition_pptx(path)
        elif ext in (".xlsx",".xls"): 
            # Lazy import to save memory
            from unstructured.partition.xlsx import partition_xlsx
            elems=partition_xlsx(path)
        elif ext==".csv": 
            # Lazy import to save memory
            from unstructured.partition.csv import partition_csv
            elems=partition_csv(path)
        elif ext in (".txt",".rtf"): 
            elems=partition_text(path)
        elif ext in (".html",".htm"): 
            elems=partition_html(path)
        elif ext in (".eml",".msg"): 
            # Lazy import to save memory
            from unstructured.partition.email import partition_email
            elems=partition_email(path)
        else: 
            # Lazy import to save memory
            from unstructured.partition.auto import partition
            elems=partition(path)
            
    except ImportError as e:
        raise HTTPException(400, f"Document processing not available for {ext} files: {str(e)}")
    except Exception as e:
        raise HTTPException(400, f"Failed to process document: {str(e)}")
        
    if not elems: 
        raise HTTPException(400,"No content extracted")
    txt = "\n\n".join(str(e) for e in elems if str(e).strip())
    txt = "\n".join(l.strip() for l in txt.splitlines() if l.strip())
    return txt

async def call_gemini(ctx: str, qs: List[str]) -> Tuple[List[Decision], List[str]]:
    system = """
You are an expert insurance analyst.
Extract JSON per question: decision,amount,type,coverage_section,verbatim,rationale,confidence.
Return ONLY an array, in order.
Document:
{ctx}

Questions:
{qs}
"""
    prompt=system.format(ctx=ctx,qs="\n".join(f"{i+1}. {q}" for i,q in enumerate(qs)))
    model=GenerativeModel("gemini-1.5-flash")
    resp=model.generate_content(prompt,generation_config={"max_output_tokens":4096,"temperature":0.1,"response_mime_type":"application/json"})
    try:
        arr=json.loads(resp.text)
    except:
        arr=[{"decision":"Error","amount":"Not specified","type":"error","coverage_section":"System","verbatim":"","rationale":"","confidence":"low"}]*len(qs)
    decs,rats=[],[]
    for it in arr:
        decs.append(Decision(
            decision=it.get("decision","") or "",
            amount=it.get("amount","") or "",
            type=it.get("type","") or "",
            coverage_section=it.get("coverage_section","") or "",
            verbatim=it.get("verbatim","") or "",
            confidence=it.get("confidence","") or ""
        ))
        rats.append(it.get("rationale","") or "")
    return decs,rats

@app.get("/health",response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy",strategy="fast",top_k=0,timestamp=time.time())

@app.get("/test")
async def test():
    return {
        "message": "API is working",
        "project_id": PROJECT_ID,
        "region": REGION,
        "max_tokens": MAX_TOKENS,
        "api_token_set": bool(API_TOKEN)
    }

@app.post("/test_gemini", dependencies=[Depends(check_token)])
async def test_gemini(request: dict):
    """Test Gemini AI without document processing"""
    try:
        questions = request.get("questions", ["What is 2+2?"])
        sample_text = "This is a test document. It contains basic information for testing purposes."
        
        decs, rats = await call_gemini(sample_text, questions)
        return {
            "message": "Gemini test successful",
            "decisions": [{"decision": d.decision, "confidence": d.confidence} for d in decs],
            "rationales": rats
        }
    except Exception as e:
        raise HTTPException(500, f"Gemini test failed: {str(e)}")

@app.post("/test_without_gemini", dependencies=[Depends(check_token)])
async def test_without_gemini(req: QARequest):
    """Test API structure without Gemini AI - returns mock data"""
    try:
        t0 = time.time()
        
        # Mock processing
        mock_decisions = []
        for i, question in enumerate(req.questions):
            mock_decisions.append(Decision(
                decision=f"Mock decision for question {i+1}",
                amount="$1000",
                type="mock_insurance",
                coverage_section="Mock Section A",
                verbatim=f"Mock evidence for: {question}",
                confidence="high"
            ))
        
        return StructuredQAResponse(
            decisions=mock_decisions,
            document_info={"ext": get_ext(req.documents), "tokens": 100},
            processing_time=time.time() - t0,
            questions_processed=len(req.questions)
        )
    except Exception as e:
        raise HTTPException(500, f"Error in mock test: {str(e)}")

@app.post("/structured_qa",response_model=StructuredQAResponse,dependencies=[Depends(check_token)])
async def structured_qa(req: QARequest):
    try:
        t0=time.time()
        path=await download_file(req.documents)
        txt=parse_doc(path)
        ctx=truncate_to_tokens(txt,MAX_TOKENS)
        decs,_=await call_gemini(ctx,req.questions)
        return StructuredQAResponse(
            decisions=decs,
            document_info={"ext":get_ext(req.documents),"tokens":count_tokens(ctx)},
            processing_time=time.time()-t0,
            questions_processed=len(req.questions)
        )
    except Exception as e:
        raise HTTPException(500, f"Error processing request: {str(e)}")

@app.post("/hackrx/run",dependencies=[Depends(check_token)])
async def hackrx_run(req: QARequest):
    try:
        t0=time.time()
        path=await download_file(req.documents)
        txt=parse_doc(path)
        ctx=truncate_to_tokens(txt,MAX_TOKENS)
        decs,rats=await call_gemini(ctx,req.questions)
        answers=[]
        for d,r in zip(decs,rats):
            parts=[]
            if d.decision: parts.append(f"Decision: {d.decision}")
            if d.amount: parts.append(f"Amount: {d.amount}")
            if d.type: parts.append(f"Type: {d.type}")
            if d.coverage_section: parts.append(f"Section: {d.coverage_section}")
            if d.verbatim: parts.append(f'Evidence: "{d.verbatim}"')
            if r: parts.append(f"Rationale: {r}")
            if d.confidence: parts.append(f"Confidence: {d.confidence}")
            answers.append("; ".join(parts))
        return {"answers":answers}
    except Exception as e:
        raise HTTPException(500, f"Error processing request: {str(e)}")
