import os, httpx, traceback
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession
from datetime import datetime

BASE_DIR      = Path(__file__).resolve().parent.parent
ENV_PATH      = BASE_DIR / ".env"
FRONTEND_DIR  = BASE_DIR / "frontend"
STATIC_DIR    = FRONTEND_DIR / "static"
TEMPLATES_DIR = FRONTEND_DIR / "templates"

load_dotenv(dotenv_path=ENV_PATH)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

from backend.database import init_db, get_db
from backend.models   import Patient, Session as ChatSession, Message
from backend.knowledge import search_knowledge, format_knowledge_context

app = FastAPI(title="ORIANMed Phase 2")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.on_event("startup")
def startup():
    init_db()
    print("[ORIANMed] Phase 2 ready")

BASE_SYSTEM_PROMPT = """You are ORIANMed, a warm and helpful medical AI assistant.
Help patients understand their symptoms and guide them on next steps.

Rules:
1. Ask one clarifying question at a time — duration, severity 1-10, location.
2. Never give a definitive diagnosis — say "this may indicate" or "common causes include".
3. If emergency symptoms detected (chest pain, difficulty breathing, stroke, severe bleeding) — tell user to call 112 or 911 immediately.
4. Always recommend seeing a real doctor.
5. Use the patient profile and medical knowledge provided to personalize your response.

You are NOT a replacement for a doctor."""

EMERGENCY_KEYWORDS = [
    "chest pain","can't breathe","cannot breathe","difficulty breathing",
    "stroke","unconscious","heart attack","severe bleeding","suicidal",
    "overdose","not breathing","seizure","anaphylaxis","allergic reaction",
]

def check_emergency(text):
    return any(kw in text.lower() for kw in EMERGENCY_KEYWORDS)

class PatientCreate(BaseModel):
    name: str
    age: int | None = None
    gender: str | None = None
    conditions: str | None = None

class MessageIn(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[MessageIn]
    session_id: str | None = None
    patient_id: str | None = None

class ChatResponse(BaseModel):
    reply: str
    is_emergency: bool
    backend_used: str
    confidence: float
    session_id: str
    error: str = ""

def try_nvidia(messages):
    key = os.getenv("NVIDIA_NIM_API_KEY","").strip()
    if not key or not key.startswith("nvapi-"):
        raise RuntimeError("NVIDIA key invalid")
    with httpx.Client(verify=False, timeout=30) as c:
        r = c.post("https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
            json={"model":"meta/llama-3.1-8b-instruct","messages":messages,"temperature":0.3,"max_tokens":600})
    if r.status_code != 200:
        raise RuntimeError(f"NIM {r.status_code}: {r.text[:150]}")
    return r.json()["choices"][0]["message"]["content"]

def try_groq(messages):
    key = os.getenv("GROQ_API_KEY","").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    with httpx.Client(timeout=30) as c:
        r = c.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
            json={"model":"llama3-8b-8192","messages":messages,"temperature":0.3,"max_tokens":600})
    if r.status_code != 200:
        raise RuntimeError(f"Groq {r.status_code}: {r.text[:150]}")
    return r.json()["choices"][0]["message"]["content"]

def try_hf(messages):
    key = os.getenv("HF_API_KEY","").strip()
    headers = {"Content-Type":"application/json"}
    if key: headers["Authorization"] = f"Bearer {key}"
    prompt = BASE_SYSTEM_PROMPT + "\n\n"
    for m in messages:
        if m["role"]=="user": prompt += f"Patient: {m['content']}\n"
        elif m["role"]=="assistant": prompt += f"ORIANMed: {m['content']}\n"
    prompt += "ORIANMed:"
    for model in ["mistralai/Mistral-7B-Instruct-v0.2","HuggingFaceH4/zephyr-7b-beta"]:
        try:
            with httpx.Client(timeout=25) as c:
                r = c.post(f"https://api-inference.huggingface.co/models/{model}",
                    headers=headers,
                    json={"inputs":prompt,"parameters":{"max_new_tokens":300,"return_full_text":False}})
            if r.status_code == 200:
                data = r.json()
                if isinstance(data,list) and data:
                    text = data[0].get("generated_text","").strip()
                    if text: return text.split("Patient:")[0].strip()
        except Exception as e:
            print(f"[HF:{model}] {e}")
    raise RuntimeError("HuggingFace all models failed")

def call_ai(messages):
    for name, fn in [("NVIDIA NIM",try_nvidia),("Groq",try_groq),("HuggingFace",try_hf)]:
        try:
            print(f"[ORIANMed] trying {name}...")
            reply = fn(messages)
            print(f"[ORIANMed] success via {name}")
            return reply, name
        except Exception as e:
            print(f"[ORIANMed] {name} failed: {e}")
    raise RuntimeError("All AI backends failed")

def estimate_confidence(reply, knowledge_results):
    score = 0.70
    if knowledge_results: score += 0.15
    if any(p in reply.lower() for p in ["i'm not sure","unclear","difficult to say","consult"]): score -= 0.20
    if "cannot help" in reply.lower() or "don't know" in reply.lower(): score -= 0.30
    return round(min(max(score,0.1),1.0),2)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/patients")
async def create_patient(data: PatientCreate, db: DBSession = Depends(get_db)):
    patient = Patient(name=data.name, age=data.age, gender=data.gender, conditions=data.conditions)
    db.add(patient); db.commit(); db.refresh(patient)
    return {"patient_id": patient.id, "name": patient.name}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: DBSession = Depends(get_db)):
    last_msg  = req.messages[-1].content if req.messages else ""
    emergency = check_emergency(last_msg)

    session = None
    if req.session_id:
        session = db.query(ChatSession).filter(ChatSession.id == req.session_id).first()
    if not session:
        session = ChatSession(patient_id=req.patient_id or "anonymous")
        db.add(session); db.commit(); db.refresh(session)

    patient_context = ""
    if req.patient_id and req.patient_id != "anonymous":
        patient = db.query(Patient).filter(Patient.id == req.patient_id).first()
        if patient:
            parts = []
            if patient.name:       parts.append(f"Name: {patient.name}")
            if patient.age:        parts.append(f"Age: {patient.age}")
            if patient.gender:     parts.append(f"Gender: {patient.gender}")
            if patient.conditions: parts.append(f"Known conditions: {patient.conditions}")
            if parts: patient_context = "Patient profile:\n" + "\n".join(parts)

    knowledge_results = search_knowledge(last_msg, top_k=3)
    knowledge_context = format_knowledge_context(knowledge_results)

    system_parts = [BASE_SYSTEM_PROMPT]
    if patient_context:   system_parts.append(patient_context)
    if knowledge_context: system_parts.append(knowledge_context)
    system_prompt = "\n\n".join(system_parts)

    messages = [{"role":"system","content":system_prompt}]
    messages += [{"role":m.role,"content":m.content} for m in req.messages]

    try:
        reply, backend = call_ai(messages)
        confidence     = estimate_confidence(reply, knowledge_results)
        error          = ""
    except Exception as e:
        reply = f"ALL BACKENDS FAILED.\n\n{str(e)}"
        backend = "none"; confidence = 0.0; error = str(e)

    db.add(Message(session_id=session.id, role="user", content=last_msg,
                   is_emergency=emergency, timestamp=datetime.utcnow()))
    if backend != "none":
        db.add(Message(session_id=session.id, role="assistant", content=reply,
                       confidence=confidence, backend_used=backend,
                       is_emergency=emergency, timestamp=datetime.utcnow()))
    db.commit()

    return ChatResponse(reply=reply, is_emergency=emergency, backend_used=backend,
                        confidence=confidence, session_id=session.id, error=error)

@app.get("/history/{session_id}")
async def get_history(session_id: str, db: DBSession = Depends(get_db)):
    messages = db.query(Message).filter(Message.session_id == session_id)\
        .order_by(Message.timestamp).all()
    return [{"role":m.role,"content":m.content,"confidence":m.confidence,
             "timestamp":m.timestamp.isoformat() if m.timestamp else None} for m in messages]

@app.get("/health")
async def health():
    nim = os.getenv("NVIDIA_NIM_API_KEY","")
    groq = os.getenv("GROQ_API_KEY","")
    return {"status":"ORIANMed Phase 2 running","phase":2,
            "features":["memory","knowledge_base","confidence_scoring","patient_profiles"],
            "nvidia_key_valid":nim.startswith("nvapi-"),"groq_key_set":bool(groq),
            "qdrant_connected":bool(os.getenv("QDRANT_URL","")),"db":"SQLite"}

@app.get("/test-nim")
async def test_nim():
    key = os.getenv("NVIDIA_NIM_API_KEY","").strip()
    if not key: return {"error":"not set"}
    try:
        with httpx.Client(verify=False,timeout=15) as c:
            r = c.post("https://integrate.api.nvidia.com/v1/chat/completions",
                headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
                json={"model":"meta/llama-3.1-8b-instruct",
                      "messages":[{"role":"user","content":"Say hi."}],"max_tokens":20})
        return {"status_code":r.status_code,"body":r.json() if r.status_code==200 else r.text}
    except Exception as e:
        return {"error":str(e)}

@app.get("/test-groq")
async def test_groq():
    key = os.getenv("GROQ_API_KEY","").strip()
    if not key: return {"error":"not set"}
    try:
        with httpx.Client(timeout=15) as c:
            r = c.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
                json={"model":"llama3-8b-8192",
                      "messages":[{"role":"user","content":"Say hi."}],"max_tokens":20})
        return {"status_code":r.status_code,"body":r.json() if r.status_code==200 else r.text}
    except Exception as e:
        return {"error":str(e)}