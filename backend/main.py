# backend/main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AzureOpenAI
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import azure.cognitiveservices.speech as speechsdk
import asyncio, threading, json
from datetime import datetime

# Environment variables

from dotenv import load_dotenv
import os

load_dotenv()

SPEECH_KEY         = os.getenv("SPEECH_KEY", "")
SPEECH_REGION      = os.getenv("SPEECH_REGION", "")
AZURE_OAI_KEY      = os.getenv("AZURE_OAI_KEY", "")
AZURE_OAI_ENDPOINT = os.getenv("AZURE_OAI_ENDPOINT", "")
AZURE_OAI_MODEL    = os.getenv("AZURE_OAI_MODEL", "gpt-4.1-mini")
LANGUAGE_KEY       = os.getenv("LANGUAGE_KEY", "")
LANGUAGE_ENDPOINT  = os.getenv("LANGUAGE_ENDPOINT", "")

# Clients
openai_client = AzureOpenAI(
    api_key=AZURE_OAI_KEY,
    azure_endpoint=AZURE_OAI_ENDPOINT,
    api_version="2024-12-01-preview",
)

language_client = TextAnalyticsClient(
    endpoint=LANGUAGE_ENDPOINT,
    credential=AzureKeyCredential(LANGUAGE_KEY),
)

# FastAPI app & CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session state
class SessionState:
    def reset(self):
        self.active           = False
        self.member           = ""
        self.targets:         list[str]      = []
        self.scoreboard:      dict[str, int] = {}
        self.qa_log:          list[dict]     = []
        self.recognizer       = None
        self.ws               = None
        self.loop             = None
        self.pron_scores:     list[dict]     = []
        self.full_transcript: list[str]      = []
    def __init__(self):
        self.reset()

state = SessionState()

# Helpers
def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")

async def _send(ws: WebSocket, event: str, data: dict):
    try:
        await ws.send_text(json.dumps({"event": event, **data}))
    except Exception:
        pass

def send_sync(event: str, data: dict):
    if state.ws and state.loop:
        asyncio.run_coroutine_threadsafe(
            _send(state.ws, event, data),
            state.loop,
        )

# PII masking
def mask_pii(text: str) -> str:
    try:
        result = language_client.recognize_pii_entities(
            documents=[{"id": "1", "language": "en", "text": text}]
        )[0]
        if result.is_error:
            return text
        masked = text
        for entity in sorted(result.entities, key=lambda e: e.offset, reverse=True):
            masked = (
                masked[: entity.offset]
                + f"[{entity.category.upper()}]"
                + masked[entity.offset + entity.length :]
            )
        return masked
    except Exception:
        return text

# Question detection filters
VOCAB_TRIGGERS = [
    "what is", "what are", "what does", "what do",
    "what's the difference between",
    "how do you say", "how do you use",
    "can you explain", "what does that mean",
    "meaning of", "definition of",
]
FILTER_OUT = [
    "i don't know if", "i don't know whether",
    "i don't know what to", "i don't know how to",
]

def _is_question(text: str) -> bool:
    norm = text.lower().strip()
    if any(f in norm for f in FILTER_OUT):
        return False
    if any(t in norm for t in VOCAB_TRIGGERS):
        return True
    if "i don't know" in norm and "?" in text:
        return True
    return False

# STT callback
def on_recognized(evt):
    if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
        return
    text = evt.result.text.strip()
    if not text:
        return

    state.full_transcript.append(text)
    send_sync("transcript", {"text": text, "time": _now()})

    norm = text.lower()
    for expr in state.targets:
        if expr in norm:
            state.scoreboard[expr] = state.scoreboard.get(expr, 0) + 1
            send_sync("target_hit", {
                "expr":  expr,
                "count": state.scoreboard[expr],
                "text":  text,
            })

    if _is_question(text):
        threading.Thread(target=answer_question, args=(text,), daemon=True).start()

    # Extract pronunciation assessment scores
    try:
        pron = speechsdk.PronunciationAssessmentResult(evt.result)
        entry = {
            "text":         text,
            "accuracy":     round(pron.accuracy_score,     1),
            "fluency":      round(pron.fluency_score,      1),
            "completeness": round(pron.completeness_score, 1),
            "pron_score":   round(pron.pronunciation_score,1),
        }
        state.pron_scores.append(entry)
        send_sync("pron_score", entry)
    except Exception:
        pass

# AI Tutor response
def answer_question(question: str):
    send_sync("qa_thinking", {"question": question})
    try:
        res = openai_client.chat.completions.create(
            model=AZURE_OAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise IELTS speaking tutor. "
                        "2-3 sentences max. "
                        "Vocabulary: meaning + 1 example. "
                        "Grammar: rule + example. "
                        "End with one encouraging line."
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_tokens=120,
        )
        answer = res.choices[0].message.content.strip()
        entry  = {"time": _now(), "question": question, "answer": answer}
        state.qa_log.append(entry)
        send_sync("qa_answer", entry)
    except Exception as e:
        send_sync("qa_error", {"error": str(e)})

# Pronunciation average calculator
def calc_pron_summary() -> dict:
    if not state.pron_scores:
        return {}
    n = len(state.pron_scores)
    return {
        "accuracy":     round(sum(p["accuracy"]     for p in state.pron_scores) / n, 1),
        "fluency":      round(sum(p["fluency"]      for p in state.pron_scores) / n, 1),
        "completeness": round(sum(p["completeness"] for p in state.pron_scores) / n, 1),
        "pron_score":   round(sum(p["pron_score"]   for p in state.pron_scores) / n, 1),
        "detail":       state.pron_scores,
    }

# Feedback builder — reused by /session/stop and /generate-feedback
def build_feedback(transcript: str, question: str, part: int, pron_summary: dict) -> dict:
    pron_text = json.dumps(pron_summary) if pron_summary else "Not available"
    pron_score_val = pron_summary.get("pron_score", 0) / 10 if pron_summary else 6.5
    try:
        res = openai_client.chat.completions.create(
            model=AZURE_OAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict but encouraging IELTS Speaking examiner. Return only valid JSON.",
                },
                {
                    "role": "user",
                    "content": f"""Evaluate this IELTS Speaking Part {part} response.

Question: {question}
Response: {transcript}

Azure Speech pronunciation scores (0-100):
{pron_text}

Return exactly this JSON:
{{
  "band_score": 6.5,
  "fluency_coherence": {{
    "score": 6.5,
    "comment": "2 sentence assessment",
    "strengths": ["strength 1", "strength 2"],
    "improvements": ["fix 1", "fix 2"]
  }},
  "lexical_resource": {{
    "score": 6.0,
    "comment": "2 sentence assessment",
    "good_expressions": ["expr 1", "expr 2"],
    "suggestions": ["replace X with Y"]
  }},
  "grammatical_range": {{
    "score": 6.5,
    "comment": "2 sentence assessment",
    "errors": [{{"original": "wrong", "corrected": "right", "rule": "brief rule"}}]
  }},
  "pronunciation": {{
    "score": {pron_score_val},
    "pron_score": {pron_summary.get("pron_score", 0) if pron_summary else 0},
    "accuracy_score": {pron_summary.get("accuracy", 0) if pron_summary else 0},
    "fluency_score": {pron_summary.get("fluency", 0) if pron_summary else 0},
    "completeness_score": {pron_summary.get("completeness", 0) if pron_summary else 0},
    "comment": "detailed pronunciation feedback based on the Azure scores above",
    "specific_improvements": ["tip 1", "tip 2"]
  }},
  "improved_response": "Full rewritten response at Band 7-8 level keeping same ideas",
  "key_phrases_to_practice": ["phrase 1", "phrase 2", "phrase 3"]
}}""",
                },
            ],
            max_tokens=900,
        )
        raw = res.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        return {"error": str(e)}

# Session start
class SessionStart(BaseModel):
    member:  str
    targets: list[str]

@app.post("/session/start")
async def session_start(body: SessionStart):
    state.reset()
    state.member  = body.member
    state.targets = [t.lower().strip() for t in body.targets]
    state.active  = True
    return {"status": "ok"}

# Session stop — PII mask transcript then auto-generate feedback
@app.post("/session/stop")
async def session_stop():
    if state.recognizer:
        state.recognizer.stop_continuous_recognition()
    state.active = False

    pron_summary = calc_pron_summary()
    full_text    = " ".join(state.full_transcript)

    # Mask PII from transcript
    masked_full = mask_pii(full_text) if full_text.strip() else ""

    # Auto-generate feedback using masked transcript
    auto_feedback = None
    if masked_full.strip():
        auto_feedback = build_feedback(
            transcript   = masked_full,
            question     = "IELTS Speaking session full transcript",
            part         = 2,
            pron_summary = pron_summary,
        )

    return {
        "member":          state.member,
        "scoreboard":      state.scoreboard,
        "qa_log":          state.qa_log,
        "pron_summary":    pron_summary,
        "auto_feedback":   auto_feedback,
        "full_transcript": masked_full,   # 마스킹된 버전 반환
    }

# PII masking endpoint
@app.post("/mask-pii")
async def mask_pii_endpoint(body: dict):
    text   = body.get("text", "")
    masked = mask_pii(text)
    return {"original": text, "masked": masked}

# Pronunciation score endpoint (real-time during session)
@app.get("/pron-summary")
async def get_pron_summary():
    return calc_pron_summary()

# Question generation
class QuestionRequest(BaseModel):
    part:  int = 2
    topic: str = ""

@app.post("/generate-question")
async def generate_question(req: QuestionRequest):
    part_info = {
        1: {"desc": "Part 1 — simple personal question, short answer expected.", "prep": 0,  "answer": 45},
        2: {"desc": "Part 2 — cue card, 1-2 minute talk, exactly 4 bullet points.", "prep": 15, "answer": 120},
        3: {"desc": "Part 3 — abstract discussion, opinion-based.", "prep": 0,  "answer": 90},
    }
    info      = part_info[req.part]
    topic_hint= f"Topic area: {req.topic}." if req.topic else "Choose a random IELTS topic."
    try:
        res = openai_client.chat.completions.create(
            model=AZURE_OAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an IELTS examiner. Return only valid JSON, no markdown."},
                {"role": "user",   "content": f"""Generate one IELTS Speaking question.
{info['desc']}
{topic_hint}

Return exactly this JSON:
{{
  "part": {req.part},
  "topic": "short topic name",
  "question": "full question text",
  "bullet_points": ["point 1","point 2","point 3","point 4"],
  "prep_seconds": {info['prep']},
  "answer_seconds": {info['answer']}
}}
For Part 1 and 3, bullet_points must be []."""},
            ],
            max_tokens=300,
        )
        raw = res.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        return {"error": str(e), "part": req.part, "topic": "", "question": f"Error: {e}",
                "bullet_points": [], "prep_seconds": 15, "answer_seconds": 120}

# Target expression generation
@app.post("/generate-targets")
async def generate_targets():
    try:
        res = openai_client.chat.completions.create(
            model=AZURE_OAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an IELTS tutor. Return only valid JSON."},
                {"role": "user",   "content": """Generate 5 natural IELTS Speaking Band 7+ target expressions.
Mix: discourse markers, collocations, idiomatic phrases.
Avoid clichés. Each usable mid-sentence naturally.

Return exactly: {"expressions": ["expr1","expr2","expr3","expr4","expr5"]}"""},
            ],
            max_tokens=120,
        )
        return json.loads(res.choices[0].message.content)
    except Exception:
        return {"expressions": [
            "in terms of", "make a huge difference",
            "boost my confidence", "on the other hand", "quite challenging",
        ]}

# Feedback generation (Get Feedback button)
class FeedbackRequest(BaseModel):
    transcript:   str
    question:     str
    part:         int  = 2
    pron_summary: dict = {}

@app.post("/generate-feedback")
async def generate_feedback(req: FeedbackRequest):
    # Mask PII then generate feedback
    masked = mask_pii(req.transcript) if req.transcript.strip() else req.transcript
    return build_feedback(
        transcript   = masked,
        question     = req.question,
        part         = req.part,
        pron_summary = req.pron_summary,
    )

# WebSocket — STT + Pronunciation Assessment
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.ws   = ws
    state.loop = asyncio.get_event_loop()

    cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    cfg.speech_recognition_language = "en-US"

    # Pronunciation assessment config
    pron_config = speechsdk.PronunciationAssessmentConfig(
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True,
    )

    audio = speechsdk.audio.AudioConfig(use_default_microphone=True)
    rec   = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio)
    pron_config.apply_to(rec)

    rec.recognized.connect(on_recognized)
    rec.start_continuous_recognition()
    state.recognizer = rec

    await _send(ws, "ready", {"message": "STT + Pronunciation Assessment ready"})

    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        rec.stop_continuous_recognition()
        state.ws   = None
        state.loop = None
