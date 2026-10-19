# MedSage（睿医）— AI-Powered Clinical Decision Support Agent

**MedSage（睿医）** is a production-grade, multi-agent clinical decision-support system for rural and low-resource healthcare environments. It combines conversational triage, multi-step diagnosis, voice and vision workflows, retrieval-augmented medical knowledge, auditability, and operational tooling into one full-stack application.

> **About this project.** MedSage is a **secondary development** built on top of an existing healthcare AI demo. The original codebase was primarily "model + prompts + a web page". This project closes the gap between a **demo** and a **genuine enterprise-grade agent system** by adding the engineering depth that real agents require: task planning, context engineering, observability, evaluation, and human-in-the-loop — rather than simply wiring a model behind a UI.

---

## What This Secondary Development Adds

The enhancements below are the core of this project, and are deliberately highlighted because they are what turn a chatbot into a real agent.

### 1. Multi-Agent Orchestration (Task Planning)
- **LangGraph state machine** with conditional routing: emergency bypass, image analysis, and lung-sound nodes are only executed when their input exists (`backend/app/services/agents/graph.py`).
- **Autonomous ReAct loop** with a tool harness: tool registration, retry with fallback, per-tool observations cache, and sync-tool execution via thread pool (`backend/app/services/agents/loop.py`, `harness.py`).

### 2. Context Engineering
- **Patient long-term memory injection**: allergies and chronic conditions are auto-loaded from the patient profile and merged into the diagnosis context (`backend/app/api/v1/endpoints/agents.py`).
- **Server-side conversation history fallback**: the chat endpoint recovers recent session history from the database when the client loses context, with a rolling context window (`backend/app/api/v1/endpoints/chat.py`).

### 3. Pluggable LLM Backend (Domestic Model Support)
- The LLM client previously hardcoded Google Gemini. It is now **provider-agnostic** and supports any OpenAI-compatible endpoint — Alibaba Qwen (通义千问), DeepSeek, Zhipu GLM, and more — via environment configuration (`backend/app/services/llm/gemini_client.py`, `backend/app/core/config.py`). No VPN required.

### 4. Evaluation & Observability
- Built an **offline evaluation harness** (`backend/scripts/eval_agent.py` + `backend/scripts/eval_cases.json`) that measures **diagnosis accuracy**, **triage consistency**, and **confidence** against a labeled case set — so quality is quantified, not assumed.
- Existing observability already includes Sentry, Prometheus, LangSmith, structured logging, and a per-decision audit table (`model_name`, `model_version`, `prompt_version`, `confidence_band`).

### 5. Human-in-the-Loop
- AI decisions are recorded in an audit trail with **clinician override** and **feedback** paths, and medication safety checks route risky cases to "consult a clinician" (`backend/app/api/v1/endpoints/audit.py`, `medications.py`).

---

## Evaluation Results

Baseline run against the built-in 12-case evaluation set (`qwen-plus`):

| Metric | Result |
| --- | --- |
| Diagnosis accuracy | **100%** (12 / 12) |
| Triage consistency | **75%** (9 / 12) |
| Mean confidence | 0.87 |

The three triage mismatches all lean toward the **safer, more conservative** side (e.g. escalation of suspected anaphylaxis and influenza), which is the desired behavior in a clinical setting. Full per-case output is available in `backend/eval_report.json`.

Reproduce it with:

```bash
cd backend
python scripts/eval_agent.py --list-cases   # view the labeled cases
python scripts/eval_agent.py --out eval_report.json
```

---

## Core Features

- **Conversational Triage** — guided, symptom-first patient assessment.
- **Multi-Agent Diagnosis** — symptom analysis, diagnosis, treatment planning, emergency action, and report generation.
- **Live Voice Consultations** — real-time spoken intake and spoken diagnosis for accessibility.
- **Vision Workflows** — chest X-ray and medical image analysis.
- **RAG Medical Workspace** — retrieval-augmented search over indexed medical documents.
- **Evidence Grounding** — trusted source catalog (WHO, CDC, NICE) with per-response citation metadata.
- **Clinical Operations** — follow-up planning, scheduling, export, offline-sync endpoints.
- **Safety & Governance** — medication safety checks, audit trails, clinician override feedback.

---

## Evidence Grounding Architecture

Medical answers are grounded through a multi-layer retrieval strategy before generation:

1. **Trusted Source Catalog** — PostgreSQL (`medical_evidence_sources`).
2. **Local Knowledge Base** — uploaded documents chunked with `pgvector` (`medical_documents`).
3. **Reputable Public Sources** — PubMed, MedlinePlus, OpenFDA, ClinicalTrials.

Every diagnosis and RAG response carries citation metadata — provider, source URL, excerpt, similarity score, evidence level, and verification timestamps — for full auditability.

---

## Tech Stack

| Layer | Stack |
| --- | --- |
| **Frontend** | Next.js, React, TypeScript, Tailwind CSS, Radix UI, Zustand, React Query |
| **Backend** | FastAPI, SQLAlchemy, Pydantic Settings, Uvicorn, Gunicorn |
| **Data** | PostgreSQL, `pgvector`, Alembic, Redis-compatible cache |
| **AI Layer** | LangChain, LangGraph, LangSmith, pluggable LLM (Gemini / Qwen / DeepSeek / GLM) |
| **Media / Vision** | OpenCV, Pillow, PyMuPDF, pydub, gTTS, ElevenLabs |
| **Observability** | Prometheus, structured logging, Sentry |
| **Testing** | Pytest, Jest, React Testing Library |

---

## Project Architecture

```text
MedSage/
├── backend/
│   ├── alembic/         # Database migrations
│   ├── app/
│   │   ├── api/         # FastAPI route handlers
│   │   ├── core/        # Config, security, setup
│   │   ├── db/          # SQLAlchemy models & engine
│   │   ├── schemas/     # Pydantic schemas
│   │   └── services/    # Business logic (agents, RAG, voice, vision)
│   ├── scripts/         # Init scripts + eval harness (eval_agent.py, eval_cases.json)
│   ├── eval_report.json # Latest evaluation results
│   └── tests/           # Backend API and workflow coverage
├── frontend/
│   ├── public/          # Assets and images
│   └── src/             # App router, components, hooks, store
└── docs/                # Architecture and design documentation
```

---

## Screenshots

**Welcome & feature access**

![Welcome page](frontend/public/WelcomePage.png)
![Feature access](frontend/public/WelcomePage_EndPoints_Paths.png)

**Dashboard & diagnosis**

![User dashboard](frontend/public/User_Dashboard.png)
![Multi-agent diagnosis system](frontend/public/Diagnosis_System.png)
![Diagnosis result](frontend/public/Result_of_Diagnosis.png)

**Knowledge / RAG, vision & operations**

![RAG assistant](frontend/public/Knowledge_RAG_Assistant.png)
![Chest X-ray analysis](frontend/public/Chest_X_Ray_Analysis.png)
![Clinical ops console](frontend/public/Clinical_Ops_Console.png)
![Admin audit logs](frontend/public/Admin_Dashboard_Audit_Logs.png)
![Bias monitoring](frontend/public/AdminDashboard_BiasMonitoring.png)

---

## Local Development Guide

### Prerequisites
- Python 3.11+
- Node.js 20+
- PostgreSQL 16 with `pgvector`
- npm

### 1. Database Setup
```bash
cd backend
docker-compose up -d
```

### 2. Environment Variables
```bash
cd backend
copy .env.example .env
```

Configure the minimal set: `DATABASE_URL`, `SECRET_KEY`, and one LLM backend.

**Option A — Domestic OpenAI-compatible model (recommended, no VPN):**
```
LLM_PROVIDER=openai
LLM_API_KEY=sk-xxxx
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
```

**Option B — Google Gemini:**
```
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-google-ai-studio-key
```

### 3. Backend Setup
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Initialize Database
```bash
python scripts/init_db.py
```

### 5. Run the Services
```bash
# Backend API
cd backend
uvicorn app.main:app --reload          # http://127.0.0.1:8000, docs at /docs

# Frontend
cd frontend
npm install
npm run dev                            # http://localhost:3000
```

---

## Testing & Evaluation

**Backend tests:**
```bash
cd backend
pytest
```
Covers clinical triage, medication safety, sync conflict handling, and audit paths.

**Agent quality evaluation:**
```bash
cd backend
python scripts/eval_agent.py --out eval_report.json
```

**Frontend:**
```bash
cd frontend
npm test
npm run test:e2e
```

---

## Safety & Compliance

This is a **clinical decision-support application** and is **NOT** a replacement for licensed medical judgment. Any real-world deployment must undergo rigorous clinical validation, privacy/security review (HIPAA/GDPR), and appropriate regulatory safeguards.

---

## Resume / Portfolio Value

This project demonstrates:

- **Full-stack product engineering** across modern frontend, backend, and data layers.
- **Real agent capability**, not prompt chaining — task planning, tool use with recovery, context engineering, and evaluation.
- **Production-grade engineering** — observability, caching, rate limiting, audit trails, and human-in-the-loop.
- **Pluggable AI backend** that works with both global and domestic LLM providers.
- **Healthcare domain modeling** with strict grounding, auditability, and safety controls.
- **Quantified quality** — an evaluation harness backed by measurable accuracy metrics.