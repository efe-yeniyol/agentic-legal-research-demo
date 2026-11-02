# Northbridge Legal Research Agent

**MIS311 — Agentic AI System Engineering (1)**  
İstinye University · Midterm Project  

**Instructor:** Dr. Öğr. Üy. Ömer Ozan Evkaya  

> **Not legal advice.** Fictional Northbridge corpus only.

---

## İki çalıştırma seçeneği (en önemli fark)

In this demo, the **same task** can be run with two different engines. Tools (`search_corpus`, `read_section`, `verify_citation`), safety rules, and Type B variants (baseline / agent v1 / agent v2) are shared. What changes is **who chooses the next step**.

| | **Deterministic** (default) | **OpenAI** (optional) |
|--|-----------------------------|------------------------|
| **What?** | Coded policy agent: picks the next tool from the prompt and tool results using rules | OpenAI Chat Completions + **tool calling** in a real ReAct loop; the model chooses the next tool |
| **API key** | Not required | `OPENAI_API_KEY` in `.env` (`OPENAI_MODEL`, default `gpt-4o-mini`) |
| **When to use** | Offline demo, reproducible instructor scores, no network | Live LLM comparison, “real agent” experience |
| **Output stability** | Same prompt → same steps and answer | May vary by model / version |
| **Write-up table** | Yes — reported pass rates come from here | No — extra experiment; does not replace the scores |
| **UI / CLI** | Methodology: *Deterministic* · `--method deterministic` | Methodology: *OpenAI* · `--method openai` |

In short: **Deterministic = primary, measurable midterm result.** **OpenAI = second option to try the same architecture with a live model.**

---

## 1. Purpose of the assignment

This midterm reviews published evidence on **agentic AI** systems in a chosen area (law / legal research) and compares them with simpler, non-agentic LLM applications. It then builds a small demo that supports the write-up.

**Focus question:**  
*How reliable are agentic legal research systems that search, cite and draft, and when do they add value over a single LLM call?*

Classification follows Anthropic (2024) *Building effective agents*:

| Type | Definition |
|------|------------|
| **LLM application** | Single model call; steps are set by the user / code |
| **Workflow** | Multiple calls / tools; order is fixed in code |
| **Agent** | The model chooses the next step and tool; usually in a loop |

---

## 2. Rules and warnings

- **Not legal advice.** Demo outputs are not legal advice.
- The corpus is **fictional** (Republic of Northbridge). No real client data, real brand imitation, or commercial case-law text.
- The agent **cannot file** with a court / tribunal; filing requests stop for human approval.
- Requests for threat letters, asset concealment, or medical dosing are refused.
- For the literature section you **must open and read every source you cite** (academic integrity).
- AI Assessment Scale (Perkins et al., 2024): AI collaboration for the literature; AI exploration for the demo. Details are in the AI-use statement in the write-up.

---

## 3. Methodology details and variants

In addition to the table above:

**How Deterministic works**  
`agent_deterministic.py` extracts statute IDs / keywords from the user text, then builds a `search_corpus` → `read_section` → `verify_citation` path from observations when needed. On filing / unsafe / medical requests it stops without calling tools. The midterm test suite can be reproduced exactly with no API key or internet.

**How OpenAI works**  
`agent_openai.py` gives the model the same three tool schemas; the model emits `tool_calls`, the runner executes the tool, and the result is sent back to the model (ReAct). The v2 system prompt forbids inventing sources. If there is no key, the UI/CLI returns a clear error and points you to deterministic mode.

**Type B variants** (available under both methodologies):

| Variant | Role |
|---------|------|
| `baseline` | Single answer, no tools (LLM-application style) |
| `agent_v1` | Tool loop; may invent after a tool error (literature failure mode) |
| `agent_v2` | Recommended; if no source → “could not be verified”; unclear goals → clarification |

---

## 4. Setup

```bash
cd agentic
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Optional: put OPENAI_API_KEY=sk-... in .env
```

---

## 5. Running

### CLI — Type B test suite (deterministic)

```bash
PYTHONPATH=src python -m northbridge.runner --method deterministic
```

Summary scores and log: `results/test_results_deterministic.json`

### CLI — OpenAI suite (API key required)

```bash
PYTHONPATH=src python -m northbridge.runner --method openai
```

### Web UI

```bash
PYTHONPATH=src uvicorn app.main:app --reload --port 8000
```

Browser: [http://127.0.0.1:8000](http://127.0.0.1:8000)

Select methodology + variant on the left and run a prompt; on the right you see the answer, step timeline, and tool calls.

---

## 6. Project structure

```text
agentic/
  README.md                 ← this file
  midterm-writeup.md        ← midterm write-up (APA)
  .env.example
  requirements.txt
  app/
    main.py                 ← FastAPI
    static/index.html       ← simple modern UI
  src/northbridge/
    corpus.json / cases.json
    tools.py
    baseline.py
    agent_deterministic.py
    agent_openai.py
    runner.py
  results/                  ← test outputs (gitignored)
```

---

## 7. Deterministic suite summary (primary reported result)

Same 12 cases (gold answers written in advance):

| System | Pass rate | Avg steps | Avg tool calls |
|--------|-----------|-----------|----------------|
| Baseline | 33.3% | 1.00 | 0.00 |
| Agent v1 | 83.3% | 3.08 | 2.67 |
| Agent v2 | 100% | 2.42 | 1.92 |

---

## License / course use

Coursework for MIS311. Not for production legal advice. Fictional corpus © course project authors for educational use only.
