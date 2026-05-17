#!/usr/bin/env python3
"""
LLM Benchmark Runner
Testa tool calling, brain writing, multi-step agentic tasks
su modelli diversi via OpenAI-compat API.

Usage:
    python3 runner.py --case cases/01-create-person.json --model sonnet
    python3 runner.py --all --model all
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import urllib.request
import urllib.error

# === MODELLI ===
MODELS = {
    "sonnet": {
        "base_url": "https://api.anthropic.com",
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "model": "claude-sonnet-4-6-20251101",
        "format": "anthropic",
    },
    "opus": {
        "base_url": "https://api.anthropic.com",
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "model": "claude-opus-4-5-20251101",
        "format": "anthropic",
    },
    "qwen-runpod": {
        "base_url": os.getenv("RUNPOD_OLLAMA_URL", "https://altmr0kkq06hml-11434.proxy.runpod.net"),
        "api_key": "dummy",
        "model": "qwen2.5:14b",
        "format": "openai",
    },
    "openrouter-llama": {
        "base_url": "https://openrouter.ai/api",
        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "model": "meta-llama/llama-3.3-70b-instruct",
        "format": "openai",
    },
    "litellm": {
        "base_url": os.getenv("LITELLM_URL", "http://localhost:4000"),
        "api_key": os.getenv("LITELLM_KEY", "village-test-key"),
        "model": os.getenv("LITELLM_MODEL", "claude-sonnet-4-6"),
        "format": "openai",
    },
}

# === TOOL DEFINITIONS (subset brain tools) ===
TOOLS = [
    {
        "name": "create_entity",
        "description": "Crea un'entità nel brain (person, company, project, diary, log, todo)",
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["person", "company", "project", "diary", "log", "todo"]},
                "name": {"type": "string"},
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "project": {"type": "string"},
                "deadline": {"type": "string"},
                "deadline_type": {"type": "string", "enum": ["hard", "soft"]},
            },
            "required": ["type", "name"],
        },
    },
    {
        "name": "read_file",
        "description": "Leggi un file nel brain",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "update_file",
        "description": "Aggiorna un file esistente nel brain (append o replace)",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "mode": {"type": "string", "enum": ["append", "replace"]},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "search_brain",
        "description": "Cerca file nel brain per keyword o pattern",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "path": {"type": "string", "description": "Percorso in cui cercare (default: tutto il brain)"},
            },
            "required": ["query"],
        },
    },
]

# === CANNED TOOL RESULTS ===
# Risposte simulate per ogni tool call. Keyed by (tool_name, path/name).
CANNED_RESULTS = {
    ("read_file", "wiki/projects/generations-ats/index.md"): """---
type: project
status: active
client: Generations Recruitment
stack: laravel,postgresql,pgvector
---
# Generations ATS
MVP in sviluppo. Phase 1: replace Hiresweet.
Candidati, pipeline, email integrata.
Deadline: fine maggio 2026.
Contatto: Yoni Saroussi (CTO).
""",
    ("read_file", "wiki/projects/emibrain/index.md"): """---
type: project
status: active
client: Emisfera
host: emisrvbrn01
ip: 192.168.8.210
---
# EMI Brain
VM Proxmox Emisfera. ABChat installato on-prem.
Utenti attivi: Paola, Simone, Daniele.
Ultimo issue: scrollAuth fix 2026-05-11.
""",
    ("search_brain", "HireSuite"): """Trovati 3 file:
- wiki/projects/generations-hiresuite-integration/index.md
- diary/2026/2026-05-04-grw-call-operativa-piero-docs-headgren-poc-yoni.md
- wiki/projects/generations-ats/index.md
""",
    ("search_brain", "Yoni"): """Trovati 4 file:
- wiki/people/yoni-saroussi.md
- diary/2026/2026-05-04-grw-call-operativa.md
- wiki/projects/generations-ats/index.md
- todo/2026-05-10-preventivo-yoni.md
""",
}

def canned_result(tool_name: str, input_data: dict) -> str:
    """Restituisce risultato simulato per un tool call."""
    path = input_data.get("path", "") or input_data.get("query", "") or input_data.get("name", "")
    key = (tool_name, path)
    if key in CANNED_RESULTS:
        return CANNED_RESULTS[key]

    # Fallback generico per create_entity
    if tool_name == "create_entity":
        etype = input_data.get("type", "entity")
        name = input_data.get("name", "unknown")
        return f"✅ Creato {etype}: {name}"

    if tool_name == "update_file":
        return f"✅ File aggiornato: {input_data.get('path', '?')}"

    return f"✅ {tool_name} eseguito (simulato). Input: {json.dumps(input_data, ensure_ascii=False)}"


def call_anthropic(cfg: dict, messages: list, system: str) -> dict:
    """Chiama API Anthropic (formato nativo)."""
    payload = {
        "model": cfg["model"],
        "max_tokens": 2048,
        "system": system,
        "tools": TOOLS,
        "messages": messages,
    }
    req = urllib.request.Request(
        f"{cfg['base_url']}/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": cfg["api_key"],
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def call_openai(cfg: dict, messages: list, system: str) -> dict:
    """Chiama API OpenAI-compat."""
    oai_tools = []
    for t in TOOLS:
        oai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        })

    oai_messages = [{"role": "system", "content": system}] + messages
    payload = {
        "model": cfg["model"],
        "max_tokens": 2048,
        "tools": oai_tools,
        "tool_choice": "auto",
        "messages": oai_messages,
    }
    req = urllib.request.Request(
        f"{cfg['base_url']}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['api_key']}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def normalize_response(raw: dict, fmt: str) -> dict:
    """Normalizza risposta a formato comune: {text, tool_calls, stop_reason}"""
    if fmt == "anthropic":
        tool_calls = []
        text = ""
        for block in raw.get("content", []):
            if block["type"] == "text":
                text += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append({"id": block["id"], "name": block["name"], "input": block["input"]})
        return {"text": text, "tool_calls": tool_calls, "stop_reason": raw.get("stop_reason")}
    else:  # openai
        choice = raw["choices"][0]
        msg = choice["message"]
        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            tool_calls.append({
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": json.loads(tc["function"]["arguments"]),
            })
        return {"text": msg.get("content") or "", "tool_calls": tool_calls, "stop_reason": choice["finish_reason"]}


def run_case(case: dict, model_key: str) -> dict:
    """Esegue un caso di test su un modello. Gestisce il loop tool calling."""
    cfg = MODELS[model_key]
    system = case.get("system", "Sei un assistente AI che gestisce un brain personale in markdown.")
    user_message = case["prompt"]

    messages = [{"role": "user", "content": user_message}]
    tool_calls_log = []
    turns = 0
    max_turns = 8
    start = time.time()

    while turns < max_turns:
        turns += 1
        try:
            if cfg["format"] == "anthropic":
                raw = call_anthropic(cfg, messages, system)
            else:
                raw = call_openai(cfg, messages, system)
        except Exception as e:
            return {"error": str(e), "tool_calls": tool_calls_log, "turns": turns, "elapsed": time.time() - start}

        resp = normalize_response(raw, cfg["format"])

        if not resp["tool_calls"]:
            # Niente tool call → risposta finale
            return {
                "final_text": resp["text"],
                "tool_calls": tool_calls_log,
                "turns": turns,
                "elapsed": round(time.time() - start, 2),
                "stop_reason": resp["stop_reason"],
            }

        # Esegui tool calls (simulati)
        for tc in resp["tool_calls"]:
            result = canned_result(tc["name"], tc["input"])
            tool_calls_log.append({"name": tc["name"], "input": tc["input"], "result": result})

            # Aggiungi al contesto (formato dipende dal modello)
            if cfg["format"] == "anthropic":
                if turns == 1 or messages[-1]["role"] != "assistant":
                    messages.append({"role": "assistant", "content": [
                        {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]}
                    ]})
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": tc["id"], "content": result}
                ]})
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["input"])}}
                ]})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    return {
        "error": f"max_turns ({max_turns}) raggiunto",
        "tool_calls": tool_calls_log,
        "turns": turns,
        "elapsed": round(time.time() - start, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="Path al file JSON del caso")
    parser.add_argument("--all", action="store_true", help="Esegui tutti i casi in cases/")
    parser.add_argument("--model", default="sonnet", help=f"Modello: {', '.join(MODELS)} o 'all'")
    args = parser.parse_args()

    cases_dir = Path(__file__).parent.parent / "cases"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = results_dir / run_id
    run_dir.mkdir()

    model_keys = list(MODELS.keys()) if args.model == "all" else [args.model]
    case_files = sorted(cases_dir.glob("*.json")) if args.all else [Path(args.case)]

    summary = []
    for case_file in case_files:
        case = json.loads(case_file.read_text())
        print(f"\n{'='*60}")
        print(f"CASO: {case.get('id')} — {case.get('name')}")

        for model_key in model_keys:
            print(f"  → {model_key}...", end=" ", flush=True)
            result = run_case(case, model_key)
            elapsed = result.get("elapsed", "?")
            n_tools = len(result.get("tool_calls", []))
            ok = "✓" if "final_text" in result else "✗"
            print(f"{ok} {elapsed}s, {n_tools} tool calls")

            out = {"case": case, "model": model_key, "result": result, "run_id": run_id}
            out_file = run_dir / f"{case_file.stem}-{model_key}.json"
            out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False))
            summary.append({"case": case.get("id"), "model": model_key, "ok": ok, "elapsed": elapsed, "tools": n_tools})

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nRisultati in: {run_dir}")


if __name__ == "__main__":
    main()
