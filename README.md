# LLM Benchmark — Tool Calling & Brain Writing

Benchmark suite per confrontare modelli LLM su task reali di un brain personale (Village/ABChat).
Focus: tool calling, brain writing, multi-step agentic, error recovery.

## Struttura

```
cases/          # 10 casi di test in JSON
harness/        # runner.py — esegue i casi su N modelli
results/        # output grezzi per valutazione umana (gitignored)
```

## Casi

| ID | Nome | Categoria |
|----|------|-----------|
| 01 | Crea contatto persona | tool-calling-single |
| 02 | Crea diary da riunione | brain-writing |
| 03 | Aggiorna stato progetto (read→update) | multi-step |
| 04 | Todo con deadline hard | structured-output |
| 05 | Cerca prima di creare | tool-calling-strategy |
| 06 | Leggi e riassumi senza inventare | read-then-reason |
| 07 | Tre azioni in un messaggio | multi-step-complex |
| 08 | Recovery da errore tool | error-handling |
| 09 | Stop quando ha finito | behavioral |
| 10 | Sintesi settimanale multi-read | multi-read-synthesis |

## Modelli configurati

- `sonnet` — Claude Sonnet 4.6 (Anthropic API)
- `opus` — Claude Opus (Anthropic API)
- `qwen-runpod` — Qwen2.5 14B via Ollama su RunPod
- `openrouter-llama` — Llama 3.3 70B via OpenRouter
- `litellm` — qualsiasi modello via LiteLLM proxy

## Uso

```bash
# Singolo caso, singolo modello
python3 harness/runner.py --case cases/01-create-person.json --model sonnet

# Tutti i casi, tutti i modelli
python3 harness/runner.py --all --model all

# Risultati in results/YYYYMMDD-HHMMSS/
```

## Env vars

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENROUTER_API_KEY=sk-or-...
export RUNPOD_OLLAMA_URL=https://{pod-id}-11434.proxy.runpod.net
export LITELLM_URL=http://100.123.97.7:4000
export LITELLM_KEY=village-test-key
```

## Valutazione

Ogni caso ha `eval_criteria` — checklist per la valutazione umana.
Output grezzo in `results/{run_id}/{caso}-{modello}.json`.
