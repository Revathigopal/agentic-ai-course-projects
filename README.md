# Agentic AI engineering: three scaffolded projects

Companion code for a local-first course on building reliable LLM agents with Pydantic v2 and Ollama. The three projects are ordered so that each adds one idea.

| # | Project | The agent | New idea |
| --- | --- | --- | --- |
| 1 | Enquiry Triage Agent | labels customer enquiries and drafts replies | structured output, validation, retry; a code-level escalation policy |
| 2 | Invoice-to-Ledger | extracts invoice fields and posts a journal entry | never trust model arithmetic: recompute, reconcile, and post only what balances |
| 3 | Guarded Reorder Assistant | proposes purchase orders for low stock | guardrails between proposal and action, budgets, human approval |

All three share one function, `structured_call` in `src/agentic_course/core.py`: ask for JSON, validate it against a Pydantic schema, and send the validation error back to the model if it fails.

## Quick start

```bash
pip install -e ".[dev]"
agentic-course triage  examples/enquiries.txt
agentic-course ledger  examples/invoice_ok.txt examples/invoice_bad.txt
agentic-course reorder examples/stock.json
```

```text
[auto_reply] sales/normal: Hi, do you have the brass lamp in bulk? ...
[human     ] complaint/normal: My order ORD-88213 arrived damaged ...
             because complaints always go to a person

invoice_ok.txt: SBT/2026/118 from Sri Balaji Traders -> posted
  Dr  11,800.00  Purchases
  Dr     590.00  Input GST receivable
  Cr  12,390.00  Accounts payable - Sri Balaji Traders
invoice_bad.txt: KP-0457 from Kaveri Packaging -> needs_review
  problem: line 2: 10 x 220 = 2200, invoice says 2400

[needs_human_approval] 30 x LAMP-BRASS from Kumbakonam Metals = 13,500
[approved            ] 200 x WICK-COTTON from Madurai Cotton Co = 1,200
```

The businesses and figures in `examples/` are invented for the exercises.

## Running with a local model

```bash
ollama pull llama3.1
agentic-course triage examples/enquiries.txt --provider ollama --model llama3.1
```

Without `--provider`, each project uses an offline stand-in (`src/agentic_course/mock.py`) built from keyword rules and regular expressions. It is not a model; it exists so the exercises and tests run with no setup. A useful exercise is to run the same input through the stand-in and a real model and compare where the schema and the guardrails catch each.

## What the code decides, not the model

| Project | Rule enforced in code |
| --- | --- |
| Triage | complaints, high priority, large billing issues, and any draft that promises a refund or discount go to a person |
| Ledger | line totals, subtotal, GST and total are recomputed; duplicates are held; the journal must balance |
| Reorder | approved suppliers only, at most three months of cover, one order per SKU, a run budget, and human approval above a threshold |

## Exercises

1. Add a `language` field to `Triage` and route non-English enquiries to a person.
2. Extend `Invoice` to handle separate CGST and SGST lines, and post them to separate accounts.
3. Add a guardrail that rejects an order if the supplier's last delivery was late, using a new field on `StockItem`.

## Tests

```bash
pytest -q
```
