"""Command-line interface: ``agentic-course triage|ledger|reorder <file>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .core import StructuredOutputError
from .enquiry_triage import triage_enquiry
from .guarded_reorder import Policy, StockItem, run_reorder
from .invoice_to_ledger import process_invoice
from .llm import LLMError, get_llm
from .mock import mock_handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentic-course", description="Run one of the three course projects.")
    parser.add_argument("project", choices=["triage", "ledger", "reorder"])
    parser.add_argument("files", nargs="+", type=Path, help="Input file(s) for the project.")
    parser.add_argument("--provider", help="mock (default), ollama, groq, openrouter or openai.")
    parser.add_argument("--model")
    args = parser.parse_args(argv)

    try:
        llm = get_llm(args.provider, args.model, mock_handler=mock_handler)
        if args.project == "triage":
            for path in args.files:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    decision = triage_enquiry(llm, line)
                    t = decision.triage
                    print(f"[{decision.handler:<10}] {t.category}/{t.priority}: {line[:70]}")
                    for reason in decision.reasons:
                        print(f"             because {reason}")
        elif args.project == "ledger":
            seen: set[str] = set()
            for path in args.files:
                result = process_invoice(llm, path.read_text(encoding="utf-8"), seen)
                print(f"{path.name}: {result.invoice.invoice_number} from {result.invoice.vendor} -> {result.status}")
                for problem in result.problems:
                    print(f"  problem: {problem}")
                for line in result.journal:
                    side = f"Dr {line.debit:>10,.2f}" if line.debit else f"Cr {line.credit:>10,.2f}"
                    print(f"  {side}  {line.account}")
        else:
            data = json.loads(args.files[0].read_text(encoding="utf-8"))
            items = [StockItem.model_validate(item) for item in data["items"]]
            run = run_reorder(llm, items, Policy.model_validate(data["policy"]))
            for d in run.decisions:
                print(f"[{d.status:<20}] {d.order.quantity} x {d.order.sku} from {d.order.supplier} = {d.cost:,.0f}")
                for reason in d.reasons:
                    print(f"                       {reason}")
            print(f"Approved spend {run.approved_spend:,.0f}; awaiting approval {run.pending_spend:,.0f}")
            if run.missed:
                print("Still low with no order: " + ", ".join(run.missed))
    except (LLMError, StructuredOutputError, ValidationError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
