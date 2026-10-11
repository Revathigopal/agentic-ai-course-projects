"""Offline rule-based stand-ins for a language model, one per project.

These are NOT models. They use keyword rules and regular expressions so the
exercises run and are testable with no network. Switch to Ollama (or another
provider) to see how a real model behaves against the same schemas.
"""

from __future__ import annotations

import json
import re

from .llm import Message


def _triage(text: str) -> dict:
    lowered = text.lower()
    rules = [
        ("spam", ("lottery", "click here", "crypto", "seo services", "winner")),
        ("complaint", ("terrible", "worst", "unacceptable", "angry", "complain", "damaged", "rude")),
        ("billing", ("invoice", "charged", "refund", "payment", "billed", "gst")),
        ("support", ("not working", "how do i", "help", "error", "broken", "install")),
        ("sales", ("price", "quote", "buy", "bulk", "catalogue", "available")),
    ]
    category = next((name for name, words in rules if any(word in lowered for word in words)), "support")
    priority = "high" if any(word in lowered for word in ("urgent", "immediately", "today", "legal")) else "normal"
    order = re.search(r"\b(?:order|ord)[ #:-]*([A-Z0-9-]{4,})", text, flags=re.IGNORECASE)
    amount = re.search(r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    return {
        "category": category,
        "priority": priority,
        "order_number": order.group(1) if order else None,
        "amount": float(amount.group(1).replace(",", "")) if amount else None,
        "draft_reply": "Thank you for writing to us. We have noted your message and will get back to you shortly.",
    }


def _number(text: str) -> float:
    return float(text.replace(",", ""))


def _invoice(text: str) -> dict:
    def field(pattern: str) -> str:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            raise ValueError(f"mock extractor could not find /{pattern}/")
        return match.group(1).strip()

    items = []
    for match in re.finditer(r"^(.+?)\s+(\d+(?:\.\d+)?)\s*x\s*([\d,.]+)\s*=\s*([\d,.]+)\s*$", text, flags=re.MULTILINE):
        items.append(
            {
                "description": match.group(1).strip(),
                "quantity": float(match.group(2)),
                "unit_price": _number(match.group(3)),
                "amount": _number(match.group(4)),
            }
        )
    gst = re.search(r"GST\s*@?\s*([\d.]+)\s*%\s*:?\s*([\d,.]+)", text, flags=re.IGNORECASE)
    return {
        "vendor": field(r"^Vendor:\s*(.+)$"),
        "invoice_number": field(r"^Invoice (?:No|Number)\.?:\s*(\S+)"),
        "invoice_date": field(r"^Date:\s*(\d{4}-\d{2}-\d{2})"),
        "items": items,
        "subtotal": _number(field(r"^Subtotal:\s*([\d,.]+)")),
        "gst_rate_pct": float(gst.group(1)) if gst else 0.0,
        "gst_amount": _number(gst.group(2)) if gst else 0.0,
        "total": _number(field(r"^Total:\s*([\d,.]+)")),
    }


def _reorder(text: str) -> dict:
    orders = []
    for item in json.loads(text):
        if item["stock"] <= item["reorder_point"]:
            orders.append(
                {
                    "sku": item["sku"],
                    "supplier": item["suppliers"][0],
                    "quantity": item["monthly_sales"],
                    "reason": f"Stock {item['stock']} is at or below reorder point {item['reorder_point']}.",
                }
            )
    return {"orders": orders}


HANDLERS = {"enquiry_triage": _triage, "invoice_to_ledger": _invoice, "guarded_reorder": _reorder}


def mock_handler(messages: list[Message]) -> str:
    task = messages[0]["content"].splitlines()[0].removeprefix("TASK:").strip()
    return json.dumps(HANDLERS[task](messages[1]["content"]))
