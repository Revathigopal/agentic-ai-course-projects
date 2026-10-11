"""Project 2 - Invoice-to-Ledger.

The model reads invoice text and extracts the fields. Code then re-does all
the arithmetic. Only an invoice that reconciles is turned into a balanced
double-entry journal; anything else is held for review.

Concepts practised: extraction into nested schemas, never trusting model
arithmetic, and producing an accounting entry that must balance.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .core import structured_call
from .llm import LLM

INSTRUCTIONS = (
    "Extract the invoice fields exactly as printed. Do not calculate, correct or round anything; "
    "copy the numbers shown. gst_rate_pct is the tax percentage printed on the invoice."
)
TOLERANCE = 0.5  # rupees; invoices are often rounded to the nearest rupee


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LineItem(Strict):
    description: str
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    amount: float = Field(ge=0)


class Invoice(Strict):
    vendor: str = Field(min_length=1)
    invoice_number: str = Field(min_length=1)
    invoice_date: date
    items: list[LineItem] = Field(min_length=1)
    subtotal: float = Field(ge=0)
    gst_rate_pct: float = Field(ge=0, le=40)
    gst_amount: float = Field(ge=0)
    total: float = Field(ge=0)


class JournalLine(Strict):
    account: str
    debit: float = 0.0
    credit: float = 0.0


class LedgerResult(Strict):
    invoice: Invoice
    status: Literal["posted", "needs_review"]
    problems: list[str]
    journal: list[JournalLine]


def reconcile(invoice: Invoice) -> list[str]:
    """Recompute every figure on the invoice and list what does not add up."""
    problems = []
    for index, item in enumerate(invoice.items, start=1):
        expected = item.quantity * item.unit_price
        if abs(expected - item.amount) > TOLERANCE:
            problems.append(f"line {index}: {item.quantity:g} x {item.unit_price:g} = {expected:g}, invoice says {item.amount:g}")
    line_sum = sum(item.amount for item in invoice.items)
    if abs(line_sum - invoice.subtotal) > TOLERANCE:
        problems.append(f"lines add up to {line_sum:g}, subtotal says {invoice.subtotal:g}")
    expected_gst = invoice.subtotal * invoice.gst_rate_pct / 100
    if abs(expected_gst - invoice.gst_amount) > TOLERANCE:
        problems.append(f"GST at {invoice.gst_rate_pct:g}% should be {expected_gst:g}, invoice says {invoice.gst_amount:g}")
    if abs(invoice.subtotal + invoice.gst_amount - invoice.total) > TOLERANCE:
        problems.append(f"subtotal + GST = {invoice.subtotal + invoice.gst_amount:g}, total says {invoice.total:g}")
    return problems


def journal_for(invoice: Invoice) -> list[JournalLine]:
    """Purchase entry: debit expense and input tax, credit the vendor."""
    lines = [JournalLine(account="Purchases", debit=round(invoice.subtotal, 2))]
    if invoice.gst_amount:
        lines.append(JournalLine(account="Input GST receivable", debit=round(invoice.gst_amount, 2)))
    lines.append(JournalLine(account=f"Accounts payable - {invoice.vendor}", credit=round(invoice.subtotal + invoice.gst_amount, 2)))
    return lines


def is_balanced(journal: list[JournalLine]) -> bool:
    return abs(sum(line.debit for line in journal) - sum(line.credit for line in journal)) < 0.005


def process_invoice(llm: LLM, invoice_text: str, seen_invoice_numbers: set[str] | None = None) -> LedgerResult:
    invoice = structured_call(llm, "invoice_to_ledger", INSTRUCTIONS, invoice_text, Invoice)
    problems = reconcile(invoice)
    key = f"{invoice.vendor}|{invoice.invoice_number}".lower()
    if seen_invoice_numbers is not None:
        if key in seen_invoice_numbers:
            problems.append(f"invoice {invoice.invoice_number} from {invoice.vendor} was already processed")
        seen_invoice_numbers.add(key)
    if problems:
        return LedgerResult(invoice=invoice, status="needs_review", problems=problems, journal=[])
    journal = journal_for(invoice)
    assert is_balanced(journal)
    return LedgerResult(invoice=invoice, status="posted", problems=[], journal=journal)
