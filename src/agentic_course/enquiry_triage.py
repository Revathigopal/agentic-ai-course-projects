"""Project 1 - Enquiry Triage Agent.

A small business receives enquiries by email and chat. The agent labels each
one, sets a priority and drafts a reply. Code, not the model, decides which
enquiries a person must handle.

Concepts practised: structured output, schema validation, retry on error,
and a deterministic escalation policy on top of model output.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .core import structured_call
from .llm import LLM

INSTRUCTIONS = (
    "You triage customer enquiries for a small business. Choose one category and a priority, "
    "extract any order number and any amount of money mentioned, and draft a short, polite reply. "
    "Do not promise refunds, discounts or delivery dates in the draft."
)


class Triage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal["sales", "support", "billing", "complaint", "spam"]
    priority: Literal["low", "normal", "high"]
    order_number: str | None = None
    amount: float | None = Field(default=None, ge=0)
    draft_reply: str = Field(min_length=10)


class TriageDecision(BaseModel):
    triage: Triage
    handler: Literal["auto_reply", "human", "discard"]
    reasons: list[str]


FORBIDDEN_PROMISES = ("refund", "discount", "guarantee", "compensat")


def decide(triage: Triage, refund_review_amount: float = 2000) -> TriageDecision:
    """Escalation policy. Every rule here is checkable without a model."""
    reasons = []
    if triage.category == "spam":
        return TriageDecision(triage=triage, handler="discard", reasons=["classified as spam"])
    if triage.category == "complaint":
        reasons.append("complaints always go to a person")
    if triage.priority == "high":
        reasons.append("high priority")
    if triage.category == "billing" and (triage.amount or 0) >= refund_review_amount:
        reasons.append(f"billing issue of {triage.amount:g} is at or above the {refund_review_amount:g} review limit")
    promised = [word for word in FORBIDDEN_PROMISES if word in triage.draft_reply.lower()]
    if promised:
        reasons.append(f"draft makes a promise the agent may not make ('{promised[0]}')")
    return TriageDecision(triage=triage, handler="human" if reasons else "auto_reply", reasons=reasons)


def triage_enquiry(llm: LLM, enquiry: str, refund_review_amount: float = 2000) -> TriageDecision:
    result = structured_call(llm, "enquiry_triage", INSTRUCTIONS, enquiry, Triage)
    return decide(result, refund_review_amount)
