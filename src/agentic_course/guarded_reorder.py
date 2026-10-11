"""Project 3 - Guarded Reorder Assistant.

The agent looks at stock levels and proposes purchase orders. It can spend
money, so every proposal passes through guardrails before it becomes an
order, and large orders wait for a person.

Concepts practised: an agent that proposes actions, guardrails as plain code
between proposal and execution, human approval, and an audit trail.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .core import structured_call
from .llm import LLM

INSTRUCTIONS = (
    "You manage reordering for a shop. For each item whose stock is at or below its reorder point, "
    "propose one purchase order from one of the item's listed suppliers. Order enough to cover about "
    "one month of sales. Do not propose orders for items above their reorder point."
)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StockItem(Strict):
    sku: str
    stock: int = Field(ge=0)
    reorder_point: int = Field(ge=0)
    monthly_sales: int = Field(gt=0)
    unit_cost: float = Field(gt=0)
    suppliers: list[str] = Field(min_length=1)


class ProposedOrder(Strict):
    sku: str
    supplier: str
    quantity: int = Field(gt=0)
    reason: str = Field(min_length=5)


class Proposals(Strict):
    orders: list[ProposedOrder]


class Policy(Strict):
    budget: float = Field(gt=0, description="Total spend allowed in this run.")
    max_months_of_cover: float = 3.0
    approval_threshold: float = Field(default=10000, description="Orders costing more than this need a person.")


class Decision(Strict):
    order: ProposedOrder
    cost: float
    status: Literal["approved", "needs_human_approval", "rejected"]
    reasons: list[str]


class ReorderRun(Strict):
    decisions: list[Decision]
    approved_spend: float
    pending_spend: float
    missed: list[str] = Field(description="Items at or below reorder point with no usable proposal.")


def review(proposals: Proposals, items: list[StockItem], policy: Policy) -> ReorderRun:
    """Guardrails. Orders are considered in the order proposed; the budget is first come, first served."""
    by_sku = {item.sku: item for item in items}
    decisions: list[Decision] = []
    committed = 0.0  # approved + awaiting approval both reserve budget
    ordered: set[str] = set()

    for order in proposals.orders:
        item = by_sku.get(order.sku)
        if item is None:
            decisions.append(Decision(order=order, cost=0.0, status="rejected", reasons=[f"unknown SKU '{order.sku}'"]))
            continue
        cost = round(order.quantity * item.unit_cost, 2)
        reasons = []
        if order.sku in ordered:
            reasons.append("a second order for the same SKU in one run")
        if item.stock > item.reorder_point:
            reasons.append(f"stock of {item.stock} is above the reorder point of {item.reorder_point}")
        if order.supplier not in item.suppliers:
            reasons.append(f"'{order.supplier}' is not an approved supplier for {order.sku}")
        max_quantity = int(policy.max_months_of_cover * item.monthly_sales)
        if order.quantity > max_quantity:
            reasons.append(f"quantity {order.quantity} exceeds {policy.max_months_of_cover:g} months of sales ({max_quantity})")
        if not reasons and committed + cost > policy.budget:
            reasons.append(f"cost {cost:g} would exceed the remaining budget of {policy.budget - committed:g}")

        if reasons:
            status = "rejected"
        elif cost > policy.approval_threshold:
            status, reasons = "needs_human_approval", [f"cost {cost:g} is above the approval threshold of {policy.approval_threshold:g}"]
        else:
            status = "approved"
        if status != "rejected":
            committed += cost
            ordered.add(order.sku)
        decisions.append(Decision(order=order, cost=cost, status=status, reasons=reasons))

    missed = [item.sku for item in items if item.stock <= item.reorder_point and item.sku not in ordered]
    return ReorderRun(
        decisions=decisions,
        approved_spend=round(sum(d.cost for d in decisions if d.status == "approved"), 2),
        pending_spend=round(sum(d.cost for d in decisions if d.status == "needs_human_approval"), 2),
        missed=missed,
    )


def run_reorder(llm: LLM, items: list[StockItem], policy: Policy) -> ReorderRun:
    payload = json.dumps([item.model_dump() for item in items])
    proposals = structured_call(llm, "guarded_reorder", INSTRUCTIONS, payload, Proposals)
    return review(proposals, items, policy)
