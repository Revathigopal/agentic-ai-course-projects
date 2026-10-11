import json
from pathlib import Path

import pytest

from agentic_course.cli import main
from agentic_course.core import StructuredOutputError, structured_call
from agentic_course.enquiry_triage import Triage, decide, triage_enquiry
from agentic_course.guarded_reorder import Policy, ProposedOrder, Proposals, StockItem, review, run_reorder
from agentic_course.invoice_to_ledger import Invoice, is_balanced, journal_for, process_invoice, reconcile
from agentic_course.llm import MockLLM
from agentic_course.mock import mock_handler

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def llm():
    return MockLLM(mock_handler)


# --- shared pattern -----------------------------------------------------


def test_structured_call_feeds_validation_error_back():
    good = {"category": "sales", "priority": "low", "draft_reply": "Thank you for your message."}
    replies = iter(["not json", json.dumps({**good, "priority": "critical"}), json.dumps(good)])
    model = MockLLM(lambda messages: next(replies))
    result = structured_call(model, "t", "i", "hello", Triage)
    assert result.priority == "low" and len(model.calls) == 3
    assert "priority" in model.calls[2][-1]["content"]  # the schema error was shown to the model


def test_structured_call_gives_up():
    with pytest.raises(StructuredOutputError):
        structured_call(MockLLM(lambda m: "{}"), "t", "i", "hello", Triage, max_attempts=2)


# --- project 1: enquiry triage ------------------------------------------


@pytest.mark.parametrize(
    "text, category, handler",
    [
        ("Please send a price quote for 50 lamps", "sales", "auto_reply"),
        ("The item arrived damaged, this is unacceptable", "complaint", "human"),
        ("I was charged Rs. 3,500 twice on my invoice", "billing", "human"),
        ("I was charged Rs. 150 extra on my invoice", "billing", "auto_reply"),
        ("How do I install the app? Need it urgent", "support", "human"),
        ("You are a lottery winner, click here", "spam", "discard"),
    ],
)
def test_triage_policy(text, category, handler):
    decision = triage_enquiry(llm(), text)
    assert (decision.triage.category, decision.handler) == (category, handler)


def test_triage_extracts_order_number_and_amount():
    triage = triage_enquiry(llm(), "Order ORD-88213 was billed Rs. 1,250.50").triage
    assert triage.order_number == "ORD-88213" and triage.amount == 1250.5


def test_draft_that_promises_a_refund_is_never_auto_sent():
    triage = Triage(category="support", priority="low", draft_reply="Sorry! We will refund you in full today.")
    decision = decide(triage)
    assert decision.handler == "human" and "refund" in decision.reasons[0]


# --- project 2: invoice to ledger ---------------------------------------


def test_good_invoice_posts_a_balanced_journal():
    result = process_invoice(llm(), (EXAMPLES / "invoice_ok.txt").read_text())
    assert result.status == "posted" and is_balanced(result.journal)
    assert [(l.account, l.debit, l.credit) for l in result.journal] == [
        ("Purchases", 11800.0, 0.0),
        ("Input GST receivable", 590.0, 0.0),
        ("Accounts payable - Sri Balaji Traders", 0.0, 12390.0),
    ]


def test_bad_arithmetic_is_caught_and_nothing_is_posted():
    result = process_invoice(llm(), (EXAMPLES / "invoice_bad.txt").read_text())
    assert result.status == "needs_review" and result.journal == []
    assert any("line 2" in p for p in result.problems)
    assert any("total says 7180" in p for p in result.problems)


def invoice(**overrides):
    base = dict(
        vendor="V", invoice_number="1", invoice_date="2026-04-01",
        items=[{"description": "a", "quantity": 2, "unit_price": 50, "amount": 100}],
        subtotal=100, gst_rate_pct=18, gst_amount=18, total=118,
    )
    return Invoice.model_validate({**base, **overrides})


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"subtotal": 120, "gst_amount": 21.6, "total": 141.6}, "lines add up to 100"),
        ({"gst_amount": 12, "total": 112}, "GST at 18% should be 18"),
        ({"total": 150}, "total says 150"),
    ],
)
def test_reconcile_finds_each_kind_of_error(overrides, fragment):
    assert any(fragment in problem for problem in reconcile(invoice(**overrides)))


def test_reconcile_allows_rounding_and_zero_rated_invoices():
    assert reconcile(invoice(gst_amount=18.4, total=118.4)) == []
    zero = invoice(gst_rate_pct=0, gst_amount=0, total=100)
    assert reconcile(zero) == []
    journal = journal_for(zero)
    assert len(journal) == 2 and is_balanced(journal)


def test_duplicate_invoice_is_held():
    seen: set[str] = set()
    text = (EXAMPLES / "invoice_ok.txt").read_text()
    assert process_invoice(llm(), text, seen).status == "posted"
    second = process_invoice(llm(), text, seen)
    assert second.status == "needs_review" and "already processed" in second.problems[0]


# --- project 3: guarded reorder -----------------------------------------

ITEMS = [
    StockItem(sku="A", stock=2, reorder_point=10, monthly_sales=20, unit_cost=100, suppliers=["S1"]),
    StockItem(sku="B", stock=50, reorder_point=10, monthly_sales=20, unit_cost=100, suppliers=["S1", "S2"]),
    StockItem(sku="C", stock=0, reorder_point=5, monthly_sales=10, unit_cost=2000, suppliers=["S2"]),
]
POLICY = Policy(budget=30000, max_months_of_cover=3, approval_threshold=10000)


def order(sku="A", supplier="S1", quantity=20):
    return ProposedOrder(sku=sku, supplier=supplier, quantity=quantity, reason="stock is low")


def only(decision_run):
    assert len(decision_run.decisions) == 1
    return decision_run.decisions[0]


def test_valid_small_order_is_approved():
    d = only(review(Proposals(orders=[order()]), ITEMS, POLICY))
    assert (d.status, d.cost) == ("approved", 2000)


@pytest.mark.parametrize(
    "proposal, fragment",
    [
        (order(sku="Z"), "unknown SKU"),
        (order(sku="B"), "above the reorder point"),
        (order(supplier="Cheap Unknown Ltd"), "not an approved supplier"),
        (order(quantity=61), "exceeds 3 months of sales (60)"),
    ],
)
def test_guardrails_reject_with_reason(proposal, fragment):
    d = only(review(Proposals(orders=[proposal]), ITEMS, POLICY))
    assert d.status == "rejected" and fragment in d.reasons[0]


def test_large_order_waits_for_a_person_and_reserves_budget():
    run = review(Proposals(orders=[order(sku="C", supplier="S2", quantity=10), order(quantity=60), order(quantity=20)]), ITEMS, POLICY)
    first, second, third = run.decisions
    assert first.status == "needs_human_approval" and first.cost == 20000
    assert second.status == "approved" and second.cost == 6000
    assert third.status == "rejected" and "second order for the same SKU" in third.reasons[0]
    assert (run.approved_spend, run.pending_spend) == (6000, 20000)


def test_budget_is_enforced_and_missed_items_are_reported():
    tight = Policy(budget=5000, approval_threshold=10000)
    run = review(Proposals(orders=[order(quantity=20), order(sku="C", supplier="S2", quantity=5)]), ITEMS, tight)
    assert run.decisions[0].status == "approved"
    assert run.decisions[1].status == "rejected" and "remaining budget of 3000" in run.decisions[1].reasons[0]
    assert run.missed == ["C"]
    assert run.approved_spend + run.pending_spend <= tight.budget


def test_reorder_end_to_end_with_offline_backend():
    data = json.loads((EXAMPLES / "stock.json").read_text())
    items = [StockItem.model_validate(i) for i in data["items"]]
    run = run_reorder(llm(), items, Policy.model_validate(data["policy"]))
    assert {d.order.sku for d in run.decisions} == {"LAMP-BRASS", "WICK-COTTON", "OIL-SESAME-1L"}
    assert run.missed == []
    assert run.approved_spend + run.pending_spend <= data["policy"]["budget"]


def test_cli_runs_all_three(capsys):
    assert main(["triage", str(EXAMPLES / "enquiries.txt")]) == 0
    assert main(["ledger", str(EXAMPLES / "invoice_ok.txt"), str(EXAMPLES / "invoice_bad.txt")]) == 0
    assert main(["reorder", str(EXAMPLES / "stock.json")]) == 0
    out = capsys.readouterr().out
    assert "needs_review" in out and "needs_human_approval" in out and "discard" in out
    assert main(["ledger", str(EXAMPLES / "missing.txt")]) == 1
