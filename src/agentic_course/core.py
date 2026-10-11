"""The one pattern all three projects share: ask for JSON, validate it, retry with the error."""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .llm import LLM, extract_json

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """The model could not produce output that satisfies the schema."""


def structured_call(llm: LLM, task: str, instructions: str, user_content: str, schema: type[T], max_attempts: int = 3) -> T:
    """Get a validated ``schema`` instance from the model.

    ``task`` is a short tag placed on the first line of the system prompt. A
    reply that is not valid JSON, or does not match the schema, is sent back
    with the validation error so the model can correct itself.
    """
    system = (
        f"TASK:{task}\n{instructions}\n\nReply with one JSON object only. It must match this JSON schema:\n"
        f"{json.dumps(schema.model_json_schema())}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_content}]
    error = ""
    for _ in range(max_attempts):
        reply = llm.complete(messages)
        try:
            return schema.model_validate(extract_json(reply))
        except (ValueError, ValidationError) as exc:
            error = str(exc)
            messages = messages + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"Your reply was rejected: {error}\nSend a corrected JSON object."},
            ]
    raise StructuredOutputError(f"{task}: no valid output after {max_attempts} attempts: {error}")
