from __future__ import annotations

import json
import logging

from .parse import extract_json

logger = logging.getLogger(__name__)


def _default_estimate_tokens(item) -> int:
    """Conservative token estimate for CJK-heavy JSON (~2 chars/token)."""
    return len(json.dumps(item, ensure_ascii=False)) // 2


def merge_structured(
    items: list[dict],
    llm,
    merge_prompt: str,
    budget_tokens: int,
    estimate_tokens=None,
    required_key: str = None,
) -> dict:
    """Hierarchically merge structured dicts with an LLM (tree merge).

    Splits `items` into batches that fit within `budget_tokens`, asks the LLM
    to merge each batch (prompt = merge_prompt + JSON array of the batch),
    then repeats on the merged results until one dict remains. This avoids
    the context overflow of a single flat merge over all items.

    llm: an object with `complete(prompt, system=None, temperature=...) -> str`
         (e.g. LMStudioClient).
    merge_prompt: instructions for merging; the batch is appended as a JSON
                  array after it. The LLM must answer with a single JSON object.
    budget_tokens: max estimated input tokens per merge call.
    estimate_tokens: optional callable(item) -> int; defaults to
                     len(json.dumps(item, ensure_ascii=False)) // 2.

    Each LLM response is parsed with extract_json; on parse failure the batch
    is retried once at temperature=0 before raising.
    required_key: schema anchor (e.g. "topics") — a merge result that is
    valid JSON but missing this key is retried with an explicit reminder,
    then raises (models sometimes drop the outer report shape and return
    a bare topic object).
    """
    if estimate_tokens is None:
        estimate_tokens = _default_estimate_tokens

    if not items:
        raise ValueError("merge_structured: items is empty")
    if len(items) == 1:
        return items[0]

    current = list(items)
    while len(current) > 1:
        batches = _make_batches(current, estimate_tokens, budget_tokens)

        # Guard against no-progress loops: if batching degenerated into all
        # singletons (every item alone exceeds the budget), force pairwise
        # merges so the item count still halves each round.
        if all(len(b) == 1 for b in batches):
            batches = [current[i:i + 2] for i in range(0, len(current), 2)]

        merged = []
        for batch in batches:
            if len(batch) == 1:
                merged.append(batch[0])
            else:
                merged.append(_merge_batch(batch, llm, merge_prompt, required_key))
        current = merged

    return current[0]


def _make_batches(items: list, estimate_tokens, budget_tokens: int) -> list[list]:
    """Greedily pack items into batches under the token budget.

    Every batch has at least one item (an oversized item becomes a singleton).
    """
    batches = []
    batch = []
    batch_tokens = 0
    for item in items:
        cost = estimate_tokens(item)
        if batch and batch_tokens + cost > budget_tokens:
            batches.append(batch)
            batch = []
            batch_tokens = 0
        batch.append(item)
        batch_tokens += cost
    if batch:
        batches.append(batch)
    return batches


def _merge_batch(batch: list, llm, merge_prompt: str, required_key: str = None):
    """Merge one batch via the LLM; retry once at temperature=0 on bad JSON
    or on a schema-shape miss (required_key absent)."""
    prompt = "%s\n\n%s" % (
        merge_prompt,
        json.dumps(batch, ensure_ascii=False),
    )

    def _ok(result):
        return (isinstance(result, dict)
                and (required_key is None or required_key in result))

    response = llm.complete(prompt, max_tokens=4096)
    try:
        result = extract_json(response)
        if _ok(result):
            return result
    except ValueError:
        pass

    retry_prompt = prompt if required_key is None else (
        prompt + "\n\nREMINDER: output MUST be a JSON object with a "
                 "top-level \"%s\" array, exactly as specified." % required_key)
    response = llm.complete(retry_prompt, temperature=0, max_tokens=4096)
    try:
        result = extract_json(response)
    except ValueError as e:
        raise ValueError(
            "merge_structured: LLM merge output was not valid JSON after retry: %s" % e
        )
    if not _ok(result):
        if isinstance(result, dict) and result:
            # Salvage: the model answered with the inner shape (e.g. a bare
            # topic object) — wrap it instead of failing the whole run.
            logger.warning(
                "merge output missing %r after retry; coercing %s into a wrapper",
                required_key, list(result)[:5])
            return {"title": result.get("title", ""), required_key: [result]}
        raise ValueError(
            "merge_structured: merge output missing required key %r after retry"
            % required_key)
    return result
