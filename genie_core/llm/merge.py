from __future__ import annotations

import json

from .parse import extract_json


def _default_estimate_tokens(item) -> int:
    """Conservative token estimate for CJK-heavy JSON (~2 chars/token)."""
    return len(json.dumps(item, ensure_ascii=False)) // 2


def merge_structured(
    items: list[dict],
    llm,
    merge_prompt: str,
    budget_tokens: int,
    estimate_tokens=None,
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
                merged.append(_merge_batch(batch, llm, merge_prompt))
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


def _merge_batch(batch: list, llm, merge_prompt: str):
    """Merge one batch via the LLM; retry once at temperature=0 on bad JSON."""
    prompt = "%s\n\n%s" % (
        merge_prompt,
        json.dumps(batch, ensure_ascii=False),
    )

    response = llm.complete(prompt, max_tokens=4096)
    try:
        return extract_json(response)
    except ValueError:
        pass

    response = llm.complete(prompt, temperature=0, max_tokens=4096)
    try:
        return extract_json(response)
    except ValueError as e:
        raise ValueError(
            "merge_structured: LLM merge output was not valid JSON after retry: %s" % e
        )
