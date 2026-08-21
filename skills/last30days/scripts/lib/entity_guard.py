"""Fail-closed evidence gating for named-entity research plans.

The planner contract quotes multi-word proper nouns in ``search_query``.
Those quoted strings are therefore a useful, domain-independent signal that a
run is about a specific product, company, project, or person.  This module
turns them into retrieval anchors and prevents source adapters from converting
an exact entity search into a bag of generic words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from . import schema


_QUOTED_RE = re.compile(r'["\u201c\u201d]([^"\u201c\u201d]{2,})["\u201c\u201d]')
_TITLE_CASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9.+#'-]*\s+){1,}[A-Z][A-Za-z0-9.+#'-]*\b"
)
_META_TERMS = {
    "automation", "automations", "best", "community", "comparison",
    "discussion", "experience", "experiences", "feature", "features",
    "guide", "integration", "integrations", "latest", "news", "opinion",
    "opinions", "production", "review", "reviews", "tutorial", "tutorials",
    "update", "updates", "use", "uses", "workflow", "workflows",
}


def _normalized(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())


def _compact(text: str) -> str:
    return re.sub(r"[^\w]+", "", text.casefold())


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = " ".join(value.split()).strip()
        key = _normalized(clean)
        if not clean or not key or key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return tuple(result)


def _acronym(anchor: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", anchor)
    if len(words) < 3:
        return ""
    value = "".join(word[0] for word in words).upper()
    return value if len(value) >= 3 else ""


@dataclass(frozen=True)
class EntityGate:
    """Named-entity anchors plus supporting context terms for one plan."""

    anchors: tuple[str, ...]
    context_terms: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return bool(self.anchors)


def from_plan(plan: schema.QueryPlan) -> EntityGate:
    """Derive a strict entity gate from the plan's search queries.

    Quoted values are authoritative because the skill tells planners to quote
    proper nouns only.  Title-cased multi-word phrases are a compatibility
    fallback for older plans.  Ordinary topical searches therefore remain
    untouched.
    """
    searches = [subquery.search_query for subquery in plan.subqueries]
    quoted = [match.group(1) for text in searches for match in _QUOTED_RE.finditer(text)]
    anchors = _dedupe(quoted)
    if not anchors:
        anchors = _dedupe(
            match.group(0)
            for text in searches
            for match in _TITLE_CASE_RE.finditer(text)
        )
    if not anchors:
        return EntityGate(())

    context: list[str] = []
    anchor_norms = [_normalized(anchor) for anchor in anchors]
    for text in searches:
        cleaned = _normalized(text)
        for anchor in anchor_norms:
            cleaned = cleaned.replace(anchor, " ")
        for token in cleaned.split():
            if len(token) >= 3 and token not in _META_TERMS:
                context.append(token)
    return EntityGate(anchors=anchors, context_terms=_dedupe(context))


def from_search_query(search_query: str) -> EntityGate:
    """Derive anchors from one adapter query using the planner's same contract."""
    quoted = _dedupe(match.group(1) for match in _QUOTED_RE.finditer(search_query))
    if quoted:
        return EntityGate(anchors=quoted)
    title_cased = _dedupe(match.group(0) for match in _TITLE_CASE_RE.finditer(search_query))
    return EntityGate(anchors=title_cased)


def retry_query(gate: EntityGate) -> str:
    """Return the narrowest safe retry expression for an entity plan."""
    if not gate.active:
        return ""
    return " OR ".join(f'"{anchor}"' for anchor in gate.anchors)


def item_matches(
    item: schema.SourceItem,
    gate: EntityGate,
    resolved_handles: set[str] | None = None,
) -> bool:
    """Whether an item carries a stable identity signal for the named entity."""
    if not gate.active or item.source == "corpus":
        return True

    normalized_handles = {
        handle.lstrip("@").casefold() for handle in (resolved_handles or set()) if handle
    }
    author_handle = (item.author or "").lstrip("@").casefold()
    if author_handle and author_handle in normalized_handles:
        return True

    primary = " ".join(
        value
        for value in (item.title, item.body, item.snippet)
        if value
    )
    identity = " ".join(
        value
        for value in (item.author, item.container, item.url)
        if value
    )
    normalized_primary = _normalized(primary)
    normalized_identity = _normalized(identity)
    compact_identity = _compact(identity)
    compact_primary_tokens = {
        _compact(token) for token in re.findall(r"[#@]?[\w.-]+", primary)
    }

    for anchor in gate.anchors:
        normalized_anchor = _normalized(anchor)
        compact_anchor = _compact(anchor)
        acronym = _acronym(anchor)

        # Author, container, and URL are identity-bearing surfaces.  A
        # case-insensitive match there is enough even when the post body does
        # not repeat its own author's/company's name.
        if (
            normalized_anchor in normalized_identity
            or (len(compact_anchor) >= 5 and compact_anchor in compact_identity)
        ):
            return True

        # Exact casing is a strong signal for title-cased proper nouns.
        if anchor in primary:
            return True

        # Compact brand spellings and conventional acronyms are accepted.
        if len(compact_anchor) >= 5 and compact_anchor in compact_primary_tokens:
            return True
        if acronym and re.search(rf"\b{re.escape(acronym)}\b", primary, re.IGNORECASE):
            return True

        # A case-insensitive phrase can be ordinary prose (for example,
        # "follow up, boss").  Require one plan-derived context term before
        # treating that weaker form as entity evidence.
        if normalized_anchor in normalized_primary and any(
            re.search(rf"\b{re.escape(term)}\b", normalized_primary)
            for term in gate.context_terms
        ):
            return True
    return False


def filter_items(
    items: Iterable[schema.SourceItem],
    gate: EntityGate,
    resolved_handles: set[str] | None = None,
) -> tuple[list[schema.SourceItem], int]:
    kept = [
        item for item in items
        if item_matches(item, gate, resolved_handles=resolved_handles)
    ]
    total = len(items) if isinstance(items, list) else None
    if total is None:
        return kept, 0
    return kept, total - len(kept)
