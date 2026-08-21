from __future__ import annotations

from lib import entity_guard, pipeline, schema


def _plan(*queries: str) -> schema.QueryPlan:
    return schema.QueryPlan(
        intent="product",
        freshness_mode="strict_recent",
        cluster_mode="workflow",
        raw_topic="fixture",
        subqueries=[
            schema.SubQuery(
                label=f"q{index}",
                search_query=query,
                ranking_query=query,
                sources=["x"],
            )
            for index, query in enumerate(queries, start=1)
        ],
        source_weights={"x": 1.0},
    )


def _item(text: str, *, source: str = "x", author: str = "someone") -> schema.SourceItem:
    return schema.SourceItem(
        item_id=text,
        source=source,
        title=text,
        body="",
        snippet="",
        url="https://example.com/item",
        author=author,
    )


def test_fub_is_a_fixture_not_a_hardcoded_domain_rule() -> None:
    gate = entity_guard.from_plan(
        _plan('"Follow Up Boss" CRM automation', '"Follow Up Boss" workflow')
    )

    assert gate.anchors == ("Follow Up Boss",)
    assert entity_guard.item_matches(_item("Follow Up Boss launches a new workflow"), gate)
    assert entity_guard.item_matches(_item("FUB launches a new workflow"), gate)
    assert not entity_guard.item_matches(_item("Remember to follow up, boss"), gate)
    assert not entity_guard.item_matches(_item("Generic n8n automation tutorial"), gate)


def test_same_gate_works_for_an_unrelated_company() -> None:
    gate = entity_guard.from_plan(_plan('"Acme Analytics" customer stories'))

    assert gate.anchors == ("Acme Analytics",)
    assert entity_guard.item_matches(_item("Acme Analytics customer story"), gate)
    assert entity_guard.item_matches(
        _item("We shipped dashboards today", author="Acme Analytics"), gate
    )
    assert not entity_guard.item_matches(_item("The best analytics dashboards"), gate)


def test_resolved_first_party_handle_is_identity_evidence() -> None:
    gate = entity_guard.from_plan(_plan('"Ada Lovelace" current projects'))

    assert entity_guard.item_matches(
        _item("Shipping a new prototype", author="ada_dev"),
        gate,
        resolved_handles={"ada_dev"},
    )


def test_unquoted_topical_plan_does_not_activate_strict_mode() -> None:
    gate = entity_guard.from_plan(_plan("AI video tools", "creator workflow trends"))

    assert not gate.active
    assert entity_guard.item_matches(_item("Any topical evidence remains eligible"), gate)


def test_filter_runs_before_source_counts_and_fusion() -> None:
    gate = entity_guard.from_plan(_plan('"Acme Analytics" dashboards'))
    good = _item("Acme Analytics dashboard release")
    bad = _item("Watermarking research paper")
    bundle = schema.RetrievalBundle()
    bundle.add_items("q1", "x", [good, bad])

    removed = pipeline._apply_entity_evidence_gate(bundle, gate)

    assert removed == {"x": 1}
    assert bundle.items_by_source["x"] == [good]
    assert bundle.items_by_source_and_query[("q1", "x")] == [good]
