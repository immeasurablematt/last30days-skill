from __future__ import annotations

from unittest import mock

from lib import bird_x, linkedin


def test_bird_named_entity_never_relaxes_to_generic_tokens() -> None:
    calls: list[str] = []

    def fake_search(query: str, count: int, timeout: int) -> dict:
        calls.append(query)
        return {"items": []}

    with mock.patch.object(bird_x, "_run_bird_search", side_effect=fake_search):
        result = bird_x.search_x(
            '"Follow Up Boss" automation', "2026-07-21", "2026-08-20"
        )

    assert result == {"items": []}
    assert calls == ['("Follow Up Boss") since:2026-07-21']
    assert all("follow up since:" not in query.casefold() for query in calls)


def test_bird_non_entity_query_keeps_existing_retry_behavior() -> None:
    responses = [{"items": []}, {"items": []}, {"items": []}, {"items": []}]
    with mock.patch.object(bird_x, "_run_bird_search", side_effect=responses) as search:
        bird_x.search_x("creator workflow trends", "2026-07-21", "2026-08-20")

    assert search.call_count > 1


def test_linkedin_company_page_is_not_treated_as_person_profile() -> None:
    items = [{
        "author": "Acme Analytics",
        "author_url": "https://www.linkedin.com/company/acme-analytics",
    }]
    assert linkedin._best_author_match(items, "Acme Analytics") == ""


def test_linkedin_person_profile_still_enriches() -> None:
    items = [{
        "author": "Ada Lovelace",
        "author_url": "https://www.linkedin.com/in/ada-lovelace",
    }]
    assert linkedin._best_author_match(items, "Ada Lovelace") == (
        "https://www.linkedin.com/in/ada-lovelace"
    )
