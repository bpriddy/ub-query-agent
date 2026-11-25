"""
The resourcing tools (gub-agent#117) over a real localhost MockGub serving
UB's resourcing relay (`/relay/resourcing/*`).

Pinned here: the relay's error code reaches the caller through the real HTTP
client (RESOURCING_NO_GRANT, RESOURCING_CAPACITY_OFF), and the /ask turn's
source report reads it as a failed source, still named "runn" on the wire
until UB #302; and a plain answer carries the leg's freshness. The joins,
filters and paging are pinned in tests/unit/test_resourcing_tools.py.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

from gub_agent.agents.ask import SourceLog
from gub_agent.tools.resourcing import resourcing_allocations, resourcing_capacity

META = {
    "sources": [{"source": "runn", "asOf": "2026-11-24T06:00:00Z", "contractVersion": 1}],
    "limit": 1000,
    "offset": 0,
    "nextOffset": None,
}


async def test_no_grant_comes_back_with_the_relays_code_and_fails_the_source(gub):
    gub.routes[("GET", "/relay/resourcing/projects")] = (
        403,
        {
            "code": "RESOURCING_NO_GRANT",
            "message": 'Reading resourcing data needs the "Resourcing reader" grant.',
            "details": {"grant": "resourcing_reader"},
        },
    )

    result = await resourcing_allocations(project_name="chevy")

    assert result["error"] is True
    assert (result["status"], result["code"]) == (403, "RESOURCING_NO_GRANT")

    log = SourceLog()
    log.called("resourcing_allocations")
    log.answered("resourcing_allocations", result)
    reports = log.reports()
    assert reports["runn"].status == "error"
    assert reports["runn"].error.code == "RESOURCING_NO_GRANT"
    assert reports["ub"].status == "skipped"


async def test_capacity_off_is_a_failed_source_not_a_crash(gub):
    gub.routes[("GET", "/relay/resourcing/capacity")] = (
        503,
        {
            "code": "RESOURCING_CAPACITY_OFF",
            "message": "Resourcing capacity is not served yet in this environment.",
        },
    )

    result = await resourcing_capacity("2026-12-01", "2026-12-31", team="Design")

    assert (result["error"], result["status"], result["code"]) == (
        True,
        503,
        "RESOURCING_CAPACITY_OFF",
    )
    [target] = [t for _, t in gub.targets if t.startswith("/relay/resourcing/capacity")]
    sent = parse_qs(urlsplit(target).query)
    assert sent["from"] == ["2026-12-01"] and sent["to"] == ["2026-12-31"]
    assert sent["team.contains"] == ["Design"]

    log = SourceLog()
    log.called("resourcing_capacity")
    log.answered("resourcing_capacity", result)
    assert log.reports()["runn"].error.code == "RESOURCING_CAPACITY_OFF"


async def test_capacity_answers_with_the_legs_freshness(gub):
    gub.routes[("GET", "/relay/resourcing/capacity")] = (
        200,
        {
            "data": [
                {
                    "source": "runn",
                    "name": "Ada Lovelace",
                    "email": "ada@anomaly.com",
                    "team": "Design",
                    "personId": "1",
                    "capacityHours": 120,
                    "allocatedHours": 90,
                    "availableHours": 30,
                    "utilizationPercent": 75,
                    "timeOffHours": 8,
                },
            ],
            "meta": {"sources": META["sources"]},
        },
    )

    result = await resourcing_capacity("2026-12-01", "2026-12-31")

    assert result["people"][0]["available_h"] == 30.0
    assert result["totals"][0]["utilization_pct"] == 75
    assert result["sources"] == [
        {"source": "runn", "as_of": "2026-11-24T06:00:00Z", "contract_version": 1}
    ]
    log = SourceLog()
    log.called("resourcing_capacity")
    log.answered("resourcing_capacity", result)
    assert log.reports()["runn"].status == "ok"


def test_a_ub_not_found_is_an_answer_not_a_failed_source():
    log = SourceLog()
    log.called("get_campaign")
    log.answered("get_campaign", {"error": True, "status": 404, "message": "not found"})
    assert log.reports()["ub"].status == "ok"
    log.answered("get_campaign", {"error": True, "status": 0, "message": "unreachable"})
    assert log.reports()["ub"].status == "ok"  # one call did answer

    down = SourceLog()
    down.called("get_campaign")
    down.answered("get_campaign", {"error": True, "status": 503, "message": "down"})
    assert down.reports()["ub"].error.code == "HTTP_503"
