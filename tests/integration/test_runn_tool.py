"""
runn_allocations (gub-agent#89) over a real localhost MockGub serving UB's
Runn relay (`/relay/runn/*`).

Pinned here: the joins (a person by email, a project by name, names attached
to every allocation, minutes turned into hours, role names, working days,
confirmed vs tentative, placeholders); a date range alone is enough; several
matching projects come back for the user to choose; the relay's error code reaches the caller
(RUNN_NO_GRANT), which is what the /ask turn reports as a failed source; and
the turn's source report reads it that way.
"""

from __future__ import annotations

from gub_agent.agents.ask import SourceLog
from gub_agent.tools.runn import runn_allocations

PEOPLE = {
    "values": [
        {"id": 1, "firstName": "Ada", "lastName": "Lovelace", "email": "Ada@anomaly.com"},
        {"id": 2, "firstName": "Alan", "lastName": "Turing", "email": "alan@anomaly.com"},
    ],
    "nextCursor": None,
}
PROJECTS = {
    "values": [
        {"id": 10, "name": "Nike Holiday", "isArchived": False},
        {"id": 11, "name": "Nike Spring", "isArchived": False},
        {"id": 12, "name": "Chevy Q3", "isArchived": False},
    ],
    "nextCursor": None,
}


def _assignments(*rows: dict) -> dict:
    return {"values": list(rows), "nextCursor": None}


def _runn(gub, *assignments: dict, roles=None, contracts=None) -> None:
    gub.routes[("GET", "/relay/runn/people")] = (200, PEOPLE)
    gub.routes[("GET", "/relay/runn/projects")] = (200, PROJECTS)
    gub.routes[("GET", "/relay/runn/assignments")] = (200, _assignments(*assignments))
    if roles is not None:
        gub.routes[("GET", "/relay/runn/roles")] = (200, {"values": roles, "nextCursor": None})
    if contracts is not None:
        gub.routes[("GET", "/relay/runn/contracts")] = (
            200,
            {"values": contracts, "nextCursor": None},
        )


async def test_a_person_by_email_with_their_projects_named(gub):
    _runn(
        gub,
        {
            "personId": 1,
            "projectId": 12,
            "roleId": 5,
            "startDate": "2026-11-01",
            "endDate": "2026-12-15",
            "minutesPerDay": 240,
            "isBillable": True,
        },
        roles=[{"id": 5, "name": "Strategist"}],
        contracts=[
            {
                "personId": 1,
                "startDate": "2025-01-01",
                "endDate": None,
                "rosteredDays": {
                    "monday": 480,
                    "tuesday": 480,
                    "wednesday": 480,
                    "thursday": 0,
                    "friday": 0,
                },
            }
        ],
    )

    gub.routes[("GET", "/relay/runn/time-offs/leave")] = (
        200,
        {
            "values": [
                {"personId": 1, "startDate": "2026-12-21", "endDate": "2027-01-04",
                 "minutesPerDay": None}
            ],
            "nextCursor": None,
        },
    )  # fmt: skip

    result = await runn_allocations(person_email="ada@anomaly.com", start_date="2026-11-01")

    assert result == {
        "count": 1,
        "allocations": [
            {
                "person": "Ada Lovelace",
                "email": "Ada@anomaly.com",
                "project": "Chevy Q3",
                "start_date": "2026-11-01",
                "end_date": "2026-12-15",
                "hours_per_day": 4.0,
                "billable": True,
                "status": "confirmed",
                "role": "Strategist",
                "works": "Mon-Wed 8h",
            }
        ],
        "leave": [
            {
                "person": "Ada Lovelace",
                "start_date": "2026-12-21",
                "end_date": "2027-01-04",
                "whole_days": True,
            }
        ],
    }
    [leave_call] = [t for m, t in gub.targets if t.startswith("/relay/runn/time-offs/leave")]
    assert "personId=1" in leave_call and "startDate=2026-11-01" in leave_call
    [assignments_call] = [t for m, t in gub.targets if t.startswith("/relay/runn/assignments")]
    assert "personId=1" in assignments_call and "startDate=2026-11-01" in assignments_call


async def test_a_project_by_name_with_its_people_named(gub):
    _runn(
        gub,
        {"personId": 1, "projectId": 12, "minutesPerDay": 480},
        {"personId": 2, "projectId": 12, "minutesPerDay": 120},
    )

    result = await runn_allocations(project_name="chevy q3")

    assert [(a["person"], a["project"], a["hours_per_day"]) for a in result["allocations"]] == [
        ("Ada Lovelace", "Chevy Q3", 8.0),
        ("Alan Turing", "Chevy Q3", 2.0),
    ]
    # Undated assignments cover any day: two people, 10 h a day between them.
    [summary] = result["summary"]
    assert (summary["people"], summary["hours_per_day"]) == (2, 10.0)
    # Roles and contracts are context: a relay without them still answers, and says so.
    assert result["notes"] == [
        "Role names could not be read from Runn.",
        "Working days could not be read from Runn.",
    ]


async def test_tentative_work_and_placeholders_are_marked(gub):
    gub.routes[("GET", "/relay/runn/people")] = (
        200,
        {
            "values": [*PEOPLE["values"], {"id": 9, "firstName": "Designer", "lastName": "TBC"}],
            "nextCursor": None,
        },
    )
    gub.routes[("GET", "/relay/runn/projects")] = (
        200,
        {"values": [{"id": 20, "name": "Pitch", "isConfirmed": False}], "nextCursor": None},
    )
    gub.routes[("GET", "/relay/runn/assignments")] = (
        200,
        _assignments(
            {"personId": 1, "projectId": 20, "minutesPerDay": 480},
            {"personId": 9, "projectId": 20, "minutesPerDay": 480, "isPlaceholder": True},
        ),
    )

    result = await runn_allocations(start_date="2026-12-01", end_date="2026-12-31")

    assert [(a["person"], a["status"], a.get("placeholder")) for a in result["allocations"]] == [
        ("Ada Lovelace", "tentative", None),
        ("Designer TBC", "tentative", True),
    ]
    [people_call] = [t for m, t in gub.targets if t.startswith("/relay/runn/people")]
    assert "includePlaceholders=true" in people_call


async def test_several_matching_projects_are_returned_to_choose_from(gub):
    _runn(gub)

    result = await runn_allocations(project_name="nike")

    assert [p["name"] for p in result["projects"]] == ["Nike Holiday", "Nike Spring"]
    assert "allocations" not in result
    assert not any(t.startswith("/relay/runn/assignments") for _, t in gub.targets)


async def test_an_unknown_person_has_no_runn_record_which_is_not_free(gub):
    _runn(gub)
    result = await runn_allocations(person_email="nobody@anomaly.com")
    assert result["allocations"] == [] and "nobody@anomaly.com" in result["note"]
    assert "do not call them free" in result["note"]


async def test_no_grant_comes_back_with_the_relays_code_and_fails_the_runn_source(gub):
    gub.routes[("GET", "/relay/runn/projects")] = (
        403,
        {"code": "RUNN_NO_GRANT", "message": 'Reading Runn needs the "Runn reader" grant.'},
    )

    result = await runn_allocations(project_name="chevy")

    assert result["error"] is True
    assert (result["status"], result["code"]) == (403, "RUNN_NO_GRANT")

    log = SourceLog()
    log.called("runn_allocations")
    log.answered("runn_allocations", result)
    reports = log.reports()
    assert reports["runn"].status == "error"
    assert reports["runn"].error.code == "RUNN_NO_GRANT"
    assert reports["ub"].status == "skipped"


async def test_no_filter_at_all_is_refused_without_a_call(gub):
    result = await runn_allocations()
    assert result["error"] is True and gub.requests == []


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
