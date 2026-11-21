"""
float_allocations and float_availability over a real localhost MockGub
serving UB's Float relay (`/relay/float/*`, shaped as UB answers:
`{values, pagination}` for lists).

Pinned here: a person found by email through the relay's filter; a project by
name; names attached to every allocation, including multi-person ones; a
one-sided date range closed for Float; several matching projects come back
for the user to choose; capacity from the people report with the person's
time off; and the relay's error code reaching the caller (FLOAT_NO_GRANT).
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

from gub_agent.tools.float import float_allocations, float_availability

ADA = {"people_id": 1, "name": "Ada Lovelace", "email": "Ada@anomaly.com"}
ALAN = {"people_id": 2, "name": "Alan Turing", "email": "alan@anomaly.com"}
PROJECTS = [
    {"project_id": 10, "name": "Nike Holiday", "active": 1},
    {"project_id": 11, "name": "Nike Spring", "active": 0},
    {"project_id": 12, "name": "Chevy Q3", "active": 1},
]


def _list(*rows: dict) -> dict:
    return {
        "values": list(rows),
        "pagination": {"page": 1, "pageCount": 1, "totalCount": len(rows), "perPage": 200},
    }


def _query(gub, prefix: str) -> dict[str, list[str]]:
    [target] = [t for _, t in gub.targets if t.startswith(prefix)]
    return parse_qs(urlsplit(target).query)


async def test_a_person_by_email_with_their_projects_named(gub):
    gub.routes[("GET", "/relay/float/people")] = (200, _list(ADA))
    gub.routes[("GET", "/relay/float/tasks")] = (
        200,
        _list(
            {
                "task_id": 5,
                "people_id": 1,
                "project_id": 12,
                "start_date": "2026-11-02",
                "end_date": "2026-12-15",
                "hours": 4,
                "billable": 1,
                "status": 2,
            }
        ),
    )
    gub.routes[("GET", "/relay/float/projects/12")] = (200, PROJECTS[2])

    result = await float_allocations(person_email="ada@anomaly.com", start_date="2026-11-01")

    assert result == {
        "count": 1,
        "allocations": [
            {
                "person": "Ada Lovelace",
                "email": "Ada@anomaly.com",
                "project": "Chevy Q3",
                "start_date": "2026-11-02",
                "end_date": "2026-12-15",
                "hours_per_day": 4,
                "billable": True,
                "status": "confirmed",
            }
        ],
    }
    assert _query(gub, "/relay/float/people")["email"] == ["ada@anomaly.com"]
    sent = _query(gub, "/relay/float/tasks")
    assert sent["people_id"] == ["1"]
    # Float takes a range only as a pair: a lone start is closed a year on.
    assert (sent["start_date"], sent["end_date"]) == (["2026-11-01"], ["2027-11-01"])


async def test_a_project_by_name_with_everyone_on_it_named(gub):
    gub.routes[("GET", "/relay/float/projects")] = (200, _list(*PROJECTS))
    gub.routes[("GET", "/relay/float/tasks")] = (
        200,
        _list(
            {"task_id": 1, "people_id": 1, "project_id": 12, "hours": 8},
            {"task_id": 2, "people_id": None, "people_ids": [1, 2], "project_id": 12, "hours": 2},
        ),
    )
    gub.routes[("GET", "/relay/float/people/1")] = (200, ADA)
    gub.routes[("GET", "/relay/float/people/2")] = (200, ALAN)

    result = await float_allocations(project_name="chevy q3")

    assert [(a["person"], a["project"], a["hours_per_day"]) for a in result["allocations"]] == [
        ("Ada Lovelace", "Chevy Q3", 8),
        ("Ada Lovelace", "Chevy Q3", 2),
        ("Alan Turing", "Chevy Q3", 2),
    ]
    sent = _query(gub, "/relay/float/tasks")
    assert sent["project_id"] == ["12"] and "start_date" not in sent


async def test_several_matching_projects_are_returned_to_choose_from(gub):
    gub.routes[("GET", "/relay/float/projects")] = (200, _list(*PROJECTS))

    result = await float_allocations(project_name="nike")

    assert result["projects"] == [
        {"name": "Nike Holiday", "archived": False},
        {"name": "Nike Spring", "archived": True},
    ]
    assert not any(t.startswith("/relay/float/tasks") for _, t in gub.targets)


async def test_an_unknown_person_is_a_plain_empty_answer(gub):
    gub.routes[("GET", "/relay/float/people")] = (200, _list())
    result = await float_allocations(person_email="nobody@anomaly.com")
    assert result["allocations"] == [] and "nobody@anomaly.com" in result["note"]


async def test_no_grant_comes_back_with_the_relays_code(gub):
    gub.routes[("GET", "/relay/float/projects")] = (
        403,
        {"code": "FLOAT_NO_GRANT", "message": 'Reading Float needs the "Float reader" grant.'},
    )
    result = await float_allocations(project_name="chevy")
    assert result["error"] is True
    assert (result["status"], result["code"]) == (403, "FLOAT_NO_GRANT")


async def test_not_configured_reaches_the_caller(gub):
    gub.routes[("GET", "/relay/float/reports/people")] = (
        503,
        {"code": "FLOAT_NOT_CONFIGURED", "message": "no FLOAT_API_KEY"},
    )
    result = await float_availability(start_date="2026-12-01", end_date="2026-12-31")
    assert (result["status"], result["code"]) == (503, "FLOAT_NOT_CONFIGURED")


async def test_neither_person_nor_project_is_refused_without_a_call(gub):
    result = await float_allocations()
    assert result["error"] is True and gub.requests == []


async def test_availability_for_one_person_with_their_time_off(gub):
    gub.routes[("GET", "/relay/float/people")] = (200, _list(ADA))
    gub.routes[("GET", "/relay/float/reports/people")] = (
        200,
        _list(
            {
                "people_id": 1,
                "name": "Ada Lovelace",
                "department": "Design",
                "capacity": 120,
                "scheduled": 80,
                "unscheduled": 40,
                "timeoff": 16,
            }
        ),
    )
    gub.routes[("GET", "/relay/float/timeoffs")] = (
        200,
        _list(
            {
                "timeoff_id": 3,
                "timeoff_type_id": 7,
                "people_ids": [1, 2],
                "start_date": "2026-12-10",
                "end_date": "2026-12-11",
                "full_day": 1,
                "status": 2,
            },
            {"timeoff_id": 4, "timeoff_type_id": 7, "people_ids": [2], "full_day": 1},
        ),
    )
    gub.routes[("GET", "/relay/float/timeoff-types")] = (
        200,
        _list({"timeoff_type_id": 7, "timeoff_type_name": "Vacation"}),
    )

    result = await float_availability(
        person_email="ada@anomaly.com", start_date="2026-12-01", end_date="2026-12-31"
    )

    assert result["people"] == [
        {
            "person": "Ada Lovelace",
            "department": "Design",
            "capacity_hours": 120,
            "scheduled_hours": 80,
            "unscheduled_hours": 40,
            "time_off_hours": 16,
        }
    ]
    assert result["time_off"] == [
        {
            "type": "Vacation",
            "start_date": "2026-12-10",
            "end_date": "2026-12-11",
            "full_day": True,
            "hours_per_day": None,
            "tentative": False,
        }
    ]
    sent = _query(gub, "/relay/float/reports/people")
    assert sent == {"start_date": ["2026-12-01"], "end_date": ["2026-12-31"], "people_id": ["1"]}


async def test_availability_by_department_sorted_by_free_hours(gub):
    gub.routes[("GET", "/relay/float/reports/people")] = (
        200,
        _list(
            {"name": "Ada", "department": "Design", "unscheduled": 10},
            {"name": "Alan", "department": "Engineering", "unscheduled": 50},
            {"name": "Grace", "department": "Design Ops", "unscheduled": 30},
        ),
    )
    result = await float_availability(
        department="design", start_date="2026-12-01", end_date="2026-12-31"
    )
    assert [p["person"] for p in result["people"]] == ["Grace", "Ada"]
    assert "time_off" not in result


async def test_lists_page_until_float_runs_out(gub, monkeypatch):
    from gub_agent.tools import float as float_tool

    monkeypatch.setattr(float_tool, "PAGE", 2)
    gub.routes[("GET", "/relay/float/projects")] = (
        200,
        {
            "values": PROJECTS[:2],
            "pagination": {"page": 1, "pageCount": 1, "totalCount": 2, "perPage": 2},
        },
    )
    await float_allocations(project_name="nothing like it")
    assert len([t for _, t in gub.targets if t.startswith("/relay/float/projects")]) == 1
