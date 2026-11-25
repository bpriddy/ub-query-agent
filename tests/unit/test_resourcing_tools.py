"""
resourcing.py (gub-agent#117) over a fake relay: `gub_get` / `gub_post` are
replaced at the tool modules' own bindings (as test_fast_path.py does) by a
fake that answers `/relay/resourcing/*` with resourcing's query syntax
(tests/helpers.py `resourcing_read`) and UB's `/org/offices` + `/org/query`.

Pinned here: blended rows keep their leg (`source`) and every answer carries
each leg's `asOf`; ids are only unique within a leg; a date range reads the
open-ended allocations apart and merges them; an office becomes UB staff,
joined to the planner's people by email then by name, and capacity is read
for those people only; who could not be placed is said; role and team are
`.contains` filters on the people; paging follows `nextOffset` and stops at
null (or at the page cap, said); a long `.in` list is split under the
relay's length limit; and the relay's error codes come back as errors.
"""

from __future__ import annotations

from typing import Any

import pytest

from gub_agent.tools import capacity as capacity_mod
from gub_agent.tools import resourcing
from gub_agent.tools.resourcing import (
    MAX_IN_CHARS,
    _chunks,
    _get,
    resourcing_allocations,
    resourcing_capacity,
)
from tests.helpers import resourcing_read

AS_OF = {"runn": "2026-11-24T06:00:00Z", "float": "2026-11-23T22:00:00Z"}

LA = {"id": "office-la", "name": "Los Angeles", "syncCity": "Los Angeles"}
LONDON = {"id": "office-ldn", "name": "London", "syncCity": "London"}


class FakeRelay:
    def __init__(self, legs: dict[str, dict[str, list[dict]]]) -> None:
        #: leg → view → rows (without `source`, as a leg serves them)
        self.legs = legs
        self.calls: list[tuple[str, dict]] = []
        self.capacity: list[dict] | dict = []
        self.errors: dict[str, dict] = {}
        self.offices = [LA, LONDON]
        self.staff: list[dict] = []

    def _meta(self, legs: list[str], page: dict) -> dict:
        sources = [{"source": s, "asOf": AS_OF[s], "contractVersion": 1} for s in legs]
        return {**page, "sources": sources}

    async def get(self, path: str, tool_context: Any = None, **params: Any) -> Any:
        params = {k: str(v) for k, v in params.items() if v is not None}
        self.calls.append((path, params))
        for prefix, err in self.errors.items():
            if path.startswith(prefix):
                return err
        if path == "/org/offices":
            return self.offices
        view = path.removeprefix("/relay/resourcing/")
        if view == "capacity":
            if isinstance(self.capacity, dict):
                return self.capacity
            status, body = resourcing_read(self.capacity, params, reserved=("from", "to"))
            return body if status == 200 else {"error": True, "status": status, **body}
        legs = [s for s, views in self.legs.items() if view in views]
        if not legs:
            return {"error": True, "status": 404, "code": "RESOURCING_NOT_FOUND"}
        rows = [{"source": s, **r} for s in legs for r in self.legs[s][view]]
        status, body = resourcing_read(rows, params)
        if status != 200:
            return {"error": True, "status": status, **body}
        return {**body, "meta": self._meta(legs, body["meta"])}

    async def post(self, path: str, body: dict, tool_context: Any = None) -> Any:
        assert path == "/org/query" and body["entity"] == "staff"
        office = body["filter"].get("officeId", {})
        rows = [
            s
            for s in self.staff
            if ("eq" in office and s["officeId"] == office["eq"])
            or ("is_null" in office and s["officeId"] is None)
        ]
        return {"results": rows, "truncated": False}

    def sent(self, view: str) -> list[dict]:
        return [p for path, p in self.calls if path == f"/relay/resourcing/{view}"]


@pytest.fixture
def relay(monkeypatch):
    fake = FakeRelay({})
    monkeypatch.setattr(resourcing, "gub_get", fake.get)
    monkeypatch.setattr(capacity_mod, "gub_get", fake.get)
    monkeypatch.setattr(capacity_mod, "gub_post", fake.post)
    return fake


def _person(pid: str, name: str, email: str | None, **extra) -> dict:
    return {"id": pid, "name": name, "email": email, "team": "Creative", "is_archived": False,
            "role": "Art Director", "tags": [], **extra}  # fmt: skip


def _assignment(aid: str, person: str, project: str, start: str, end: str | None, **extra):
    return {"id": aid, "person_id": person, "project_id": project, "start_date": start,
            "end_date": end, "minutes_per_day": 240, "is_placeholder": False,
            "role": "Art Director", "is_billable": True, **extra}  # fmt: skip


def _staff(sid: str, name: str, email: str | None, office: dict | None) -> dict:
    return {"id": sid, "fullName": name, "email": email, "title": "Art Director",
            "officeId": office["id"] if office else None}  # fmt: skip


# ── resourcing_allocations ────────────────────────────────────────────────────


async def test_blended_rows_keep_their_leg_and_every_legs_freshness(relay):
    # The same ids in both legs: a row is joined within its own leg only.
    relay.legs = {
        "runn": {
            "people": [_person("1", "Ada Lovelace", "ada@anomaly.com")],
            "projects": [
                {
                    "id": "10",
                    "name": "Nike Holiday",
                    "client": "Nike",
                    "is_archived": False,
                    "is_tentative": False,
                }
            ],  # fmt: skip
            "assignments": [_assignment("100", "1", "10", "2026-11-02", "2026-12-18")],
        },
        "float": {
            "people": [_person("1", "Alan Turing", "alan@anomaly.com")],
            "projects": [
                {
                    "id": "10",
                    "name": "Nike Holiday",
                    "client": "Nike",
                    "is_archived": False,
                    "is_tentative": True,
                }
            ],  # fmt: skip
            "assignments": [_assignment("100", "1", "10", "2026-12-01", "2026-12-05")],
        },
    }

    result = await resourcing_allocations(project_name="nike holiday")

    rows = {(a["source"], a["person"], a["status"]) for a in result["allocations"]}
    assert rows == {("runn", "Ada Lovelace", "confirmed"), ("float", "Alan Turing", "tentative")}
    assert result["sources"] == [
        {"source": "float", "as_of": AS_OF["float"], "contract_version": 1},
        {"source": "runn", "as_of": AS_OF["runn"], "contract_version": 1},
    ]
    assert result["allocations"][0]["client"] == "Nike"


async def test_a_date_range_reads_open_ended_allocations_apart_and_merges_them(relay):
    relay.legs = {
        "runn": {
            "people": [_person("1", "Ada Lovelace", "ada@anomaly.com")],
            "projects": [
                {
                    "id": "10",
                    "name": "Retainer",
                    "client": None,
                    "is_archived": False,
                    "is_tentative": False,
                }
            ],  # fmt: skip
            "assignments": [
                _assignment("100", "1", "10", "2026-01-05", "2026-03-31"),  # over before
                _assignment("101", "1", "10", "2026-06-01", None),  # open-ended
                _assignment("102", "1", "10", "2026-11-30", "2026-12-04"),
                _assignment("103", "1", "10", "2027-02-01", None),  # starts after
            ],
        }
    }

    result = await resourcing_allocations(start_date="2026-11-23", end_date="2026-12-31")

    assert [(a["start_date"], a["end_date"]) for a in result["allocations"]] == [
        ("2026-06-01", None),
        ("2026-11-30", "2026-12-04"),
    ]
    assert result["allocations"][0]["open_ended"] is True
    # AND-only filters: one read for "ends after the start", one for "never ends".
    reads = relay.sent("assignments")
    assert {r.get("end_date.gte") for r in reads} == {"2026-11-23", None}
    assert {r.get("end_date.is_null") for r in reads} == {"true", None}
    assert all(r["start_date.lte"] == "2026-12-31" for r in reads)


async def test_a_person_with_no_email_in_the_planner_is_found_by_the_name_the_email_spells(
    relay,
):
    relay.legs = {
        "runn": {
            "people": [_person("7", "Jane Park", None)],
            "projects": [
                {
                    "id": "10",
                    "name": "Vela",
                    "client": None,
                    "is_archived": False,
                    "is_tentative": False,
                }
            ],  # fmt: skip
            "assignments": [_assignment("100", "7", "10", "2026-11-02", "2026-12-18")],
            "working_patterns": [
                {
                    "person_id": "7",
                    "start_date": "2025-01-01",
                    "end_date": None,
                    "minutes_per_day": 480,
                    "minutes_by_weekday": {
                        "monday": 480,
                        "tuesday": 480,
                        "wednesday": 480,
                        "thursday": 480,
                        "friday": 0,
                    },
                }
            ],  # fmt: skip
            "time_off": [
                {
                    "person_id": "7",
                    "start_date": "2026-12-21",
                    "end_date": "2027-01-04",
                    "minutes_per_day": None,
                    "type": "time_off",
                },
                {
                    "person_id": "7",
                    "start_date": "2026-11-26",
                    "end_date": "2026-11-26",
                    "minutes_per_day": None,
                    "type": "holiday",
                },
            ],  # fmt: skip
        }
    }

    result = await resourcing_allocations(person_email="Jane.Park@anomaly.com")

    assert result["allocations"][0]["person"] == "Jane Park"
    assert result["allocations"][0]["works"] == "Mon-Thu 8h"
    assert "by name" in result["notes"][0] and "no email" in result["notes"][0]
    assert [t["type"] for t in result["leave"]] == ["time_off"]  # holidays are not leave
    assert relay.sent("people")[0]["email"] == "jane.park@anomaly.com"


async def test_several_projects_named_alike_are_offered_back(relay):
    relay.legs = {
        "runn": {
            "projects": [
                {
                    "id": "10",
                    "name": "Nike Holiday",
                    "client": "Nike",
                    "is_archived": False,
                    "is_tentative": False,
                },
                {
                    "id": "11",
                    "name": "Nike Spring",
                    "client": "Nike",
                    "is_archived": True,
                    "is_tentative": False,
                },
            ],  # fmt: skip
        }
    }
    result = await resourcing_allocations(project_name="nike")
    assert [p["name"] for p in result["projects"]] == ["Nike Holiday", "Nike Spring"]
    assert relay.sent("assignments") == []


async def test_no_grant_comes_back_as_an_error_with_its_code(relay):
    relay.errors["/relay/resourcing"] = {
        "error": True,
        "status": 403,
        "message": "Access denied",
        "code": "RESOURCING_NO_GRANT",
    }
    result = await resourcing_allocations(project_name="nike")
    assert (result["error"], result["code"]) == (True, "RESOURCING_NO_GRANT")
    assert "Resourcing reader" in result["hint"]


# ── resourcing_capacity ───────────────────────────────────────────────────────


def _cap(pid: str, name: str, email: str | None, cap: float, alloc: float, source="runn"):
    return {"source": source, "id": pid, "personId": pid, "name": name, "email": email,
            "team": "Creative", "role": "Art Director", "is_archived": False,
            "capacityHours": cap, "allocatedHours": alloc,
            "availableHours": max(cap - alloc, 0),
            "utilizationPercent": round(100 * alloc / cap) if cap else None,
            "timeOffHours": 0}  # fmt: skip


async def test_an_office_becomes_ub_staff_joined_by_email_then_name(relay):
    relay.staff = [
        _staff("s1", "Marcus Bell", "marcus.bell@anomaly.com", LA),
        _staff("s2", "Sofia Reyes", "sofia.reyes@anomaly.com", LA),
        _staff("s3", "Kai Nakamura", "kai.nakamura@anomaly.com", LA),  # no planner record
    ]
    relay.legs = {
        "runn": {
            "people": [
                _person("1", "Marcus Bell", "marcus.bell@anomaly.com"),
                _person("2", "Sofia Reyes", None),  # no email: matched by name
                _person("3", "Nobody Known", None),  # no email, no match: unplaced
                _person("4", "Oliver Hughes", "oliver.hughes@anomaly.com"),  # London
            ]
        }
    }
    relay.capacity = [
        _cap("1", "Marcus Bell", "marcus.bell@anomaly.com", 24, 0),
        _cap("2", "Sofia Reyes", None, 24, 12),
        _cap("3", "Nobody Known", None, 40, 0),
        _cap("4", "Oliver Hughes", "oliver.hughes@anomaly.com", 40, 40),
    ]

    result = await resourcing_capacity("2026-11-23", "2026-11-27", office="LA")

    assert [r["person"] for r in result["people"]] == ["Marcus Bell", "Sofia Reyes"]
    assert result["people"][1]["office_from"] == "ub_name_match"
    assert all(r["office"] == "Los Angeles" for r in result["people"])
    # Capacity is read for the office's people only.
    [cap_call] = relay.sent("capacity")
    assert cap_call["id.in"] == "1,2" and cap_call["from"] == "2026-11-23"
    assert result["totals"][0] == {
        "people": 2,
        "capacity_h": 48.0,
        "allocated_h": 12.0,
        "available_h": 36.0,
        "time_off_h": 0.0,
        "utilization_pct": 25,
        "over_capacity": 0,
    }
    assert any("Sofia Reyes" in w for w in result["warnings"])
    assert any("1 people in the planner have no email" in w for w in result["warnings"])
    assert [p["person"] for p in result["no_record"]["people"]] == ["Kai Nakamura"]
    assert result["sources"][0] == {"source": "runn", "as_of": AS_OF["runn"],
                                    "contract_version": 1}  # fmt: skip


async def test_role_and_team_are_contains_filters_on_people(relay):
    relay.capacity = [
        _cap("1", "Ana", "ana@anomaly.com", 40, 50),
        {**_cap("2", "Ben", "ben@anomaly.com", 40, 0), "role": "Strategist"},
    ]
    result = await resourcing_capacity(
        "2027-01-04", "2027-01-08", role="strategy", team="planning teams"
    )
    [call] = relay.sent("capacity")
    assert call["role.contains"] == "strateg" and call["team.contains"] == "planning team"
    assert result["filters"]["role"] == "role contains 'strateg'"

    relay.calls.clear()
    everyone = await resourcing_capacity("2027-01-04", "2027-01-08")
    assert everyone["lists"][0]["over_capacity"] == "Ana"
    assert everyone["lists"][0]["nothing_booked"] == "Ben"
    assert "over_capacity" in everyone["people"][-1]["flags"]


async def test_one_person_by_email_reads_only_their_row(relay):
    relay.legs = {"runn": {"people": [_person("9", "Isaac Moore", "isaac.moore@anomaly.com"),
                                      _person("8", "Ivy Moore", "ivy@anomaly.com")]}}  # fmt: skip
    relay.capacity = [
        _cap("9", "Isaac Moore", "isaac.moore@anomaly.com", 160, 24),
        _cap("8", "Ivy Moore", "ivy@anomaly.com", 160, 24),
    ]
    result = await resourcing_capacity("2026-11-18", "2026-12-24", person="Isaac.Moore@anomaly.com")
    assert [r["person"] for r in result["people"]] == ["Isaac Moore"]
    assert relay.sent("capacity")[0]["id.in"] == "9"


async def test_capacity_off_comes_back_as_an_error_with_its_code(relay):
    relay.capacity = {
        "error": True,
        "status": 503,
        "message": "Company records returned an unexpected error (503)",
        "code": "RESOURCING_CAPACITY_OFF",
    }
    result = await resourcing_capacity("2026-11-23", "2026-11-27")
    assert (result["error"], result["code"]) == (True, "RESOURCING_CAPACITY_OFF")
    assert "resourcing_allocations" in result["hint"]


async def test_a_window_past_100_days_is_refused_without_a_call(relay):
    result = await resourcing_capacity("2027-01-01", "2027-04-30")
    assert result["error"] is True and "100 days" in result["message"]
    assert relay.calls == []


# ── reading the relay ─────────────────────────────────────────────────────────


async def test_paging_follows_next_offset_and_stops_at_null(relay):
    relay.legs = {"runn": {"people": [_person(str(i), f"P{i}", None) for i in range(2500)]}}
    got = await _get("/relay/resourcing/people", None)
    assert len(got.rows) == 2500 and got.cut is False
    assert [c.get("offset") for c in relay.sent("people")] == [None, "1000", "2000"]


async def test_paging_stops_at_the_page_cap_and_says_so(relay):
    relay.legs = {"runn": {"people": [_person(str(i), f"P{i}", None) for i in range(6000)]}}
    got = await _get("/relay/resourcing/people", None)
    assert len(got.rows) == 5000 and got.cut is True


def test_a_long_in_list_is_split_under_the_relays_limit():
    ids = [f"{70000000 + i}" for i in range(300)]
    parts = _chunks(ids)
    assert len(parts) > 1 and all(len(p) <= MAX_IN_CHARS for p in parts)
    assert ",".join(parts).split(",") == ids
