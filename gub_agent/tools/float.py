"""
float.py — Float (float.com) as a resourcing source for `/ask`.

Tools:
  float_allocations   — who is scheduled on what, and for how many hours a day
  float_availability  — a person's capacity, scheduled hours and time off

Float is read through UB's relay (`/relay/float/*`), as the user, so the
"Float reader" grant is checked in UB and nowhere else. It mirrors runn.py:
Float's API is shaped for planners (ids, pages, hours per day), and these tools
do the joins a planner would, so the model spends one round per question.

Offered only on the `/ask` engine (`config.ASK_CONTRACT`) and only when
`config.FLOAT_SOURCE` is on: an engine without it keeps today's tools.

The relay's error codes come back as they are (`FLOAT_NO_GRANT`,
`FLOAT_NOT_CONFIGURED`, `FLOAT_RATE_LIMITED`, …), so the turn can report Float
as unavailable rather than fail the answer.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from ._client import gub_get

RELAY = "/relay/float"
PAGE = 200
# Bounds on what one call may fetch through the relay. Float allows 200 GET
# requests a minute per company (30 on reports), shared by every UB caller.
MAX_PAGES = 5
MAX_NAME_LOOKUPS = 30
MAX_PROJECT_MATCHES = 5
# Float takes a date range only as a start/end pair; a one-sided range is
# closed this far from the given end. With no date at all, availability looks
# this far ahead from today.
OPEN_RANGE_DAYS = 365
DEFAULT_WINDOW_DAYS = 28

STATUS = {0: "draft", 1: "tentative", 2: "confirmed", 3: "complete", 4: "canceled"}


def _is_error(response: Any) -> bool:
    return isinstance(response, dict) and response.get("error") is True


async def _all(path: str, tool_context: Any, **params: Any) -> list[dict] | dict:
    """Every row of a list endpoint, up to MAX_PAGES pages; an error dict on failure."""
    rows: list[dict] = []
    for page_no in range(1, MAX_PAGES + 1):
        page = await gub_get(path, tool_context, page=page_no, **{"per-page": PAGE}, **params)
        if _is_error(page):
            return page
        values = page.get("values") or []
        rows.extend(values)
        page_count = (page.get("pagination") or {}).get("pageCount")
        if len(values) < PAGE or (isinstance(page_count, int) and page_no >= page_count):
            break
    return rows


def _range(start_date: str | None, end_date: str | None) -> dict[str, str]:
    """A start/end pair for Float, closing a one-sided range; {} for none."""
    if not start_date and not end_date:
        return {}
    span = timedelta(days=OPEN_RANGE_DAYS)
    try:
        if start_date and not end_date:
            end_date = (date.fromisoformat(start_date) + span).isoformat()
        elif end_date and not start_date:
            start_date = (date.fromisoformat(end_date) - span).isoformat()
    except ValueError:
        pass  # sent as given; the relay answers 400 FLOAT_BAD_QUERY
    return {"start_date": start_date or "", "end_date": end_date or ""}


def _people_of(row: dict) -> list[int]:
    """The people on an allocation or time off: `people_id`, else `people_ids`."""
    if row.get("people_id"):
        return [row["people_id"]]
    return [p for p in row.get("people_ids") or [] if p]


async def _person_by_email(email: str, tool_context: Any) -> dict | None:
    """The Float person with this email, None if there is none, or an error dict."""
    found = await gub_get(f"{RELAY}/people", tool_context, email=email.strip())
    if _is_error(found):
        return found
    wanted = email.strip().lower()
    return next(
        (p for p in found.get("values") or [] if (p.get("email") or "").lower() == wanted), None
    )


async def _names(ids: list[int], resource: str, tool_context: Any) -> dict[int, dict] | dict:
    """Records by id for the first MAX_NAME_LOOKUPS ids."""
    found: dict[int, dict] = {}
    for record_id in ids[:MAX_NAME_LOOKUPS]:
        record = await gub_get(f"{RELAY}/{resource}/{record_id}", tool_context)
        if _is_error(record):
            if record.get("status") == 404:
                continue
            return record
        found[record_id] = record
    return found


async def float_allocations(
    person_email: str | None = None,
    project_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    tool_context: Any = None,
) -> dict:
    """
    Look up scheduled allocations in Float, the resource-scheduling tool: who
    is scheduled on which project, over which dates, for how many hours a day.

    Use this for questions about who is scheduled or booked on a project, what
    a person is scheduled on, or how much of someone's time a project takes.
    UB's own records say who is on a team or campaign; Float says how their
    time is scheduled. Give a person's email (from UB's staff records) or a
    project's name, or both.

    Examples:
    - "who is booked on the Nike holiday project?" → project_name="Nike holiday"
    - "what is ada@anomaly.com scheduled on next month?" →
      person_email="ada@anomaly.com", start_date="2026-12-01", end_date="2026-12-31"

    Args:
        person_email: The person's work email, exactly as in UB's staff record
        project_name: All or part of the Float project's name (case-insensitive)
        start_date: Only allocations ending on or after this date (YYYY-MM-DD)
        end_date: Only allocations starting on or before this date (YYYY-MM-DD)

    Returns:
        dict with 'allocations' (person, email, project, start_date, end_date,
        hours_per_day, billable, status). When a project name matches several
        projects, 'projects' lists them and nothing else: ask the user which
        one. An 'error' dict when Float can't be read, with its 'code'
        (FLOAT_NO_GRANT means the user lacks the Float reader grant).
    """
    if not person_email and not project_name:
        return {
            "error": True,
            "status": 400,
            "message": "Give a person_email, a project_name, or both.",
        }

    filters: dict[str, Any] = _range(start_date, end_date)
    person: dict | None = None
    project: dict | None = None

    if person_email:
        found = await _person_by_email(person_email, tool_context)
        if _is_error(found):
            return found
        if found is None:
            return {
                "allocations": [],
                "note": f"Float has no person with the email {person_email}.",
            }
        person = found
        filters["people_id"] = person["people_id"]

    if project_name:
        projects = await _all(f"{RELAY}/projects", tool_context)
        if _is_error(projects):
            return projects
        wanted = project_name.strip().lower()
        matches = [p for p in projects if wanted in (p.get("name") or "").lower()]
        exact = [p for p in matches if (p.get("name") or "").lower() == wanted]
        if exact:
            matches = exact[:1]
        if not matches:
            return {"allocations": [], "note": f"Float has no project named like {project_name!r}."}
        if len(matches) > 1:
            return {
                "projects": [
                    {"name": p.get("name"), "archived": p.get("active") == 0}
                    for p in matches[:MAX_PROJECT_MATCHES]
                ],
                "note": "Several Float projects match; ask the user which one.",
            }
        project = matches[0]
        filters["project_id"] = project["project_id"]

    tasks = await _all(f"{RELAY}/tasks", tool_context, **filters)
    if _is_error(tasks):
        return tasks

    if person is not None:
        people_by_id: dict[int, dict] | dict = {person["people_id"]: person}
    else:
        person_ids = list(dict.fromkeys(p for t in tasks for p in _people_of(t)))
        people_by_id = await _names(person_ids, "people", tool_context)
        if _is_error(people_by_id):
            return people_by_id
    if project is not None:
        projects_by_id: dict[int, dict] | dict = {project["project_id"]: project}
    else:
        project_ids = list(dict.fromkeys(t["project_id"] for t in tasks if t.get("project_id")))
        projects_by_id = await _names(project_ids, "projects", tool_context)
        if _is_error(projects_by_id):
            return projects_by_id

    allocations = []
    for t in tasks:
        what = projects_by_id.get(t.get("project_id")) or {}
        for people_id in _people_of(t):
            if person is not None and people_id != person["people_id"]:
                continue
            who = people_by_id.get(people_id) or {}
            hours = t.get("hours")
            allocations.append(
                {
                    "person": who.get("name"),
                    "email": who.get("email"),
                    "project": what.get("name"),
                    "start_date": t.get("start_date"),
                    "end_date": t.get("end_date"),
                    "hours_per_day": round(hours, 2) if isinstance(hours, (int, float)) else None,
                    "billable": t.get("billable") == 1 if t.get("billable") is not None else None,
                    "status": STATUS.get(t.get("status")),
                }
            )
    return {"count": len(allocations), "allocations": allocations}


async def float_availability(
    person_email: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    department: str | None = None,
    tool_context: Any = None,
) -> dict:
    """
    Look up capacity in Float, the resource-scheduling tool: for a date range,
    how many hours each person can work, how many are scheduled, how many are
    free, and their time off.

    Use this for questions about who is free, how busy someone is, or who is
    off. Give a person's email (from UB's staff records) for one person, or a
    department name for a group; with neither, everyone in Float is returned.

    Examples:
    - "is ada@anomaly.com free next week?" → person_email="ada@anomaly.com",
      start_date="2026-11-23", end_date="2026-11-27"
    - "who in Design has time in December?" → department="Design",
      start_date="2026-12-01", end_date="2026-12-31"

    Args:
        person_email: The person's work email, exactly as in UB's staff record
        start_date: First day of the range (YYYY-MM-DD); default today
        end_date: Last day of the range (YYYY-MM-DD); default four weeks on
        department: All or part of a Float department's name (case-insensitive)

    Returns:
        dict with 'people' (person, department, capacity_hours, scheduled_hours,
        unscheduled_hours, time_off_hours) sorted by most unscheduled first,
        and with one person, their 'time_off' entries. Capacity already
        excludes time off and holidays. An 'error' dict when Float can't be
        read, with its 'code'.
    """
    if not start_date:
        start_date = date.today().isoformat()
    if not end_date:
        try:
            end_date = (
                date.fromisoformat(start_date) + timedelta(days=DEFAULT_WINDOW_DAYS)
            ).isoformat()
        except ValueError:
            return {"error": True, "status": 400, "message": "start_date must be YYYY-MM-DD."}

    params: dict[str, Any] = {"start_date": start_date, "end_date": end_date}
    person: dict | None = None
    if person_email:
        found = await _person_by_email(person_email, tool_context)
        if _is_error(found):
            return found
        if found is None:
            return {"people": [], "note": f"Float has no person with the email {person_email}."}
        person = found
        params["people_id"] = person["people_id"]

    report = await gub_get(f"{RELAY}/reports/people", tool_context, **params)
    if _is_error(report):
        return report
    rows = report.get("values") or []
    if department:
        wanted = department.strip().lower()
        rows = [r for r in rows if wanted in (r.get("department") or "").lower()]

    people = sorted(
        (
            {
                "person": r.get("name"),
                "department": r.get("department"),
                "capacity_hours": r.get("capacity"),
                "scheduled_hours": r.get("scheduled"),
                "unscheduled_hours": r.get("unscheduled"),
                "time_off_hours": r.get("timeoff"),
            }
            for r in rows
        ),
        key=lambda p: -(p["unscheduled_hours"] or 0),
    )
    result: dict[str, Any] = {
        "start_date": start_date,
        "end_date": end_date,
        "count": len(people),
        "people": people,
    }
    if person is not None:
        result["email"] = person.get("email")
        timeoffs = await _all(
            f"{RELAY}/timeoffs", tool_context, start_date=start_date, end_date=end_date
        )
        if _is_error(timeoffs):
            return timeoffs
        types = await _all(f"{RELAY}/timeoff-types", tool_context)
        type_names = (
            {}
            if _is_error(types)
            else {t.get("timeoff_type_id"): t.get("timeoff_type_name") for t in types}
        )
        result["time_off"] = [
            {
                "type": type_names.get(t.get("timeoff_type_id")),
                "start_date": t.get("start_date"),
                "end_date": t.get("end_date"),
                "full_day": t.get("full_day") == 1,
                "hours_per_day": t.get("hours"),
                "tentative": t.get("status") == 1,
            }
            for t in timeoffs
            if person["people_id"] in _people_of(t)
        ]
    return result


FLOAT_TOOLS = [float_allocations, float_availability]
