"""
runn.py — Runn as a source for `/ask` (gub-agent#89).

Tools:
  runn_allocations  — who is allocated to what, and for how long, from Runn

Runn is read through UB's relay (`/relay/runn/*`, UB #164), as the user, so
the "Runn reader" grant is checked in UB and nowhere else. Runn's API is shaped
for planners (ids, pages, minutes per day); this tool does the joins a planner
would, so the model spends one round per staffing question, not five.

Offered only on the `/ask` engine (`config.ASK_CONTRACT`): the chat bot's
engine keeps today's tools and prompt.

The relay's error codes come back as they are (`RUNN_NO_GRANT`,
`RUNN_RATE_LIMITED`, …), so the `/ask` turn can report Runn as a failed
source rather than fail the answer.
"""

from __future__ import annotations

from typing import Any

from ._client import gub_get

RELAY = "/relay/runn"
PAGE = 200
# Bounds on what one call may fetch through the relay. Runn allows 120
# requests a minute per key, shared by every UB caller; these keep one turn
# well inside it.
MAX_PAGES = 5
MAX_NAME_LOOKUPS = 30
MAX_PROJECT_MATCHES = 5


def _is_error(response: Any) -> bool:
    return isinstance(response, dict) and response.get("error") is True


async def _all(path: str, tool_context: Any, **params: Any) -> list[dict] | dict:
    """Every row of a list endpoint, up to MAX_PAGES pages; an error dict on failure."""
    rows: list[dict] = []
    cursor: str | None = None
    for _ in range(MAX_PAGES):
        page = await gub_get(path, tool_context, limit=PAGE, cursor=cursor, **params)
        if _is_error(page):
            return page
        rows.extend(page.get("values") or [])
        cursor = page.get("nextCursor")
        if not cursor:
            break
    return rows


def _full_name(person: dict) -> str | None:
    name = " ".join(p for p in (person.get("firstName"), person.get("lastName")) if p)
    return name or None


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


async def runn_allocations(
    person_email: str | None = None,
    project_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    tool_context: Any = None,
) -> dict:
    """
    Look up staffing allocations in Runn, the resource-planning tool: who is
    assigned to which project, over which dates, for how many hours a day.

    Use this for questions about who is staffed or allocated on a project, what
    a person is allocated to, or how much of someone's time a project takes.
    UB's own records say who is on a team or campaign; Runn says how their time
    is planned. Give a person's email (from UB's staff records) or a project's
    name, or both.

    Examples:
    - "who is allocated to the Nike holiday project?" → project_name="Nike holiday"
    - "what is ada@anomaly.com on next month?" → person_email="ada@anomaly.com",
      start_date="2026-12-01", end_date="2026-12-31"

    Args:
        person_email: The person's work email, exactly as in UB's staff record
        project_name: All or part of the Runn project's name (case-insensitive)
        start_date: Only allocations ending on or after this date (YYYY-MM-DD)
        end_date: Only allocations starting on or before this date (YYYY-MM-DD)

    Returns:
        dict with 'allocations' (person, email, project, start_date, end_date,
        hours_per_day, billable). When a project name matches several projects,
        'projects' lists them and nothing else: ask the user which one. An
        'error' dict when Runn can't be read, with its 'code' (RUNN_NO_GRANT
        means the user lacks the Runn reader grant).
    """
    if not person_email and not project_name:
        return {
            "error": True,
            "status": 400,
            "message": "Give a person_email, a project_name, or both.",
        }

    filters: dict[str, Any] = {"startDate": start_date, "endDate": end_date}
    person: dict | None = None
    project: dict | None = None

    if person_email:
        people = await _all(f"{RELAY}/people", tool_context)
        if _is_error(people):
            return people
        wanted = person_email.strip().lower()
        person = next((p for p in people if (p.get("email") or "").lower() == wanted), None)
        if person is None:
            return {"allocations": [], "note": f"Runn has no person with the email {person_email}."}
        filters["personId"] = person["id"]

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
            return {"allocations": [], "note": f"Runn has no project named like {project_name!r}."}
        if len(matches) > 1:
            return {
                "projects": [
                    {"name": p.get("name"), "archived": p.get("isArchived")}
                    for p in matches[:MAX_PROJECT_MATCHES]
                ],
                "note": "Several Runn projects match; ask the user which one.",
            }
        project = matches[0]
        filters["projectId"] = project["id"]

    assignments = await _all(f"{RELAY}/assignments", tool_context, **filters)
    if _is_error(assignments):
        return assignments

    if person is not None:
        people_by_id: dict[int, dict] | dict = {person["id"]: person}
    else:
        person_ids = list(dict.fromkeys(a["personId"] for a in assignments if a.get("personId")))
        people_by_id = await _names(person_ids, "people", tool_context)
        if _is_error(people_by_id):
            return people_by_id
    if project is not None:
        projects_by_id: dict[int, dict] | dict = {project["id"]: project}
    else:
        project_ids = list(dict.fromkeys(a["projectId"] for a in assignments if a.get("projectId")))
        projects_by_id = await _names(project_ids, "projects", tool_context)
        if _is_error(projects_by_id):
            return projects_by_id

    allocations = []
    for a in assignments:
        who = people_by_id.get(a.get("personId")) or {}
        what = projects_by_id.get(a.get("projectId")) or {}
        minutes = a.get("minutesPerDay")
        allocations.append(
            {
                "person": _full_name(who),
                "email": who.get("email"),
                "project": what.get("name"),
                "start_date": a.get("startDate"),
                "end_date": a.get("endDate"),
                "hours_per_day": round(minutes / 60, 2)
                if isinstance(minutes, (int, float))
                else None,
                "billable": a.get("isBillable"),
            }
        )
    return {"count": len(allocations), "allocations": allocations}


RUNN_TOOLS = [runn_allocations]
