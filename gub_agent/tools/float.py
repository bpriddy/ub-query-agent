"""
float.py — Float (float.com) as a resourcing source for `/ask`.

Tools:
  float_allocations   — who is scheduled on what, and for how many hours a day
  float_availability  — a person's capacity, scheduled hours and time off
  float_capacity      — everyone's capacity, bookings and free hours over a
                        window, by office, role, skill (tag) or team
                        (department); the same answer shape as runn_capacity

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
from .capacity import (
    Booking,
    Person,
    Scope,
    TimeOff,
    bounded,
    capacity_answer,
    load_of,
    parse_window,
    resolve_office,
    term_matches,
    to_date,
    ub_directory,
)

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


async def _collect(
    path: str, tool_context: Any, max_pages: int = MAX_PAGES, **params: Any
) -> tuple[list[dict], bool] | dict:
    """Every row of a list endpoint, up to `max_pages` pages, and whether it
    was cut short; an error dict on failure."""
    rows: list[dict] = []
    for page_no in range(1, max_pages + 1):
        page = await gub_get(path, tool_context, page=page_no, **{"per-page": PAGE}, **params)
        if _is_error(page):
            return page
        values = page.get("values") or []
        rows.extend(values)
        page_count = (page.get("pagination") or {}).get("pageCount")
        if len(values) < PAGE or (isinstance(page_count, int) and page_no >= page_count):
            return rows, False
    return rows, True


async def _all(path: str, tool_context: Any, **params: Any) -> list[dict] | dict:
    """Every row of a list endpoint, up to MAX_PAGES pages; an error dict on failure."""
    got = await _collect(path, tool_context, **params)
    return got if _is_error(got) else got[0]


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


def _no_record(email: str) -> str:
    return (
        f"Float has no person with the email {email}. Their time is not scheduled in Float: "
        "do not call them free or unbooked; say they have no Float record."
    )


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
                "note": _no_record(person_email),
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
            return {"people": [], "note": _no_record(person_email)}
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


# ── float_capacity ────────────────────────────────────────────────────────────

CAPACITY_MAX_PAGES = 25
#: Float's task and time-off statuses (developer.float.com).
TENTATIVE, CANCELED = 1, 4
#: Float project statuses that are not yet confirmed work: draft, tentative.
UNCONFIRMED_PROJECT = {0, 1}
DEFAULT_DAY_MINUTES = 480


def _tag_names(person: dict) -> list[str]:
    out = []
    for t in person.get("tags") or []:
        name = t.get("name") if isinstance(t, dict) else t
        if isinstance(name, str) and name:
            out.append(name)
    return out


def _week_pattern(person: dict) -> list[float] | None:
    """Minutes Monday..Sunday from `work_days_hours` (Float sends Sun..Sat),
    or None for the company's default week."""
    hours = person.get("work_days_hours")
    if not isinstance(hours, list) or len(hours) != 7 or not any(hours):
        return None
    return [float(hours[(i + 1) % 7] or 0) * 60 for i in range(7)]


def _pattern(week: list[float], start: date | None, end: date | None):
    def minutes(day: date) -> float:
        if (start and day < start) or (end and day > end):
            return 0
        return week[day.weekday()]

    return minutes


async def float_capacity(
    start: str,
    end: str,
    office: str | None = None,
    role: str | None = None,
    skill: str | None = None,
    team: str | None = None,
    person: str | None = None,
    tool_context: Any = None,
) -> dict:
    """
    Work out, from Float, how much time people have over a date range: each
    person's capacity (their working days and hours, less time off and
    public holidays, as Float reports it), their confirmed and tentative
    scheduled hours, and their free hours, with totals. All the arithmetic is
    done here; the answer has the same shape as `runn_capacity`.

    Use this for: who is free or available, who has capacity, who is over
    capacity or underbooked, who is on the bench, how booked or utilized a
    team or office is, who is part-time, who is off in a period, and who
    could cover for someone. One call covers everyone; filter by office,
    role, skill or team as the question says.

    Examples:
    - "who's free for art direction in LA next week?" → start="2026-11-23",
      end="2026-11-27", office="LA", role="art director"
    - "how booked is the Strategy department in Q1?" → start="2027-01-01",
      end="2027-03-31", team="Strategy"

    Args:
        start: First day of the window (YYYY-MM-DD)
        end: Last day of the window (YYYY-MM-DD), at most about six months on
        office: A UB office as the user said it ("LA", "New York", "London")
        role: A job role as the user said it ("art director", "strategy")
        skill: A skill, matched to Float's people tags ("motion design")
        team: A Float department ("Design")
        person: One person, by work email or full name, for a question about
            them alone

    Returns:
        The same as `runn_capacity`: 'start', 'end', 'weekdays',
        'public_holidays', 'totals', 'lists', 'people' (one row
        each, most free first, with capacity_h, confirmed_h, tentative_h,
        free_h, free_if_tentative_drops_h, booked_pct, works, leave,
        holidays, contract_ends, projects, flags), 'no_record' (UB staff in
        scope with no Float record: NOT free), 'warnings', or an 'error'
        dict with its 'code'.
    """
    window = parse_window(start, end)
    if isinstance(window, dict):
        return window
    office_rec, err = await resolve_office(office, tool_context)
    if err is not None:
        return err

    span = {"start_date": window.start.isoformat(), "end_date": window.end.isoformat()}
    pages = CAPACITY_MAX_PAGES
    fetched = await bounded(
        [
            lambda: _collect(f"{RELAY}/people", tool_context, pages, active="1"),
            lambda: gub_get(f"{RELAY}/reports/people", tool_context, **span),
            lambda: _collect(f"{RELAY}/tasks", tool_context, pages, **span),
            lambda: _collect(f"{RELAY}/projects", tool_context, pages),
            lambda: _collect(f"{RELAY}/roles", tool_context, pages),
            lambda: _collect(f"{RELAY}/timeoffs", tool_context, pages, **span),
            lambda: _collect(f"{RELAY}/public-holidays", tool_context, pages, **span),
            lambda: _collect(f"{RELAY}/holidays", tool_context, pages),
            lambda: ub_directory(tool_context, office_rec),
        ]
    )
    (people_r, report_r, tasks_r, projects_r, roles_r, timeoffs_r, public_r, team_hol_r,
     directory) = fetched  # fmt: skip
    for required in (people_r, tasks_r, projects_r):
        if _is_error(required):
            return required
    if _is_error(directory):
        return directory

    warnings: list[str] = []
    if any(r[1] for r in (people_r, tasks_r) if not _is_error(r)):
        warnings.append("Float returned more rows than one call reads; totals may be short.")

    def optional(result: Any, what: str) -> list[dict]:
        if _is_error(result):
            warnings.append(f"{what} could not be read from Float ({result.get('status')}).")
            return []
        return result[0] if isinstance(result, tuple) else result.get("values") or []

    report = {r.get("people_id"): r for r in optional(report_r, "The people report")}
    roles = {r.get("id"): r.get("name") for r in optional(roles_r, "Roles")}
    timeoffs = optional(timeoffs_r, "Time off")
    projects = {p.get("project_id"): p for p in projects_r[0]}

    # Public holidays by region, and the company's own (team) holidays.
    calendars: dict[str, list[tuple[date, str]]] = {}
    for h in optional(public_r, "Public holidays"):
        cal = f"region {h.get('region')}"
        for raw in h.get("dates") or []:
            if day := to_date(raw):
                calendars.setdefault(cal, []).append((day, h.get("name") or "Public holiday"))
    company: list[tuple[date, str]] = []
    for h in optional(team_hol_r, "Company holidays"):
        first, last = to_date(h.get("date")), to_date(h.get("end_date")) or to_date(h.get("date"))
        while first and last and first <= last:
            company.append((first, h.get("name") or "Company holiday"))
            first = date.fromordinal(first.toordinal() + 1)
    if company:
        calendars["company"] = company

    off_by_person: dict[int, list[TimeOff]] = {}
    for t in timeoffs:
        s, e = to_date(t.get("start_date")), to_date(t.get("end_date"))
        if not (s and e):
            continue
        whole = t.get("full_day") == 1 or not t.get("hours")
        minutes = None if whole else float(t["hours"]) * 60
        for pid in _people_of(t):
            off_by_person.setdefault(pid, []).append(TimeOff(s, e, "leave", minutes))

    tasks_by_person: dict[int, list[Booking]] = {}
    for t in tasks_r[0]:
        s, e = to_date(t.get("start_date")), to_date(t.get("end_date"))
        status = t.get("status")
        if not (s and e) or status == CANCELED:
            continue
        proj = projects.get(t.get("project_id")) or {}
        if proj.get("status") == CANCELED:
            continue
        confirmed = status != TENTATIVE and proj.get("status") not in UNCONFIRMED_PROJECT
        for pid in _people_of(t):
            tasks_by_person.setdefault(pid, []).append(
                Booking(s, e, float(t.get("hours") or 0) * 60, confirmed, proj.get("name"))
            )

    scope = Scope(office=office_rec, role=role, skill=skill, team=team, person=person)
    people = [p for p in people_r[0] if p.get("people_type_id") != 3]  # 3 = placeholder
    if skill:
        scope.skill_keys = {
            p.get("people_id") for p in people if any(term_matches(skill, t) for t in _tag_names(p))
        }

    planned: list[Person] = []
    for p in people:
        pid = p.get("people_id")
        hired, leaves = to_date(p.get("start_date")), to_date(p.get("end_date"))
        cal = f"region {p.get('region_id')}" if p.get("region_id") else None
        off = list(off_by_person.get(pid, []))
        for c in (cal, "company"):
            off += [TimeOff(d, d, "holiday", None, n) for d, n in calendars.get(c or "", [])]
        rep = report.get(pid) or {}
        reported = rep.get("capacity")
        week = _week_pattern(p)
        if week is None:
            # The company's default week, Monday to Friday. Its hours are not
            # in the API; the people report's capacity over the window gives
            # them, else a standard 8 h day.
            day_minutes = DEFAULT_DAY_MINUTES
            if isinstance(reported, (int, float)) and reported > 0:
                probe = Person(
                    key=pid,
                    name="",
                    email=None,
                    pattern=_pattern([1, 1, 1, 1, 1, 0, 0], hired, leaves),
                    time_off=off,
                )
                days = load_of(probe, window).capacity
                if days:
                    day_minutes = round(reported * 60 / days / 15) * 15
            week = [day_minutes] * 5 + [0, 0]
        dept = p.get("department") or {}
        planned.append(
            Person(
                key=pid,
                name=p.get("name") or f"Float person {pid}",
                email=p.get("email"),
                pattern=_pattern(week, hired, leaves),
                role=roles.get(p.get("role_id")) or p.get("job_title"),
                team=dept.get("name") if isinstance(dept, dict) else None,
                planner_offices=_tag_names(p),
                time_off=off,
                bookings=tasks_by_person.get(pid, []),
                employment_end=leaves,
                calendar=cal,
                reported_capacity=reported * 60 if isinstance(reported, (int, float)) else None,
            )
        )
    if "company" in calendars:
        for person in planned:
            person.calendar = person.calendar or "company"

    return capacity_answer(
        source="Float",
        people=planned,
        directory=directory,
        window=window,
        scope=scope,
        calendars=calendars,
        warnings=warnings,
    )


FLOAT_TOOLS = [float_allocations, float_availability, float_capacity]
