"""
runn.py — Runn as a source for `/ask` (gub-agent#89).

Tools:
  runn_allocations  — who is allocated to what, and for how long, from Runn
  runn_capacity     — everyone's capacity, bookings and free hours over a
                      window, by office, role, skill or team

Runn is read through UB's relay (`/relay/runn/*`, UB #164), as the user, so
the "Runn reader" grant is checked in UB and nowhere else. Runn's API is shaped
for planners (ids, pages, minutes per day); these tools do the joins and the
arithmetic a planner would, in code, so the model spends one round per
staffing question, not five, and never sums hours itself.

Offered only on the `/ask` engine (`config.ASK_CONTRACT`): the chat bot's
engine keeps today's tools and prompt.

The relay's error codes come back as they are (`RUNN_NO_GRANT`,
`RUNN_RATE_LIMITED`, …), so the `/ask` turn can report Runn as a failed
source rather than fail the answer.
"""

from __future__ import annotations

import bisect
from datetime import date
from typing import Any

from ._client import gub_get
from .capacity import (
    WEEKDAYS,
    Booking,
    Person,
    Scope,
    TimeOff,
    bounded,
    capacity_answer,
    is_error,
    normalize,
    parse_window,
    resolve_office,
    term_matches,
    to_date,
    ub_directory,
    working_days,
)

RELAY = "/relay/runn"
PAGE = 200
# Bounds on what one call may fetch through the relay. Runn allows 120
# requests a minute per key, shared by every UB caller; these keep one turn
# well inside it (a capacity call is about a dozen requests).
MAX_PAGES = 5
CAPACITY_MAX_PAGES = 25
MAX_PROJECT_MATCHES = 5
MAX_ALLOCATIONS = 150


def _is_error(response: Any) -> bool:
    return is_error(response)


async def _collect(
    path: str, tool_context: Any, max_pages: int = MAX_PAGES, **params: Any
) -> tuple[list[dict], bool] | dict:
    """Every row of a list endpoint, up to `max_pages` pages, and whether it
    was cut short; an error dict on failure."""
    rows: list[dict] = []
    cursor: str | None = None
    for _ in range(max_pages):
        page = await gub_get(path, tool_context, limit=PAGE, cursor=cursor, **params)
        if _is_error(page):
            return page
        rows.extend(page.get("values") or [])
        cursor = page.get("nextCursor")
        if not cursor:
            return rows, False
    return rows, True


async def _all(path: str, tool_context: Any, **params: Any) -> list[dict] | dict:
    """Every row of a list endpoint, up to MAX_PAGES pages; an error dict on failure."""
    got = await _collect(path, tool_context, **params)
    return got if _is_error(got) else got[0]


def _full_name(person: dict) -> str | None:
    name = " ".join(p for p in (person.get("firstName"), person.get("lastName")) if p)
    return name or None


# ── a person's working pattern, from their contracts ─────────────────────────


class Contracts:
    """A person's contracts, by start date: minutes on any day, role on any day."""

    def __init__(self, rows: list[dict]) -> None:
        dated = [(to_date(c.get("startDate")) or date.min, c) for c in rows]
        dated.sort(key=lambda x: x[0])
        self.starts = [d for d, _ in dated]
        self.rows = [c for _, c in dated]

    def on(self, day: date) -> dict | None:
        i = bisect.bisect_right(self.starts, day) - 1
        while i >= 0:
            c = self.rows[i]
            end = to_date(c.get("endDate"))
            if end is None or day <= end:
                return c
            i -= 1
        return None

    def minutes(self, day: date) -> float:
        c = self.on(day)
        if c is None:
            return 0
        rostered = c.get("rosteredDays")
        if isinstance(rostered, dict):
            return float(rostered.get(WEEKDAYS[day.weekday()]) or 0)
        return float(c.get("minutesPerDay") or 0) if day.weekday() < 5 else 0

    def current(self, day: date) -> dict | None:
        """The contract on `day`, else the latest one started before it, else the first."""
        c = self.on(day)
        if c is not None:
            return c
        i = bisect.bisect_right(self.starts, day) - 1
        return self.rows[i] if i >= 0 else (self.rows[0] if self.rows else None)

    def end(self) -> date | None:
        """When the last contract ends; None while one is open-ended."""
        if not self.rows:
            return None
        ends = [to_date(c.get("endDate")) for c in self.rows]
        return None if any(e is None for e in ends) else max(e for e in ends if e)


def _by_person(rows: list[dict]) -> dict[Any, list[dict]]:
    out: dict[Any, list[dict]] = {}
    for r in rows:
        out.setdefault(r.get("personId"), []).append(r)
    return out


def _planner_offices(person: dict) -> list[str]:
    """Office names Runn holds for a person: a select custom field named like
    "office" or "location", then the person's tags."""
    out: list[str] = []
    for f in ((person.get("customFields") or {}).get("select")) or []:
        if any(w in normalize(f.get("name")) for w in ("office", "location", "studio")):
            out.extend(v.get("name") for v in f.get("values") or [] if v.get("name"))
    out.extend(t.get("name") for t in person.get("tags") or [] if t.get("name"))
    return out


# ── runn_allocations ──────────────────────────────────────────────────────────


async def _find_person(email: str, people: list[dict]) -> tuple[dict | None, str]:
    """The Runn person for a UB email: by email, else by the name the email
    spells (jane.park@… → Jane Park) when exactly one person has it."""
    wanted = email.strip().lower()
    hit = next((p for p in people if (p.get("email") or "").lower() == wanted), None)
    if hit is not None:
        return hit, "email"
    spelled = normalize(wanted.split("@", 1)[0].replace(".", " ").replace("_", " "))
    same = [p for p in people if normalize(_full_name(p)) == spelled]
    if len(same) == 1:
        return same[0], "name"
    return None, ""


async def runn_allocations(
    person_email: str | None = None,
    project_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    tool_context: Any = None,
) -> dict:
    """
    Look up staffing allocations in Runn, the resource-planning tool: who is
    assigned to which project, over which dates, for how many hours a day,
    and whether that work is confirmed or tentative.

    Use this for who is staffed on a project, what a person is on, or every
    booking in a date range. Give a person's email (from UB's staff records),
    a project's name, a date range, or any mix. For who is FREE, over
    capacity, or how booked a team or office is, use `runn_capacity`: it
    does the arithmetic.

    Examples:
    - "who is allocated to the Nike holiday project?" → project_name="Nike holiday"
    - "what is ada@anomaly.com on next month?" → person_email="ada@anomaly.com",
      start_date="2026-12-01", end_date="2026-12-31"
    - "which projects have open placeholder roles in December?" →
      start_date="2026-12-01", end_date="2026-12-31"

    Args:
        person_email: The person's work email, exactly as in UB's staff record
        project_name: All or part of the Runn project's name (case-insensitive)
        start_date: Only allocations ending on or after this date (YYYY-MM-DD)
        end_date: Only allocations starting on or before this date (YYYY-MM-DD)

    Returns:
        dict with 'allocations' (person, email, project, start_date, end_date,
        hours_per_day, billable, status "confirmed"/"tentative", placeholder,
        role, works = the person's working days); for a person, their
        'leave'; for a project, a 'summary' (people on it and their summed
        hours a day on start_date, else today). A placeholder row is an
        unfilled role, not a person. When a project name matches several
        projects, 'projects' lists them and nothing else: ask the user which
        one. A person with no Runn record comes back with a note saying so:
        that is not the same as free. An 'error' dict when Runn can't be
        read, with its 'code' (RUNN_NO_GRANT means the user lacks the Runn
        reader grant).
    """
    if not person_email and not project_name and not start_date and not end_date:
        return {
            "error": True,
            "status": 400,
            "message": "Give a person_email, a project_name, a date range, or a mix.",
        }

    filters: dict[str, Any] = {"startDate": start_date, "endDate": end_date}
    notes: list[str] = []
    person: dict | None = None

    projects = await _all(f"{RELAY}/projects", tool_context)
    if _is_error(projects):
        return projects
    projects_by_id = {p.get("id"): p for p in projects}

    people = await _all(f"{RELAY}/people", tool_context, includePlaceholders="true")
    if _is_error(people):
        return people
    people_by_id = {p.get("id"): p for p in people}

    if person_email:
        person, how = await _find_person(person_email, people)
        if person is None:
            return {
                "allocations": [],
                "note": f"Runn has no person with the email {person_email}. Their time is not "
                "planned in Runn: do not call them free or unbooked; say they have no Runn "
                "record.",
            }
        if how == "name":
            notes.append(
                f"Matched {person_email} to Runn's {_full_name(person)} by name: Runn holds "
                f"a different email ({person.get('email')})."
            )
        filters["personId"] = person["id"]

    if project_name:
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
        filters["projectId"] = matches[0]["id"]

    got = await _collect(f"{RELAY}/assignments", tool_context, CAPACITY_MAX_PAGES, **filters)
    if _is_error(got):
        return got
    assignments, cut = got
    assignments = [a for a in assignments if not a.get("isTemplate")]

    # Role names and working days are context, not the answer: a relay
    # without those routes still answers, with a note.
    roles: dict[Any, str] = {}
    contracts: dict[Any, Contracts] = {}
    calls = [
        lambda: _all(f"{RELAY}/roles", tool_context),
        lambda: _all(f"{RELAY}/contracts", tool_context),
    ]
    if person is not None:
        # A person's leave answers "when is X back?" and "is X off?".
        calls.append(
            lambda: _all(
                f"{RELAY}/time-offs/leave",
                tool_context,
                personId=person["id"],
                startDate=start_date or date.today().isoformat(),
                endDate=end_date,
            )
        )
    extra = await bounded(calls)
    if _is_error(extra[0]):
        notes.append("Role names could not be read from Runn.")
    else:
        roles = {r.get("id"): r.get("name") for r in extra[0]}
    if _is_error(extra[1]):
        notes.append("Working days could not be read from Runn.")
    else:
        contracts = {k: Contracts(v) for k, v in _by_person(extra[1]).items()}
    around = to_date(start_date) or date.today()

    allocations = []
    for a in assignments:
        who = people_by_id.get(a.get("personId")) or {}
        what = projects_by_id.get(a.get("projectId")) or {}
        minutes = a.get("minutesPerDay")
        placeholder = bool(a.get("isPlaceholder"))
        row: dict[str, Any] = {
            "person": _full_name(who),
            "email": who.get("email"),
            "project": what.get("name"),
            "start_date": a.get("startDate"),
            "end_date": a.get("endDate"),
            "hours_per_day": round(minutes / 60, 2) if isinstance(minutes, (int, float)) else None,
            "billable": a.get("isBillable"),
            "status": "tentative" if what.get("isConfirmed") is False else "confirmed",
            "role": roles.get(a.get("roleId")),
        }
        if placeholder:
            row["placeholder"] = True
        c = contracts.get(a.get("personId"))
        if c is not None and not placeholder:
            row["works"] = working_days(
                Person(key=None, name="", email=None, pattern=c.minutes), around
            )
        allocations.append(row)

    out: dict[str, Any] = {"count": len(allocations), "allocations": allocations[:MAX_ALLOCATIONS]}
    if person is not None:
        if _is_error(extra[2]):
            notes.append("Leave could not be read from Runn.")
        else:
            out["leave"] = [
                {"person": _full_name(person), "start_date": t.get("startDate"),
                 "end_date": t.get("endDate"),
                 "whole_days": t.get("minutesPerDay") is None}
                for t in sorted(extra[2], key=lambda t: t.get("startDate") or "")
            ] or [{"person": _full_name(person), "leave": "none booked"}]  # fmt: skip
    if "projectId" in filters:
        # "how many people are on X, and how many hours a day?" on the day
        # asked about (start_date, else today): the sum is done here.
        on = to_date(start_date) or date.today()
        live = [
            a
            for a in allocations
            if (to_date(a["start_date"]) or date.min) <= on <= (to_date(a["end_date"]) or date.max)
        ]
        people_on = {a["person"] for a in live if not a.get("placeholder")}
        out["summary"] = [
            {
                "on_date": on.isoformat(),
                "people": len(people_on),
                "hours_per_day": round(sum(a["hours_per_day"] or 0 for a in live), 2),
                "open_placeholder_roles": sum(1 for a in live if a.get("placeholder")),
            }
        ]
    if len(allocations) > MAX_ALLOCATIONS or cut:
        notes.append("More allocations exist than are listed; narrow the dates or the filter.")
    if notes:
        out["notes"] = notes
    return out


# ── runn_capacity ─────────────────────────────────────────────────────────────


async def runn_capacity(
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
    Work out, from Runn, how much time people have over a date range: each
    person's capacity (their contract's hours on their working days, less
    leave and public holidays), their confirmed and tentative bookings, and
    their free hours, with totals. All the arithmetic is done here.

    Use this for: who is free or available, who has capacity, who is over
    capacity or underbooked, who is on the bench, how booked or utilized a
    team or office is, who is part-time, who is off or on leave in a period,
    and who could cover for someone. One call covers everyone; filter by
    office, role, skill or team as the question says.

    Examples:
    - "who's free for art direction in LA next week?" → start="2026-11-23",
      end="2026-11-27", office="LA", role="art director"
    - "how booked is the strategy team in Q1?" → start="2027-01-01",
      end="2027-03-31", role="strategist"
    - "who has motion design skills and is free next week?" →
      skill="motion design", start/end the week

    Args:
        start: First day of the window (YYYY-MM-DD)
        end: Last day of the window (YYYY-MM-DD), at most about six months on
        office: A UB office as the user said it ("LA", "New York", "London")
        role: A job role as the user said it ("art director", "strategy")
        skill: A Runn skill ("motion design")
        team: A Runn team ("Design")
        person: One person, by work email or full name, for a question about
            them alone ("is Isaac overbooked before Christmas?")

    Returns:
        dict with 'start', 'end', 'weekdays' (Mon-Fri days in the window),
        'public_holidays' that fall in it, 'totals' (one row: capacity_h,
        confirmed_h, tentative_h, free_h, confirmed_pct, booked_pct,
        over_capacity count), 'lists' (on_leave, over_capacity,
        nothing_booked, part_time: every name, whole) and 'people', one row
        each, most free first: capacity_h, confirmed_h, tentative_h, free_h,
        free_if_tentative_drops_h, booked_pct, works (their working days),
        leave, holidays, contract_ends, projects, and flags (over_capacity,
        part_time, on_leave_whole_window, contract_ends_in_window). 'no_record'
        lists UB staff in scope with no Runn record: they are NOT free.
        'warnings' says what could not be read or was matched by name. An
        'error' dict when Runn or UB can't be read, with its 'code'.
    """
    window = parse_window(start, end)
    if isinstance(window, dict):
        return window
    office_rec, err = await resolve_office(office, tool_context)
    if err is not None:
        return err

    span = {"startDate": window.start.isoformat(), "endDate": window.end.isoformat()}
    pages = CAPACITY_MAX_PAGES
    fetched = await bounded(
        [
            lambda: _collect(f"{RELAY}/people", tool_context, pages),
            lambda: _collect(f"{RELAY}/contracts", tool_context, pages),
            lambda: _collect(f"{RELAY}/assignments", tool_context, pages, **span),
            lambda: _collect(f"{RELAY}/projects", tool_context, pages),
            lambda: _collect(f"{RELAY}/roles", tool_context, pages),
            lambda: _collect(f"{RELAY}/teams", tool_context, pages),
            lambda: _collect(f"{RELAY}/time-offs/leave", tool_context, pages, **span),
            lambda: _collect(f"{RELAY}/time-offs/rostered-off", tool_context, pages),
            lambda: _collect(f"{RELAY}/holiday-groups", tool_context, pages),
            lambda: ub_directory(tool_context, office_rec),
        ]
    )
    (people_r, contracts_r, assign_r, projects_r, roles_r, teams_r, leave_r, rostered_r,
     groups_r, directory) = fetched  # fmt: skip
    for required in (people_r, contracts_r, assign_r, projects_r):
        if _is_error(required):
            return required
    if _is_error(directory):
        return directory

    warnings: list[str] = []
    if any(r[1] for r in (people_r, contracts_r, assign_r, projects_r) if not _is_error(r)):
        warnings.append("Runn returned more rows than one call reads; totals may be short.")

    def optional(result: Any, what: str) -> list[dict]:
        if _is_error(result):
            warnings.append(f"{what} could not be read from Runn ({result.get('status')}).")
            return []
        return result[0]

    roles = {r.get("id"): r.get("name") for r in optional(roles_r, "Roles")}
    teams = {t.get("id"): t.get("name") for t in optional(teams_r, "Teams")}
    leave = _by_person(optional(leave_r, "Leave"))
    rostered = _by_person(optional(rostered_r, "Rostered days off"))
    groups = {g.get("id"): g for g in optional(groups_r, "Public holiday calendars")}

    people = [p for p in people_r[0] if not p.get("isArchived")]
    used_groups = sorted({p.get("holidaysGroupId") for p in people if p.get("holidaysGroupId")})
    holiday_pages = await bounded(
        [
            lambda g=g: _collect(f"{RELAY}/holiday-groups/{g}/holidays", tool_context, pages)
            for g in used_groups
        ]
    )
    calendars: dict[str, list[tuple[date, str]]] = {}
    group_names: dict[Any, str] = {}
    for g, got in zip(used_groups, holiday_pages, strict=True):
        name = (groups.get(g) or {}).get("name") or f"calendar {g}"
        group_names[g] = name
        days = []
        for h in optional(got, f"Holidays of {name}"):
            day = to_date(h.get("observed")) or to_date(h.get("date"))
            if day:
                days.append((day, h.get("name") or "Public holiday"))
        calendars[name] = days

    scope = Scope(office=office_rec, role=role, skill=skill, team=team, person=person)
    if skill:
        skills_r = await _collect(f"{RELAY}/skills", tool_context, pages)
        if _is_error(skills_r):
            return skills_r
        wanted = [s for s in skills_r[0] if term_matches(skill, s.get("name"))]
        if not wanted:
            names = sorted(s.get("name") for s in skills_r[0] if s.get("name"))
            return {
                "people": [],
                "note": f"Runn has no skill like {skill!r}. Its skills: {', '.join(names[:40])}.",
            }
        holders = await bounded(
            [
                lambda s=s: _collect(f"{RELAY}/skills/{s['id']}/people", tool_context, pages)
                for s in wanted
            ]
        )
        scope.skill_keys = set()
        for got in holders:
            if _is_error(got):
                return got
            scope.skill_keys.update(p.get("id") for p in got[0])
        scope.skill = ", ".join(s.get("name") for s in wanted)

    projects = {p.get("id"): p for p in projects_r[0]}
    assignments = _by_person(
        [a for a in assign_r[0] if not a.get("isTemplate") and not a.get("isPlaceholder")]
    )
    contracts = {k: Contracts(v) for k, v in _by_person(contracts_r[0]).items()}

    planned: list[Person] = []
    for p in people:
        pid = p.get("id")
        c = contracts.get(pid) or Contracts([])
        current = c.current(window.start)
        off = [
            TimeOff(s, e, "leave", t.get("minutesPerDay"))
            for t in leave.get(pid, [])
            if (s := to_date(t.get("startDate"))) and (e := to_date(t.get("endDate")))
        ] + [
            TimeOff(s, e, "rostered_off", t.get("minutesPerDay"))
            for t in rostered.get(pid, [])
            if (s := to_date(t.get("startDate"))) and (e := to_date(t.get("endDate")))
        ]
        group = group_names.get(p.get("holidaysGroupId"))
        off += [TimeOff(d, d, "holiday", None, n) for d, n in calendars.get(group or "", [])]
        bookings = []
        for a in assignments.get(pid, []):
            s, e = to_date(a.get("startDate")), to_date(a.get("endDate"))
            if not (s and e):
                continue
            proj = projects.get(a.get("projectId")) or {}
            bookings.append(
                Booking(
                    s,
                    e,
                    float(a.get("minutesPerDay") or 0),
                    confirmed=proj.get("isConfirmed") is not False,
                    project=proj.get("name"),
                )
            )
        planned.append(
            Person(
                key=pid,
                name=_full_name(p) or f"Runn person {pid}",
                email=p.get("email"),
                pattern=c.minutes,
                role=roles.get((current or {}).get("roleId")),
                team=teams.get(p.get("teamId")),
                planner_offices=_planner_offices(p),
                time_off=off,
                bookings=bookings,
                employment_end=c.end(),
                calendar=group,
            )
        )

    return capacity_answer(
        source="Runn",
        people=planned,
        directory=directory,
        window=window,
        scope=scope,
        calendars=calendars,
        warnings=warnings,
    )


RUNN_TOOLS = [runn_allocations, runn_capacity]
