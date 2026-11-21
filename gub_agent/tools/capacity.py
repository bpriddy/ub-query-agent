"""
capacity.py — who has time over a date range, worked out in code.

Shared by `runn_capacity` (runn.py) and `float_capacity` (float.py). Each
source reads its own planner and hands over the same plain shapes: a person's
working pattern, their time off, and their bookings. Everything after that is
here, so both tools count the same way and answer in the same shape:

  capacity   the minutes a person is contracted for on each day of the
             window, less leave, rostered days off and public holidays
  confirmed  bookings on confirmed work, counted only on days with capacity
             (a planner does not count work on a day off)
  tentative  the same for tentative work, kept apart: it may not happen
  free       capacity less every booking, never below zero; and
             `free_if_tentative_drops`, capacity less confirmed work only

Office comes from UB, not the planner: UB staff joined on email, then on the
normalized full name (flagged `office_from: "ub_name_match"`), then the
planner's own office tag or field (flagged `office_from: "planner"`). UB staff
in scope who have no planner record are listed apart as "no record": their
time is not planned anywhere, so they are never counted as free.

The model is bad at summing hours across dozens of rows and good at reading
a short table, so this returns one compact row per person plus totals.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from ._client import gub_get, gub_post

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
SHORT = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
#: The longest window one call computes: a quarter and a bit.
MAX_WINDOW_DAYS = 190
#: Rows returned in full; past this the rest are counted in the totals only.
MAX_ROWS = 80
MAX_NO_RECORD = 25
#: A working day shorter than this, or a week of fewer than five days, is
#: part-time (a full day is 7.5 h in London, 8 h in the US).
PART_TIME_DAY_MINUTES = 420
#: Concurrent relay/UB requests one call makes.
CONCURRENCY = 4
#: UB's /org/query returns at most this many rows and has no paging.
UB_QUERY_LIMIT = 100

# Short forms people use for an office, beside the office's own name.
OFFICE_ALIASES = {
    "la": "los angeles",
    "nyc": "new york",
    "ny": "new york",
    "sf": "san francisco",
    "ldn": "london",
}


# ── plain shapes the sources fill ─────────────────────────────────────────────


@dataclass
class Booking:
    start: date
    end: date
    minutes_per_day: float
    confirmed: bool
    project: str | None = None


@dataclass
class TimeOff:
    start: date
    end: date
    kind: str  # "leave" | "holiday" | "rostered_off"
    #: None = the whole day.
    minutes_per_day: float | None = None
    name: str | None = None


@dataclass
class Person:
    key: Any
    name: str
    email: str | None
    #: Contracted minutes on a given day (0 = not a working day).
    pattern: Callable[[date], float]
    role: str | None = None
    team: str | None = None
    skills: list[str] = field(default_factory=list)
    #: Office names the planner itself holds for the person (a tag or field).
    planner_offices: list[str] = field(default_factory=list)
    time_off: list[TimeOff] = field(default_factory=list)
    bookings: list[Booking] = field(default_factory=list)
    employment_end: date | None = None
    #: The public-holiday calendar the person follows, by name.
    calendar: str | None = None
    #: The source's own capacity for the window, in minutes, when it reports
    #: one (Float's people report); used in place of the computed figure.
    reported_capacity: float | None = None


@dataclass
class Window:
    start: date
    end: date

    def days(self) -> Iterable[date]:
        day = self.start
        while day <= self.end:
            yield day
            day += timedelta(days=1)

    def weekdays(self) -> int:
        return sum(1 for d in self.days() if d.weekday() < 5)


def parse_window(start: str | None, end: str | None) -> Window | dict:
    """The window, or an error dict the tool returns as is."""
    if not start or not end:
        return _bad("Give start and end dates (YYYY-MM-DD).")
    try:
        window = Window(date.fromisoformat(start), date.fromisoformat(end))
    except ValueError:
        return _bad("start and end must be dates as YYYY-MM-DD.")
    if window.end < window.start:
        return _bad("end is before start.")
    if (window.end - window.start).days > MAX_WINDOW_DAYS:
        return _bad(f"The window is longer than {MAX_WINDOW_DAYS} days; ask for a shorter one.")
    return window


def _bad(message: str) -> dict:
    return {"error": True, "status": 400, "message": message}


def is_error(response: Any) -> bool:
    return isinstance(response, dict) and response.get("error") is True


def to_date(value: Any) -> date | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


# ── matching words people use to the names planners hold ──────────────────────


def normalize(text: str | None) -> str:
    """Lowercase, accents off, letters and digits only, single spaces."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", folded.lower()))


def _word_like(a: str, b: str) -> bool:
    """`strategy` ~ `strategist`, `direction` ~ `director`, `design` ~ `designer`."""
    if a == b or a.startswith(b) or b.startswith(a):
        return True
    common = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        common += 1
    return common >= max(4, round(0.7 * min(len(a), len(b))))


def term_matches(query: str | None, text: str | None) -> bool:
    """Every word of the query is like some word of the text."""
    q, t = normalize(query).split(), normalize(text).split()
    if not q or not t:
        return False
    return all(any(_word_like(w, x) for x in t) for w in q)


# ── the arithmetic ────────────────────────────────────────────────────────────


def _hours(minutes: float) -> float:
    return round(minutes / 60, 1)


def _spans(days: list[date]) -> list[str]:
    """[Nov 23, Nov 24, Nov 25] → ["2026-11-23..2026-11-25"]."""
    out: list[str] = []
    run: list[date] = []
    for day in sorted(days):
        if run and (day - run[-1]).days > 1:
            out.append(_span(run))
            run = []
        run.append(day)
    if run:
        out.append(_span(run))
    return out


def _span(run: list[date]) -> str:
    return run[0].isoformat() if len(run) == 1 else f"{run[0]}..{run[-1]}"


def working_days(person: Person, around: date) -> str:
    """The person's week in words, e.g. "Mon-Thu 8h" or "Mon-Fri 4h"."""
    monday = around - timedelta(days=around.weekday())
    week = [person.pattern(monday + timedelta(days=i)) for i in range(7)]
    on = [i for i, m in enumerate(week) if m > 0]
    if not on:
        return "no working days"
    hours = {round(week[i] / 60, 2) for i in on}
    if on == list(range(on[0], on[-1] + 1)):
        days = SHORT[on[0]] if len(on) == 1 else f"{SHORT[on[0]]}-{SHORT[on[-1]]}"
    else:
        days = ",".join(SHORT[i] for i in on)
    if len(hours) == 1:
        return f"{days} {_fmt_h(hours.pop())}"
    return ", ".join(f"{SHORT[i]} {_fmt_h(round(week[i] / 60, 2))}" for i in on)


def _fmt_h(hours: float) -> str:
    return f"{hours:g}h"


@dataclass
class Load:
    capacity: float = 0.0
    confirmed: float = 0.0
    tentative: float = 0.0
    contracted: float = 0.0  # before time off
    leave_days: list[date] = field(default_factory=list)
    holidays: list[tuple[date, str | None]] = field(default_factory=list)
    projects: dict[tuple[str | None, bool], float] = field(default_factory=dict)


def load_of(person: Person, window: Window) -> Load:
    out = Load()
    for day in window.days():
        contracted = person.pattern(day)
        if contracted <= 0:
            continue
        out.contracted += contracted
        cap = contracted
        for off in person.time_off:
            if not (off.start <= day <= off.end):
                continue
            if off.kind == "holiday":
                out.holidays.append((day, off.name))
            elif off.kind == "leave":
                out.leave_days.append(day)
            cap = 0 if off.minutes_per_day is None else max(cap - off.minutes_per_day, 0)
        out.capacity += cap
        if cap <= 0:
            continue
        for b in person.bookings:
            if b.start <= day <= b.end:
                if b.confirmed:
                    out.confirmed += b.minutes_per_day
                else:
                    out.tentative += b.minutes_per_day
                key = (b.project, b.confirmed)
                out.projects[key] = out.projects.get(key, 0) + b.minutes_per_day
    if person.reported_capacity is not None:
        out.capacity = person.reported_capacity
    return out


def person_row(person: Person, window: Window, office: str | None, office_from: str) -> dict:
    load = load_of(person, window)
    booked = load.confirmed + load.tentative
    flags: list[str] = []
    if load.capacity > 0 and booked > load.capacity:
        flags.append("over_capacity")
    if load.contracted > 0 and load.capacity == 0 and load.leave_days:
        flags.append("on_leave_whole_window")
    if load.contracted == 0:
        flags.append("not_working_in_window")
    if person.employment_end and window.start <= person.employment_end < window.end:
        flags.append("contract_ends_in_window")
    week = [person.pattern(window.start + timedelta(days=i)) for i in range(7)]
    on = [m for m in week if m > 0]
    if on and (len(on) < 5 or min(on) < PART_TIME_DAY_MINUTES):
        flags.append("part_time")
    # Flat rows, scalars only: the evidence index (agents/evidence_index.py)
    # keeps a row's scalar fields and nothing nested, and the answer is
    # grounded and cited against it.
    row: dict[str, Any] = {
        "person": person.name,
        "email": person.email,
        "office": office,
        "role": person.role,
        "works": working_days(person, window.start),
        "capacity_h": _hours(load.capacity),
        "confirmed_h": _hours(load.confirmed),
        "tentative_h": _hours(load.tentative),
        "free_h": _hours(max(load.capacity - booked, 0)),
        "free_if_tentative_drops_h": _hours(max(load.capacity - load.confirmed, 0)),
    }
    if load.capacity:
        row["booked_pct"] = round(100 * booked / load.capacity)
    if person.team:
        row["team"] = person.team
    if office_from != "ub_email":
        row["office_from"] = office_from
    if load.leave_days:
        row["leave"] = ", ".join(_spans(load.leave_days))
    if load.holidays:
        row["holidays"] = "; ".join(sorted({f"{d} {n or 'holiday'}" for d, n in load.holidays}))
    if person.employment_end and person.employment_end <= window.end:
        row["contract_ends"] = person.employment_end.isoformat()
    if load.projects:
        row["projects"] = "; ".join(
            f"{name or 'unnamed project'}{' (TENTATIVE)' if not confirmed else ''} "
            f"{_hours(minutes):g}h"
            for (name, confirmed), minutes in sorted(load.projects.items(), key=lambda kv: -kv[1])
        )
    if flags:
        row["flags"] = ", ".join(flags)
    row["_minutes"] = (load.capacity, load.confirmed, load.tentative)
    return row


def totals(rows: list[dict]) -> dict:
    """Summed from each row's exact minutes, not its rounded hours."""
    cap = sum(r["_minutes"][0] for r in rows)
    conf = sum(r["_minutes"][1] for r in rows)
    tent = sum(r["_minutes"][2] for r in rows)
    free = sum(max(c - a - b, 0) for c, a, b in (r["_minutes"] for r in rows))
    return {
        "people": len(rows),
        "capacity_h": _hours(cap),
        "confirmed_h": _hours(conf),
        "tentative_h": _hours(tent),
        "free_h": _hours(free),
        "confirmed_pct": round(100 * conf / cap) if cap else None,
        "booked_pct": round(100 * (conf + tent) / cap) if cap else None,
        "over_capacity": sum(1 for r in rows if "over_capacity" in r.get("flags", "")),
    }


# ── UB's side: offices and staff ──────────────────────────────────────────────


@dataclass
class Directory:
    offices: list[dict]
    #: UB staff (current: active or on leave), each with `officeName`.
    staff: list[dict]
    truncated: bool = False

    def office_named(self, wanted: str) -> dict | None:
        """The office a user's words name: its name, its city, an alias
        (LA, NYC) or its initials."""
        w = normalize(wanted)
        w = OFFICE_ALIASES.get(w.replace(" ", ""), w)
        for o in self.offices:
            names = {normalize(o.get("name")), normalize(o.get("syncCity"))} - {""}
            initials = {"".join(x[0] for x in n.split()) for n in names}
            if w in names or w.replace(" ", "") in initials:
                return o
        for o in self.offices:  # "new york office", "london studio"
            if any(n and n in w for n in (normalize(o.get("name")), normalize(o.get("syncCity")))):
                return o
        return None


async def bounded(calls: list[Callable[[], Awaitable[Any]]]) -> list[Any]:
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(call: Callable[[], Awaitable[Any]]) -> Any:
        async with sem:
            return await call()

    return await asyncio.gather(*(one(c) for c in calls))


async def _staff_where(flt: dict, tool_context: Any, depth: int = 0) -> tuple[list[dict], bool]:
    """Every current staff row matching `flt`. /org/query has no paging, so a
    full answer is split by department, then by the name's first letter."""
    body = {
        "entity": "staff",
        "filter": {**flt, "status": {"in": ["active", "on_leave"]}},
        "sort": [{"field": "fullName", "direction": "asc"}],
        "limit": UB_QUERY_LIMIT,
    }
    res = await gub_post("/org/query", body, tool_context)
    if is_error(res):
        raise _UbError(res)
    rows = res.get("results") or []
    if not res.get("truncated"):
        return rows, False
    if depth == 0 and "department" not in flt:
        groups = await gub_post(
            "/org/query",
            {
                "entity": "staff",
                "filter": body["filter"],
                "group_by": ["department"],
                "aggregate": {"n": {"op": "count"}},
                "limit": UB_QUERY_LIMIT,
            },
            tool_context,
        )
        if not is_error(groups) and groups.get("results"):
            parts = []
            for g in groups["results"]:
                dept = g.get("department")
                sub = {**flt, "department": {"eq": dept} if dept else {"is_null": True}}
                parts.append(lambda sub=sub: _staff_where(sub, tool_context, depth + 1))
            out, cut = [], False
            for got, t in await bounded(parts):
                out.extend(got)
                cut = cut or t
            return out, cut
    if depth <= 1 and "fullName" not in flt:
        parts = [
            lambda c=c: _staff_where(
                {**flt, "fullName": {"like": f"{c}%"}}, tool_context, depth + 2
            )
            for c in "abcdefghijklmnopqrstuvwxyz"
        ]
        out, cut = [], False
        for got, t in await bounded(parts):
            out.extend(got)
            cut = cut or t
        return out, cut
    return rows, True


class _UbError(Exception):
    def __init__(self, response: dict) -> None:
        super().__init__(response.get("message"))
        self.response = response


async def ub_directory(tool_context: Any, office: dict | None = None) -> Directory | dict:
    """UB's offices and current staff (one office's when `office` is given)."""
    offices = await gub_get("/org/offices", tool_context)
    if is_error(offices):
        return offices
    if isinstance(offices, dict):
        offices = offices.get("offices") or offices.get("results") or []
    directory = Directory(offices=list(offices), staff=[])
    if office is not None:
        targets = [office]
    else:
        targets = directory.offices
    try:
        filters = [{"officeId": {"eq": o.get("id")}} for o in targets]
        if office is None:
            filters.append({"officeId": {"is_null": True}})
        parts = [lambda f=f: _staff_where(f, tool_context) for f in filters]
        names = {o.get("id"): o.get("name") for o in directory.offices}
        for rows, cut in await bounded(parts):
            for r in rows:
                r.setdefault("officeName", names.get(r.get("officeId")))
            directory.staff.extend(rows)
            directory.truncated = directory.truncated or cut
    except _UbError as err:
        return err.response
    return directory


async def resolve_office(office: str | None, tool_context: Any) -> tuple[dict | None, dict | None]:
    """(office record, error dict). No office asked for = (None, None)."""
    if not office:
        return None, None
    offices = await gub_get("/org/offices", tool_context)
    if is_error(offices):
        return None, offices
    if isinstance(offices, dict):
        offices = offices.get("offices") or offices.get("results") or []
    found = Directory(offices=list(offices), staff=[]).office_named(office)
    if found is None:
        names = sorted(o.get("name") for o in offices if o.get("name"))
        return None, {
            "error": True,
            "status": 400,
            "message": f"UB has no office called {office!r}. Offices: {', '.join(names)}.",
        }
    return found, None


# ── putting the two together ──────────────────────────────────────────────────


@dataclass
class Scope:
    office: dict | None = None
    role: str | None = None
    skill: str | None = None
    team: str | None = None
    #: One person, by email or name.
    person: str | None = None
    #: Planner people keys that have the skill (the source resolves skills).
    skill_keys: set | None = None


def capacity_answer(
    *,
    source: str,
    people: list[Person],
    directory: Directory,
    window: Window,
    scope: Scope,
    calendars: dict[str, list[tuple[date, str]]],
    warnings: list[str],
) -> dict:
    """Join the planner's people to UB, apply the filters, compute, and shape
    the answer `runn_capacity` and `float_capacity` both return."""
    by_email: dict[str, dict] = {}
    by_name: dict[str, list[dict]] = {}
    for s in directory.staff:
        if s.get("email"):
            by_email[s["email"].strip().lower()] = s
        by_name.setdefault(normalize(s.get("fullName")), []).append(s)

    office_names = [o.get("name") for o in directory.offices if o.get("name")]
    matched_staff: set[str] = set()
    rows: list[dict] = []
    used_calendars: set[str] = set()
    for p in people:
        staff = by_email.get((p.email or "").strip().lower())
        office_from = "ub_email"
        if staff is None:
            same_name = [
                s for s in by_name.get(normalize(p.name), []) if s["id"] not in matched_staff
            ]
            if len(same_name) == 1:
                staff, office_from = same_name[0], "ub_name_match"
        if staff is not None:
            matched_staff.add(staff["id"])
            office = staff.get("officeName")
        else:
            office = _planner_office(p.planner_offices, directory)
            office_from = "planner" if office else "unknown"
        if scope.office is not None and office != scope.office.get("name"):
            continue
        if scope.role and not (
            term_matches(scope.role, p.role)
            or (
                p.role is None
                and staff is not None
                and term_matches(scope.role, staff.get("title"))
            )
        ):
            continue
        if scope.team and not term_matches(scope.team, p.team):
            continue
        if scope.skill_keys is not None and p.key not in scope.skill_keys:
            continue
        if scope.person and not _is_person(scope.person, p, staff):
            continue
        rows.append(person_row(p, window, office, office_from))
        if p.calendar:
            used_calendars.add(p.calendar)

    # UB staff in scope with no planner record: never free, never counted.
    planner_emails = {(p.email or "").strip().lower() for p in people}
    no_record = []
    for s in directory.staff:
        if s["id"] in matched_staff or (s.get("email") or "").lower() in planner_emails:
            continue
        if scope.person and not _is_person(scope.person, None, s):
            continue
        if scope.office is not None and s.get("officeName") != scope.office.get("name"):
            continue
        if scope.role and not term_matches(scope.role, s.get("title")):
            continue
        if scope.team and not term_matches(scope.team, s.get("department")):
            continue
        if scope.skill and not scope.role and not scope.team and scope.office is None:
            continue  # UB holds skills elsewhere; without another filter, too broad
        no_record.append(
            {
                "person": s.get("fullName"),
                "email": s.get("email"),
                "title": s.get("title"),
                "office": s.get("officeName"),
            }
        )

    rows.sort(key=lambda r: (-r["free_h"], r["person"] or ""))
    out: dict[str, Any] = {
        "source": source,
        "start": window.start.isoformat(),
        "end": window.end.isoformat(),
        "weekdays": window.weekdays(),
        "public_holidays": [
            {"date": d.isoformat(), "name": n, "calendar": c}
            for c in sorted(used_calendars)
            for d, n in sorted(calendars.get(c, []))
            if window.start <= d <= window.end
        ],
        "filters": {
            k: v
            for k, v in {
                "office": scope.office.get("name") if scope.office else None,
                "role": scope.role,
                "skill": scope.skill,
                "team": scope.team,
                "person": scope.person,
            }.items()
            if v
        },
        # A one-row list, so the totals are a citable evidence row.
        "totals": [totals(rows)],
        # The lists questions most often want whole, so none is cut short by
        # the row cap or lost in a long table.
        "lists": [
            {
                "on_leave": ", ".join(f"{r['person']} ({r['leave']})" for r in rows if "leave" in r)
                or "nobody",
                "over_capacity": ", ".join(
                    r["person"] for r in rows if "over_capacity" in r.get("flags", "")
                )
                or "nobody",
                "nothing_booked": ", ".join(
                    r["person"]
                    for r in rows
                    if r["capacity_h"] > 0 and r["confirmed_h"] == 0 and r["tentative_h"] == 0
                )
                or "nobody",
                "part_time": ", ".join(
                    r["person"] for r in rows if "part_time" in r.get("flags", "")
                )
                or "nobody",
            }
        ],
        "people": [{k: v for k, v in r.items() if k != "_minutes"} for r in rows[:MAX_ROWS]],
    }
    if len(rows) > MAX_ROWS:
        out["people_not_listed"] = len(rows) - MAX_ROWS
    if no_record:
        out["no_record"] = {
            "count": len(no_record),
            "note": f"In UB but not in {source}: their time is not planned there, so they are "
            "NOT free and not in the totals. Say so; never call them available.",
            "people": no_record[:MAX_NO_RECORD],
        }
    name_matched = [r["person"] for r in rows if r.get("office_from") == "ub_name_match"]
    if name_matched:
        warnings.append(
            f"Matched to UB by name, not email (the emails differ): {', '.join(name_matched)}."
        )
    if directory.truncated:
        warnings.append("UB's staff list was cut short; offices may be missing for some people.")
    if office_names and any(r.get("office_from") == "unknown" for r in rows):
        warnings.append("Some people have no office in UB or the planner.")
    if warnings:
        out["warnings"] = warnings
    return out


def _is_person(wanted: str, person: Person | None, staff: dict | None) -> bool:
    """Is this the one person asked for, by email (planner's or UB's) or name?"""
    w = wanted.strip().lower()
    emails = {(person.email or "").lower() if person else "", (staff or {}).get("email") or ""}
    if "@" in w:
        return w in {e.lower() for e in emails if e}
    names = {normalize(person.name) if person else "", normalize((staff or {}).get("fullName"))}
    return normalize(wanted) in names - {""}


def _planner_office(values: list[str], directory: Directory) -> str | None:
    for v in values:
        o = directory.office_named(v)
        if o is not None:
            return o.get("name")
    return None
