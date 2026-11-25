"""
resourcing.py — staffing and capacity for `/ask`, from resourcing (gub-agent#117).

Tools:
  resourcing_allocations  — who is on what, over which dates, for how many
                            hours a day, from the planners' assignments
  resourcing_capacity     — each person's capacity, allocated and available
                            hours over a window, by office, role, team or person

Resourcing (UB D45) blends each planner's "leg" (Runn today) into curated
views and computes capacity. It is read through UB's relay
(`/relay/resourcing/*`, gcp-universal-backend#301), as the user, so the
"Resourcing reader" grant (`resourcing_reader`) is checked in UB and nowhere
else. The relay answers `{data, meta: {sources: [{source, asOf,
contractVersion}], limit, offset, nextOffset}}`, and takes filters as
`col=v` / `col.<op>=v` (ne, in, gt, gte, lt, lte, contains, is_null), AND
only, each parameter at most once.

These tools do the joins in code (assignments + projects + people; UB's
staff directory for office, which no planner leg holds) so the model spends
one round per staffing question. They never compute capacity: resourcing
does, and its figures are quoted as they come.

Offered only on the `/ask` engine (`config.ASK_CONTRACT`).

The relay's error codes come back as they are (`RESOURCING_NO_GRANT`,
`RESOURCING_CAPACITY_OFF`, …), so the `/ask` turn reports the source as
failed rather than failing the answer.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from ._client import gub_get
from .capacity import (
    MAX_ROWS,
    WEEKDAYS,
    Person,
    bounded,
    is_error,
    normalize,
    resolve_office,
    to_date,
    ub_directory,
    working_days,
)

RELAY = "/relay/resourcing"
#: The relay's largest page.
PAGE = 1000
#: Pages one read may follow (nextOffset) before it stops and says so.
MAX_PAGES = 5
MAX_ALLOCATIONS = 150
MAX_PROJECT_MATCHES = 5
MAX_NO_RECORD = 25
#: Resourcing computes capacity over at most this many days.
MAX_CAPACITY_DAYS = 100
#: The relay refuses a parameter value longer than 1000 characters, so a long
#: `.in` list is split over several calls.
MAX_IN_CHARS = 900
#: Leave, as opposed to holidays and rostered days off.
LEAVE_TYPES = "time_off,sabbatical,mat_leave"

KeyT = tuple[str, str]  # (source, id): ids are only unique within a leg


# ── reading the relay ─────────────────────────────────────────────────────────


class _Read:
    """Rows of one or more relay reads, the legs that answered, and whether a
    read stopped before its last page."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.sources: dict[str, dict] = {}
        self.cut = False

    def add_meta(self, meta: Any) -> None:
        for s in (meta or {}).get("sources") or []:
            if isinstance(s, dict) and s.get("source"):
                self.sources[str(s["source"])] = s

    def merge(self, other: _Read) -> None:
        self.rows.extend(other.rows)
        self.sources.update(other.sources)
        self.cut = self.cut or other.cut


async def _get(path: str, tool_context: Any, max_pages: int = MAX_PAGES, **params: Any):
    """Every row of a relay read, following `nextOffset` up to `max_pages`
    pages; an error dict on failure."""
    out = _Read()
    offset = 0
    for _ in range(max_pages):
        page = await gub_get(path, tool_context, limit=PAGE, offset=offset or None, **params)
        if is_error(page):
            return page
        out.add_meta(page.get("meta"))
        data = page.get("data") or []
        out.rows.extend(r for r in data if isinstance(r, dict))
        nxt = (page.get("meta") or {}).get("nextOffset")
        if nxt is None:
            return out
        offset = int(nxt)
    out.cut = True
    return out


def _chunks(values: list[str]) -> list[str]:
    """Comma-joined `.in` values, each under the relay's length limit."""
    out: list[str] = []
    cur: list[str] = []
    size = 0
    for v in values:
        if cur and size + len(v) + 1 > MAX_IN_CHARS:
            out.append(",".join(cur))
            cur, size = [], 0
        cur.append(v)
        size += len(v) + 1
    if cur:
        out.append(",".join(cur))
    return out


async def _get_in(
    path: str, column: str, values: list[str], tool_context: Any, **params: Any
) -> _Read | dict:
    """Rows whose `column` is one of `values`: one read per chunk, merged."""
    wanted = sorted({v for v in values if v and "," not in v})
    out = _Read()
    if not wanted:
        return out
    got = await bounded(
        [
            lambda c=c: _get(path, tool_context, **{f"{column}.in": c}, **params)
            for c in _chunks(wanted)
        ]
    )
    for g in got:
        if is_error(g):
            return g
        out.merge(g)
    return out


def _key(row: dict, column: str = "id") -> KeyT:
    return (str(row.get("source") or ""), str(row.get(column) or ""))


def _sources(*reads: _Read) -> list[dict]:
    """Each leg that answered and how fresh it is, for the answer to cite."""
    merged: dict[str, dict] = {}
    for r in reads:
        merged.update(r.sources)
    return [
        {"source": s, "as_of": m.get("asOf"), "contract_version": m.get("contractVersion")}
        for s, m in sorted(merged.items())
    ]


def _with_hint(err: dict) -> dict:
    """The relay's error as is, with what it means for the answer."""
    hints = {
        "RESOURCING_NO_GRANT": 'The user lacks UB\'s "Resourcing reader" grant.',
        "RESOURCING_CAPACITY_OFF": "Resourcing does not serve capacity yet in this "
        "environment. Say capacity could not be read; resourcing_allocations still works.",
        "RESOURCING_NOT_CONFIGURED": "Resourcing is not configured in this environment.",
        "RESOURCING_BAD_QUERY": "Resourcing refused the query's filters.",
        "RESOURCING_UPSTREAM_ERROR": "Resourcing could not be read.",
    }
    hint = hints.get(str(err.get("code")))
    return {**err, "hint": hint} if hint else err


def _stem(term: str) -> str:
    """A text filter that matches the forms of a role or team word:
    "strategy" → "strateg" (Strategist), "art director" → "art direct"
    (Art Direction). Resourcing's `.contains` is case-insensitive."""
    words = term.strip().split()
    if not words:
        return term.strip()
    last = words[-1]
    for suffix in ("ists", "ers", "ors", "ion", "ist", "ing", "er", "or", "y", "s"):
        if last.lower().endswith(suffix) and len(last) - len(suffix) >= 4:
            last = last[: -len(suffix)]
            break
    return " ".join([*words[:-1], last])


def _spelled(email: str) -> str:
    """jane.park@anomaly.com → "jane park"."""
    return normalize(email.split("@", 1)[0].replace(".", " ").replace("_", " "))


async def _find_people(person: str, tool_context: Any, **params: Any) -> tuple[_Read | dict, str]:
    """People by email, else by the name the email spells (one match only),
    else, for a name, by name. Returns the read and how it matched."""
    wanted = person.strip()
    if "@" in wanted:
        got = await _get(f"{RELAY}/people", tool_context, email=wanted.lower(), **params)
        if is_error(got) or got.rows:
            return got, "email"
        name = _spelled(wanted)
        if not name:
            return got, ""
        by_name = await _get(f"{RELAY}/people", tool_context, **{"name.contains": name}, **params)
        if is_error(by_name):
            return by_name, ""
        by_name.rows = [p for p in by_name.rows if normalize(p.get("name")) == name]
        if len({_key(p) for p in by_name.rows}) == 1:
            return by_name, "name"
        by_name.rows = []
        return by_name, ""
    got = await _get(f"{RELAY}/people", tool_context, **{"name.contains": wanted}, **params)
    return got, "name"


# ── resourcing_allocations ────────────────────────────────────────────────────


def _pattern(rows: list[dict]):
    """A person's contracted minutes on a day, from their working patterns."""
    dated = sorted(rows, key=lambda r: r.get("start_date") or "")

    def minutes(day: date) -> float:
        for r in reversed(dated):
            s, e = to_date(r.get("start_date")), to_date(r.get("end_date"))
            if (s is None or s <= day) and (e is None or day <= e):
                by_day = r.get("minutes_by_weekday")
                if isinstance(by_day, dict):
                    return float(by_day.get(WEEKDAYS[day.weekday()]) or 0)
                return float(r.get("minutes_per_day") or 0) if day.weekday() < 5 else 0
        return 0

    return minutes


async def resourcing_allocations(
    person_email: str | None = None,
    project_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    tool_context: Any = None,
) -> dict:
    """
    Look up staffing allocations in resourcing (the resource planners, Runn
    today): who is assigned to which project, over which dates, for how many
    hours a day, and whether that work is confirmed or tentative.

    Use this for who is staffed on a project, what a person is on, or every
    booking in a date range. Give a person's email (from UB's staff records),
    a project's name, a date range, or any mix. For who is FREE, over
    capacity, or how booked a team or office is, use `resourcing_capacity`.

    Examples:
    - "who is allocated to the Nike holiday project?" → project_name="Nike holiday"
    - "what is ada@anomaly.com on next month?" → person_email="ada@anomaly.com",
      start_date="2026-12-01", end_date="2026-12-31"
    - "which projects have open placeholder roles in December?" →
      start_date="2026-12-01", end_date="2026-12-31"

    Args:
        person_email: The person's work email, exactly as in UB's staff record
        project_name: All or part of the project's name (case-insensitive)
        start_date: Only allocations ending on or after this date (YYYY-MM-DD);
            open-ended allocations always count
        end_date: Only allocations starting on or before this date (YYYY-MM-DD)

    Returns:
        dict with 'allocations' (person, email, project, client, start_date,
        end_date (null = open-ended), hours_per_day, billable, status
        "confirmed"/"tentative", placeholder, role, works = the person's
        working days, source = the planner); for a person, their 'leave';
        for a project, a 'summary' (people on it and their summed hours a day
        on start_date, else today); and 'sources', each planner with its
        'as_of' (when resourcing last read it): cite it. A placeholder row is
        an unfilled role, not a person. When a project name matches several
        projects, 'projects' lists them and nothing else: ask the user which
        one. A person with no record comes back with a note saying so: that
        is not the same as free. An 'error' dict when resourcing can't be
        read, with its 'code' (RESOURCING_NO_GRANT means the user lacks the
        Resourcing reader grant).
    """
    if not person_email and not project_name and not start_date and not end_date:
        return {
            "error": True,
            "status": 400,
            "message": "Give a person_email, a project_name, a date range, or a mix.",
        }
    for value in (start_date, end_date):
        if value is not None and to_date(value) is None:
            return {"error": True, "status": 400, "message": "Dates must be YYYY-MM-DD."}

    notes: list[str] = []
    reads: list[_Read] = []
    filters: dict[str, Any] = {}
    people_by_key: dict[KeyT, dict] = {}
    person_keys: set[KeyT] | None = None
    project_keys: set[KeyT] | None = None
    projects_by_key: dict[KeyT, dict] = {}

    if person_email:
        found, how = await _find_people(person_email, tool_context)
        if is_error(found):
            return _with_hint(found)
        reads.append(found)
        if not found.rows:
            return {
                "allocations": [],
                "note": f"Resourcing has no person with the email {person_email}. Their time is "
                "not planned there: do not call them free or unbooked; say they have no record.",
                "sources": _sources(found),
            }
        if how == "name":
            p = found.rows[0]
            notes.append(
                f"Matched {person_email} to {p.get('name')} by name: the planner holds "
                f"{'no email' if not p.get('email') else 'a different email (' + p['email'] + ')'}."
            )
        person_keys = {_key(p) for p in found.rows}
        people_by_key.update({_key(p): p for p in found.rows})
        filters["person_id.in"] = ",".join(sorted({k[1] for k in person_keys}))

    if project_name:
        found = await _get(
            f"{RELAY}/projects", tool_context, **{"name.contains": project_name.strip()}
        )
        if is_error(found):
            return _with_hint(found)
        reads.append(found)
        wanted = project_name.strip().lower()
        matches = found.rows
        exact = [p for p in matches if (p.get("name") or "").strip().lower() == wanted]
        if exact:
            matches = exact
        names = sorted({(p.get("name") or "").strip() for p in matches})
        if not matches:
            return {
                "allocations": [],
                "note": f"Resourcing has no project named like {project_name!r}.",
                "sources": _sources(found),
            }
        if len(names) > 1:
            return {
                "projects": [
                    {
                        "name": p.get("name"),
                        "client": p.get("client"),
                        "archived": p.get("is_archived"),
                        "source": p.get("source"),
                    }
                    for p in matches[:MAX_PROJECT_MATCHES]
                ],  # fmt: skip
                "note": "Several projects match; ask the user which one.",
                "sources": _sources(found),
            }
        # One name, possibly in more than one planner.
        project_keys = {_key(p) for p in matches}
        projects_by_key.update({_key(p): p for p in matches})
        filters["project_id.in"] = ",".join(sorted({k[1] for k in project_keys}))

    if end_date:
        filters["start_date.lte"] = end_date
    path = f"{RELAY}/assignments"
    if start_date:
        # Filters are AND only: the allocations overlapping the range are the
        # ones ending after its start plus the open-ended ones, read apart.
        got = await bounded(
            [
                lambda: _get(path, tool_context, **filters, **{"end_date.gte": start_date}),
                lambda: _get(path, tool_context, **filters, **{"end_date.is_null": "true"}),
            ]
        )
        for g in got:
            if is_error(g):
                return _with_hint(g)
        assignments = got[0]
        assignments.merge(got[1])
    else:
        assignments = await _get(path, tool_context, **filters)
        if is_error(assignments):
            return _with_hint(assignments)
    reads.append(assignments)

    # The id filters cross legs (ids are per leg): keep the legs' own rows.
    seen: set[KeyT] = set()
    rows = []
    for a in assignments.rows:
        k = _key(a)
        if k in seen:
            continue
        seen.add(k)
        src = str(a.get("source") or "")
        if person_keys is not None and (src, str(a.get("person_id"))) not in person_keys:
            continue
        if project_keys is not None and (src, str(a.get("project_id"))) not in project_keys:
            continue
        rows.append(a)
    rows.sort(key=lambda a: (a.get("start_date") or "", str(a.get("project_id")), _key(a)))
    shown = rows[:MAX_ALLOCATIONS]

    # Names for the rows shown, and working days for their people.
    need_people = sorted(
        {
            str(a.get("person_id"))
            for a in shown
            if not a.get("is_placeholder") and _key(a, "person_id") not in people_by_key
        }
    )
    need_projects = sorted(
        {str(a.get("project_id")) for a in shown if _key(a, "project_id") not in projects_by_key}
    )
    person_ids = sorted({str(a.get("person_id")) for a in shown if not a.get("is_placeholder")})
    calls = [
        lambda: _get_in(f"{RELAY}/people", "id", need_people, tool_context),
        lambda: _get_in(f"{RELAY}/projects", "id", need_projects, tool_context),
        lambda: _get_in(f"{RELAY}/working_patterns", "person_id", person_ids, tool_context),
    ]
    if person_keys is not None:
        # A person's leave answers "when is X back?" and "is X off?".
        leave_from = start_date or date.today().isoformat()
        leave_params = {"end_date.gte": leave_from, "type.in": LEAVE_TYPES}
        if end_date:
            leave_params["start_date.lte"] = end_date
        ids = sorted({k[1] for k in person_keys})
        calls.append(
            lambda: _get_in(f"{RELAY}/time_off", "person_id", ids, tool_context, **leave_params)
        )
    extra = await bounded(calls)
    for got in extra[:2]:
        if is_error(got):
            return _with_hint(got)
        reads.append(got)
    people_by_key.update({_key(p): p for p in extra[0].rows})
    projects_by_key.update({_key(p): p for p in extra[1].rows})
    patterns: dict[KeyT, list[dict]] = {}
    if is_error(extra[2]):
        notes.append("Working days could not be read from resourcing.")
    else:
        for w in extra[2].rows:
            patterns.setdefault(_key(w, "person_id"), []).append(w)
    around = to_date(start_date) or date.today()

    allocations = []
    for a in shown:
        pk = _key(a, "person_id")
        who = people_by_key.get(pk) or {}
        what = projects_by_key.get(_key(a, "project_id")) or {}
        minutes = a.get("minutes_per_day")
        placeholder = bool(a.get("is_placeholder"))
        row: dict[str, Any] = {
            "person": who.get("name") or ("placeholder" if placeholder else None),
            "email": who.get("email"),
            "project": what.get("name"),
            "client": what.get("client"),
            "start_date": a.get("start_date"),
            "end_date": a.get("end_date"),
            "hours_per_day": round(minutes / 60, 2) if isinstance(minutes, (int, float)) else None,
            "billable": a.get("is_billable"),
            "status": "tentative" if what.get("is_tentative") else "confirmed",
            "role": a.get("role"),
            "source": a.get("source"),
        }
        if a.get("end_date") is None:
            row["open_ended"] = True
        if placeholder:
            row["placeholder"] = True
        elif pk in patterns:
            row["works"] = working_days(
                Person(key=None, name="", email=None, pattern=_pattern(patterns[pk])), around
            )
        allocations.append(row)

    out: dict[str, Any] = {"count": len(rows), "allocations": allocations}
    if person_keys is not None:
        leave = extra[3]
        if is_error(leave):
            notes.append("Leave could not be read from resourcing.")
        else:
            name = next(iter(people_by_key[k].get("name") for k in person_keys), None)
            out["leave"] = [
                {"person": name, "start_date": t.get("start_date"), "end_date": t.get("end_date"),
                 "type": t.get("type"), "whole_days": t.get("minutes_per_day") is None}
                for t in sorted(leave.rows, key=lambda t: t.get("start_date") or "")
                if (str(t.get("source")), str(t.get("person_id"))) in person_keys
            ] or [{"person": name, "leave": "none booked"}]  # fmt: skip
    if project_keys is not None:
        # "how many people are on X, and how many hours a day?" on the day
        # asked about (start_date, else today): summed here.
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
    if len(rows) > MAX_ALLOCATIONS or assignments.cut:
        notes.append("More allocations exist than are listed; narrow the dates or the filter.")
    out["sources"] = _sources(*reads)
    if notes:
        out["notes"] = notes
    return out


# ── resourcing_capacity ───────────────────────────────────────────────────────


def _hours(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


async def resourcing_capacity(
    start: str,
    end: str,
    office: str | None = None,
    role: str | None = None,
    team: str | None = None,
    person: str | None = None,
    tool_context: Any = None,
) -> dict:
    """
    How much time people have over a date range, from resourcing (the
    resource planners, Runn today), which computes it: each person's capacity
    (their working pattern, less time off and holidays), allocated hours
    (assignments on days they work; confirmed and tentative together),
    available hours and utilization, with totals.

    Use this for: who is free or available, who has capacity, who is over
    capacity or underbooked, how booked or utilized a team or office is, who
    is off in a period, and who could cover for someone. One call covers
    everyone; filter by office, role, team or person as the question says.
    There is no skill filter: planners hold almost no skills. Use the role,
    or `resourcing_allocations` for what people are on.

    Examples:
    - "who's free for art direction in LA next week?" → start="2026-11-23",
      end="2026-11-27", office="LA", role="art director"
    - "how booked is the strategy team in Q1?" → start="2027-01-01",
      end="2027-03-31", role="strategist"

    Args:
        start: First day of the window (YYYY-MM-DD)
        end: Last day of the window (YYYY-MM-DD), at most 100 days on
        office: A UB office as the user said it ("LA", "New York", "London").
            Planners hold no office: it comes from UB's staff directory,
            joined on email, then on name
        role: A job role as the user said it ("art director", "strategy"),
            matched against the person's current role
        team: A planner team ("Design")
        person: One person, by work email or name, for a question about them
            alone ("is Isaac overbooked before Christmas?")

    Returns:
        dict with 'start', 'end', 'days', 'filters', 'totals' (one row:
        people, capacity_h, allocated_h, available_h, time_off_h,
        utilization_pct, over_capacity count), 'lists' (over_capacity,
        nothing_booked, no_capacity: every name, whole), 'people', one row
        each, most available first (capacity_h, allocated_h, available_h,
        utilization_pct, time_off_h, team, flags), and 'sources', each
        planner with its 'as_of' (when resourcing last read it): cite it.
        Tentative work is counted as allocated, not split out. With an
        office, 'no_record' lists UB staff there with no planner record (they
        are NOT free) and 'warnings' says who could not be placed in an
        office. An 'error' dict when resourcing or UB can't be read, with its
        'code' (RESOURCING_CAPACITY_OFF: resourcing does not serve capacity
        yet).
    """
    s, e = to_date(start), to_date(end)
    if not s or not e or len(start) != 10 or len(end) != 10:
        return {"error": True, "status": 400, "message": "Give start and end as YYYY-MM-DD."}
    if e < s:
        return {"error": True, "status": 400, "message": "end is before start."}
    days = (e - s).days + 1
    if days > MAX_CAPACITY_DAYS:
        return {
            "error": True,
            "status": 400,
            "message": f"The window is longer than {MAX_CAPACITY_DAYS} days; ask for a shorter "
            "one, or split it.",
        }

    office_rec, err = await resolve_office(office, tool_context)
    if err is not None:
        return err

    # People filters, the same for the people view and for capacity.
    people_filter: dict[str, str] = {}
    shown_filters: dict[str, str] = {}
    if role:
        people_filter["role.contains"] = _stem(role)
        shown_filters["role"] = f"role contains {_stem(role)!r}"
    if team:
        people_filter["team.contains"] = _stem(team)
        shown_filters["team"] = f"team contains {_stem(team)!r}"
    if office_rec is not None:
        shown_filters["office"] = office_rec.get("name")

    warnings: list[str] = []
    reads: list[_Read] = []
    path = f"{RELAY}/capacity"
    window = {"from": s.isoformat(), "to": e.isoformat()}
    match_how: dict[KeyT, str] = {}
    no_record: list[dict] = []

    if person:
        found, how = await _find_people(person, tool_context, **people_filter)
        if is_error(found):
            return _with_hint(found)
        reads.append(found)
        if not found.rows:
            return {
                "people": [],
                "note": f"Resourcing has no person matching {person!r}. Their time is not "
                "planned there: do not call them free; say they have no record.",
                "sources": _sources(found),
            }
        if how == "name" and "@" in person:
            warnings.append(f"Matched {person} by the name it spells; the planner's email differs.")
        shown_filters["person"] = person
        wanted_keys: set[KeyT] | None = {_key(p) for p in found.rows}
    else:
        wanted_keys = None

    if office_rec is not None:
        directory = await ub_directory(tool_context, office_rec)
        if is_error(directory):
            return directory
        if directory.truncated:
            warnings.append("UB's staff list was cut short; some of the office may be missing.")
        people = await _get(f"{RELAY}/people", tool_context, is_archived="false", **people_filter)
        if is_error(people):
            return _with_hint(people)
        reads.append(people)
        staff_by_email = {
            (st.get("email") or "").strip().lower(): st for st in directory.staff if st.get("email")
        }
        staff_by_name: dict[str, list[dict]] = {}
        for st in directory.staff:
            staff_by_name.setdefault(normalize(st.get("fullName")), []).append(st)
        matched_staff: set[Any] = set()
        no_email_unplaced = 0
        name_matched: list[str] = []
        for p in people.rows:
            k = _key(p)
            if wanted_keys is not None and k not in wanted_keys:
                continue
            st = staff_by_email.get((p.get("email") or "").strip().lower())
            if st is not None:
                match_how[k] = "ub_email"
                matched_staff.add(st.get("id"))
                continue
            same = staff_by_name.get(normalize(p.get("name")), [])
            if len(same) == 1:
                match_how[k] = "ub_name_match"
                matched_staff.add(same[0].get("id"))
                name_matched.append(p.get("name") or "")
            elif not p.get("email"):
                no_email_unplaced += 1
        if name_matched:
            warnings.append(
                "Matched to UB by name, not email (no email in the planner, or a different "
                f"one): {', '.join(sorted(name_matched))}."
            )
        if no_email_unplaced:
            warnings.append(
                f"{no_email_unplaced} people in the planner have no email and no name match in "
                f"UB's {office_rec.get('name')} staff, so they could not be matched to an office "
                "and are left out; some may work there."
            )
        if people.cut:
            warnings.append("The planner's people list was cut short; the office may be short.")
        if not role and not team and not person:
            for st in directory.staff:
                if st.get("id") not in matched_staff:
                    no_record.append(
                        {"person": st.get("fullName"), "email": st.get("email"),
                         "title": st.get("title"), "office": st.get("officeName")}
                    )  # fmt: skip
        wanted_keys = set(match_how)
        if not wanted_keys:
            capacity = _Read()
        else:
            capacity = await _get_in(
                path,
                "id",
                sorted({k[1] for k in wanted_keys}),
                tool_context,
                **window,
                **people_filter,
            )
    elif wanted_keys is not None:
        capacity = await _get_in(
            path, "id", sorted({k[1] for k in wanted_keys}), tool_context, **window,
            **people_filter,
        )  # fmt: skip
    else:
        capacity = await _get(path, tool_context, max_pages=1, **window, **people_filter)
    if is_error(capacity):
        return _with_hint(capacity)
    reads.append(capacity)

    rows: list[dict] = []
    seen: set[KeyT] = set()
    for c in capacity.rows:
        k = (str(c.get("source") or ""), str(c.get("personId") or ""))
        if k in seen or (wanted_keys is not None and k not in wanted_keys):
            continue
        seen.add(k)
        cap, alloc = _hours(c.get("capacityHours")), _hours(c.get("allocatedHours"))
        row: dict[str, Any] = {
            "person": c.get("name"),
            "email": c.get("email"),
            "capacity_h": cap,
            "allocated_h": alloc,
            "available_h": _hours(c.get("availableHours")),
            "utilization_pct": c.get("utilizationPercent"),
            "time_off_h": _hours(c.get("timeOffHours")),
            "source": c.get("source"),
        }
        if c.get("team"):
            row["team"] = c.get("team")
        if office_rec is not None:
            row["office"] = office_rec.get("name")
            if match_how.get(k) == "ub_name_match":
                row["office_from"] = "ub_name_match"
        flags = []
        if cap > 0 and alloc > cap:
            flags.append("over_capacity")
        if cap == 0:
            flags.append("no_capacity_in_window")
        if flags:
            row["flags"] = ", ".join(flags)
        rows.append(row)

    rows.sort(key=lambda r: (-r["available_h"], r["person"] or ""))
    cap_sum = sum(r["capacity_h"] for r in rows)
    alloc_sum = sum(r["allocated_h"] for r in rows)
    out: dict[str, Any] = {
        "start": s.isoformat(),
        "end": e.isoformat(),
        "days": days,
        "filters": shown_filters,
        # A one-row list, so the totals are a citable evidence row.
        "totals": [
            {
                "people": len(rows),
                "capacity_h": round(cap_sum, 1),
                "allocated_h": round(alloc_sum, 1),
                "available_h": round(sum(r["available_h"] for r in rows), 1),
                "time_off_h": round(sum(r["time_off_h"] for r in rows), 1),
                "utilization_pct": round(100 * alloc_sum / cap_sum) if cap_sum else None,
                "over_capacity": sum(1 for r in rows if "over_capacity" in r.get("flags", "")),
            }
        ],
        # The lists questions most often want whole, never cut by the row cap.
        "lists": [
            {
                "over_capacity": ", ".join(
                    r["person"] for r in rows if "over_capacity" in r.get("flags", "")
                )
                or "nobody",
                "nothing_booked": ", ".join(
                    r["person"] for r in rows if r["capacity_h"] > 0 and r["allocated_h"] == 0
                )
                or "nobody",
                "no_capacity": ", ".join(
                    r["person"] for r in rows if "no_capacity_in_window" in r.get("flags", "")
                )
                or "nobody",
            }
        ],
        "people": rows[:MAX_ROWS],
        "sources": _sources(*reads),
    }
    if len(rows) > MAX_ROWS:
        out["people_not_listed"] = len(rows) - MAX_ROWS
    if no_record:
        out["no_record"] = {
            "count": len(no_record),
            "note": "In UB but with no planner record: their time is not planned, so they are "
            "NOT free and not in the totals. Say so; never call them available.",
            "people": no_record[:MAX_NO_RECORD],
        }
    if capacity.cut:
        warnings.append("Resourcing returned more people than one call reads; totals may be short.")
    if warnings:
        out["warnings"] = warnings
    return out


RESOURCING_TOOLS = [resourcing_allocations, resourcing_capacity]
