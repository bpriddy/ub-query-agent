"""
truth.py — ground truth for the Runn resourcing eval, computed from the
fixtures in Runn's own shapes (not from world.py's tables), the way a
resource manager reads Runn's planner:

  capacity  a working day's contracted minutes: the contract that covers the
            day, its `rosteredDays` for that weekday; zero on leave and on a
            public holiday of the person's holiday group.
  booked    the minutes of every assignment that covers a day with capacity
            (Runn does not count work on a day off). Split by the project's
            `isConfirmed`: confirmed vs tentative.
  free      capacity minus booked, never below zero.

Office is the UB office joined to the Runn person by email, falling back to
the person's Runn office tag; role is the current contract's role.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import cached_property

DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def d(text: str) -> date:
    return date.fromisoformat(text)


def workdays(start: date, end: date):
    day = start
    while day <= end:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


@dataclass
class Load:
    capacity: int = 0
    confirmed: int = 0
    tentative: int = 0

    @property
    def booked(self) -> int:
        return self.confirmed + self.tentative

    @property
    def free_confirmed(self) -> int:
        return max(self.capacity - self.confirmed, 0)

    @property
    def free(self) -> int:
        return max(self.capacity - self.booked, 0)

    def share(self, minutes: int) -> float:
        return minutes / self.capacity if self.capacity else 0.0

    def add(self, other: Load) -> Load:
        return Load(
            self.capacity + other.capacity,
            self.confirmed + other.confirmed,
            self.tentative + other.tentative,
        )


class Truth:
    def __init__(self, world: dict) -> None:
        self.runn = world["runn"]
        self.ub = world["ub"]
        self.today = d(world["today"])

    # ── indexes ──
    @cached_property
    def people(self) -> dict[int, dict]:
        return {p["id"]: p for p in self.runn["people"]}

    @cached_property
    def projects(self) -> dict[int, dict]:
        return {p["id"]: p for p in self.runn["projects"]}

    @cached_property
    def roles(self) -> dict[int, str]:
        return {r["id"]: r["name"] for r in self.runn["roles"]}

    @cached_property
    def skills(self) -> dict[int, str]:
        return {s["id"]: s["name"] for s in self.runn["skills"]}

    @cached_property
    def ub_by_email(self) -> dict[str, dict]:
        return {s["email"].lower(): s for s in self.ub["staff"]}

    @cached_property
    def office_names(self) -> dict[str, str]:
        return {o["id"]: o["name"] for o in self.ub["offices"]}

    def name(self, pid: int) -> str:
        p = self.people[pid]
        return f"{p['firstName']} {p['lastName']}"

    def office(self, pid: int) -> str:
        """LA / NYC / London: UB's office by email, else the Runn office tag."""
        staff = self.ub_by_email.get((self.people[pid]["email"] or "").lower())
        if staff:
            return {"Los Angeles": "LA", "New York": "NYC", "London": "London"}[
                self.office_names[staff["officeId"]]
            ]
        tags = {t["name"] for t in self.people[pid]["tags"]}
        return next(o for o in ("LA", "NYC", "London") if o in tags)

    def contract(self, pid: int, day: date) -> dict | None:
        for c in self.runn["contracts"]:
            if c["personId"] != pid:
                continue
            if d(c["startDate"]) <= day and (c["endDate"] is None or day <= d(c["endDate"])):
                return c
        return None

    def role(self, pid: int) -> str:
        c = self.contract(pid, self.today) or next(
            c for c in self.runn["contracts"] if c["personId"] == pid
        )
        return self.roles[c["roleId"]]

    def person_skills(self, pid: int) -> set[str]:
        return {self.skills[s["id"]] for s in self.people[pid]["skills"]}

    def is_part_time(self, pid: int) -> bool:
        c = self.contract(pid, self.today)
        if c is None:
            return False
        full = 450 if self.office(pid) == "London" else 480
        return sum(c["rosteredDays"].values()) < 5 * full

    def leave(self, pid: int) -> list[dict]:
        return [t for t in self.runn["time_offs_leave"] if t["personId"] == pid]

    @cached_property
    def _off_days(self) -> dict[int, set[date]]:
        off: dict[int, set[date]] = {}
        for t in self.runn["time_offs_leave"] + self.runn["time_offs_holidays"]:
            day = d(t["startDate"])
            while day <= d(t["endDate"]):
                off.setdefault(t["personId"], set()).add(day)
                day += timedelta(days=1)
        return off

    # ── the planner's arithmetic ──
    def capacity(self, pid: int, day: date) -> int:
        if day in self._off_days.get(pid, set()):
            return 0
        c = self.contract(pid, day)
        return int(c["rosteredDays"].get(DAYS[day.weekday()], 0)) if c else 0

    def load(self, pid: int, start: date, end: date, project: int | None = None) -> Load:
        total = Load()
        rows = [
            a
            for a in self.runn["assignments"]
            if a["personId"] == pid and (project is None or a["projectId"] == project)
        ]
        for day in workdays(start, end):
            cap = self.capacity(pid, day)
            day_load = Load(capacity=cap)
            if cap:
                for a in rows:
                    if d(a["startDate"]) <= day <= d(a["endDate"]):
                        if self.projects[a["projectId"]]["isConfirmed"]:
                            day_load.confirmed += a["minutesPerDay"]
                        else:
                            day_load.tentative += a["minutesPerDay"]
            total = total.add(day_load)
        return total

    def assignments(self, pid: int, start: date, end: date) -> list[dict]:
        return [
            a
            for a in self.runn["assignments"]
            if a["personId"] == pid and d(a["startDate"]) <= end and d(a["endDate"]) >= start
        ]

    # ── selections ──
    def select(
        self,
        *,
        office: str | None = None,
        roles: tuple[str, ...] | None = None,
        skill: str | None = None,
    ) -> list[int]:
        out = []
        for pid in self.people:
            if office and self.office(pid) != office:
                continue
            if roles and self.role(pid) not in roles:
                continue
            if skill and skill not in self.person_skills(pid):
                continue
            out.append(pid)
        return out

    def availability(
        self,
        pids: list[int],
        start: date,
        end: date,
        *,
        core_at: float = 0.5,
        forbid_below: float = 0.25,
    ) -> dict:
        """core: free (counting tentative work) >= core_at of capacity;
        acceptable: free only if tentative work gives way; forbidden: little
        or no confirmed free time, or no capacity at all (leave)."""
        core, ok, bad = [], [], []
        for pid in pids:
            load = self.load(pid, start, end)
            if load.capacity == 0:
                bad.append(pid)
            elif load.share(load.free) >= core_at:
                core.append(pid)
            elif load.share(load.free_confirmed) >= core_at:
                ok.append(pid)
            elif load.share(load.free_confirmed) < forbid_below:
                bad.append(pid)
        return {
            "core": [self.name(p) for p in core],
            "acceptable": [self.name(p) for p in ok],
            "forbidden": [self.name(p) for p in bad],
        }

    def over_capacity(self, pids: list[int], start: date, end: date) -> dict:
        over = [
            p for p in pids if self.load(p, start, end).booked > self.load(p, start, end).capacity
        ]
        return {
            "core": [self.name(p) for p in over],
            "acceptable": [],
            "forbidden": [self.name(p) for p in pids if p not in over],
        }

    def utilization(self, pids: list[int], start: date, end: date) -> dict:
        total = Load()
        for p in pids:
            total = total.add(self.load(p, start, end))
        return {
            "capacity_hours": round(total.capacity / 60, 1),
            "confirmed_hours": round(total.confirmed / 60, 1),
            "tentative_hours": round(total.tentative / 60, 1),
            "confirmed_pct": round(100 * total.share(total.confirmed)),
            "total_pct": round(100 * total.share(total.booked)),
        }

    def on_leave(self, start: date, end: date) -> list[str]:
        return [
            self.name(t["personId"])
            for t in self.runn["time_offs_leave"]
            if d(t["startDate"]) <= end and d(t["endDate"]) >= start
        ]

    def understaffed(self, start: date, end: date) -> list[str]:
        placeholder_ids = {p["id"] for p in self.runn["placeholders"]}
        names = []
        for a in self.runn["assignments"]:
            if a["personId"] in placeholder_ids and d(a["startDate"]) <= end:
                if d(a["endDate"]) >= start:
                    names.append(self.projects[a["projectId"]]["name"])
        return sorted(set(names))

    def project_people(self, project: int, start: date, end: date) -> list[str]:
        placeholder_ids = {p["id"] for p in self.runn["placeholders"]}
        return sorted(
            {
                self.name(a["personId"])
                for a in self.runn["assignments"]
                if a["projectId"] == project
                and a["personId"] not in placeholder_ids
                and d(a["startDate"]) <= end
                and d(a["endDate"]) >= start
            }
        )

    def describe(self, pid: int, start: date, end: date) -> str:
        """One person's plan over a window, in words, for the judge."""
        parts = []
        for a in self.assignments(pid, start, end):
            proj = self.projects[a["projectId"]]
            state = "confirmed" if proj["isConfirmed"] else "TENTATIVE"
            parts.append(
                f"{proj['name']} ({state}) {a['minutesPerDay'] / 60:g} h/day "
                f"{a['startDate']}..{a['endDate']}"
            )
        load = self.load(pid, start, end)
        return (
            f"{self.name(pid)} [{self.office(pid)}, {self.role(pid)}]: "
            + ("; ".join(parts) or "nothing booked")
            + f". Window capacity {load.capacity / 60:g} h, confirmed {load.confirmed / 60:g} h, "
            f"tentative {load.tentative / 60:g} h."
        )
