"""
float_world.py — the same synthetic agency, as Float (float.com) holds it.

Built from the Runn fixtures, so every ground truth in truth.py holds for
Float too: the same people, contracts, bookings, leave and public holidays,
in Float API v3's shapes (developer.float.com) and with Float's gaps:

  * people: `work_days_hours` (hours Sun..Sat) only for part-timers
    (`employee_type` 0); a full-timer works the company's default week,
    which the API does not expose (the people report gives the capacity);
  * no skills: they are people tags, beside an office tag;
  * a booking is a task with `hours` per day and a `status` (1 tentative,
    2 confirmed); a project is tentative with `status` 1;
  * public holidays belong to a region (`region_id` on the person);
  * unfilled roles are placeholder people (`people_type_id` 3).

`python -m tests.evals.runn_resourcing.world` writes fixtures/float.json with
the rest.
"""

from __future__ import annotations

from .truth import DAYS, Truth, d, workdays

REGIONS = {"US": 1, "GB": 2}
TIMEOFF_TYPES = [
    {"timeoff_type_id": 1, "timeoff_type_name": "Vacation", "active": 1},
    {"timeoff_type_id": 2, "timeoff_type_name": "Parental leave", "active": 1},
]
STAMP = "2026-10-01 09:00:00"


def build(runn: dict) -> dict:
    roles = {r["id"]: r["name"] for r in runn["roles"]}
    teams = {t["id"]: t["name"] for t in runn["teams"]}
    skills = {s["id"]: s["name"] for s in runn["skills"]}
    groups = {g["id"]: g for g in runn["holiday_groups"]}
    region_of_group = {gid: REGIONS[g["countryCode"]] for gid, g in groups.items()}
    contracts: dict[int, list[dict]] = {}
    for c in runn["contracts"]:
        contracts.setdefault(c["personId"], []).append(c)
    person_skills: dict[int, list[str]] = {}
    for ps in runn["people_skills"]:
        person_skills.setdefault(ps["personId"], []).append(skills[ps["skillId"]])

    people = []
    for p in runn["people"]:
        cs = sorted(contracts[p["id"]], key=lambda c: c["startDate"])
        current = cs[-1]
        week = current["rosteredDays"]
        full_day = max(week.values())
        part_time = any(week[day] < full_day for day in DAYS[:5]) or full_day < 420
        office_tag = next(
            (t["name"] for t in p["tags"] if t["name"] in ("LA", "NYC", "London")), None
        )
        people.append(
            {
                "people_id": p["id"],
                "name": f"{p['firstName']} {p['lastName']}",
                "email": p["email"],
                "job_title": roles[current["roleId"]],
                "role_id": current["roleId"],
                "department": {
                    "department_id": p["teamId"],
                    "parent_id": None,
                    "name": teams.get(p["teamId"]),
                },
                "employee_type": 0 if part_time else 1,
                # Float's order is Sunday..Saturday.
                "work_days_hours": (
                    [0] + [week[day] / 60 for day in DAYS[:5]] + [0] if part_time else None
                ),
                "people_type_id": 2 if current["employmentType"] == "contractor" else 1,
                "active": 1,
                "tags": [
                    {"name": name, "type": 1}
                    for name in [office_tag, *person_skills.get(p["id"], [])]
                    if name
                ],
                "region_id": region_of_group.get(p["holidaysGroupId"]),
                "start_date": cs[0]["startDate"],
                "end_date": current["endDate"],
                "notes": "free text the relay drops",
                "created": STAMP,
                "modified": STAMP,
            }
        )
    for ph in runn["placeholders"]:
        people.append(
            {
                "people_id": ph["id"],
                "name": f"{ph['firstName']} {ph['lastName']}",
                "email": None,
                "job_title": ph["firstName"],
                "role_id": None,
                "department": None,
                "employee_type": 1,
                "work_days_hours": None,
                "people_type_id": 3,
                "active": 1,
                "tags": [],
                "region_id": None,
                "start_date": None,
                "end_date": None,
                "created": STAMP,
                "modified": STAMP,
            }
        )

    projects = [
        {
            "project_id": p["id"],
            "name": p["name"],
            "client_id": p["clientId"],
            "tags": [],
            "status": 2 if p["isConfirmed"] else 1,
            "stage_id": None,
            "non_billable": 1 if p["pricingModel"] == "nb" else 0,
            "active": 0 if p["isArchived"] else 1,
            "project_manager": None,
            "start_date": None,
            "end_date": None,
            "created": STAMP,
            "modified": STAMP,
        }
        for p in runn["projects"]
    ]
    confirmed = {p["id"]: p["isConfirmed"] for p in runn["projects"]}
    tasks = [
        {
            "task_id": a["id"],
            "root_task_id": a["id"],
            "parent_task_id": None,
            "project_id": a["projectId"],
            "phase_id": None,
            "task_meta_id": None,
            "name": "",
            "people_id": a["personId"],
            "start_date": a["startDate"],
            "end_date": a["endDate"],
            "start_time": None,
            "hours": a["minutesPerDay"] / 60,
            "status": 2 if confirmed[a["projectId"]] else 1,
            "billable": 1 if a["isBillable"] else 0,
            "repeat_state": 0,
            "repeat_end_date": None,
            "notes": "free text the relay drops",
            "created": STAMP,
            "modified": STAMP,
        }
        for a in runn["assignments"]
    ]
    timeoffs = [
        {
            "timeoff_id": t["id"],
            "timeoff_type_id": 2 if "Parental" in (t["note"] or "") else 1,
            "people_ids": [t["personId"]],
            "start_date": t["startDate"],
            "end_date": t["endDate"],
            "start_time": None,
            "hours": None,
            "full_day": 1,
            "status": 2,
            "repeat_state": 0,
            "repeat_end": None,
            "timeoff_notes": t["note"],
            "created": STAMP,
            "modified": STAMP,
        }
        for t in runn["time_offs_leave"]
    ]
    holidays = {h["id"]: h for h in runn["holidays"]}
    public = [
        {
            "id": hid,
            "name": holidays[hid]["name"],
            "region": region_of_group[gid],
            "dates": [holidays[hid]["date"]],
        }
        for gid, g in groups.items()
        for hid in g["holidayIds"]
    ]
    return {
        "people": people,
        "roles": [
            {
                "id": r["id"],
                "name": r["name"],
                "people_ids": r["personIds"],
                "people_count": len(r["personIds"]),
                "created": STAMP,
                "modified": STAMP,
            }
            for r in runn["roles"]
        ],
        "departments": [
            {"department_id": t["id"], "parent_id": None, "name": t["name"]} for t in runn["teams"]
        ],
        "clients": [{"client_id": c["id"], "name": c["name"]} for c in runn["clients"]],
        "projects": projects,
        "tasks": tasks,
        "timeoffs": timeoffs,
        "timeoff-types": TIMEOFF_TYPES,
        "public-holidays": public,
        "holidays": [],
    }


def people_report(world: dict, start: str, end: str, people_id: int | None = None) -> list[dict]:
    """Float's /reports/people for a window, computed from the Runn truth:
    capacity less time off and holidays, scheduled hours, the rest."""
    truth = Truth(world)
    s, e = d(start), d(end)
    rows = []
    for pid in truth.people:
        if people_id is not None and pid != people_id:
            continue
        load = truth.load(pid, s, e)
        contracted = 0
        for day in workdays(s, e):
            c = truth.contract(pid, day)
            if c:
                contracted += c["rosteredDays"].get(DAYS[day.weekday()], 0)
        scheduled = (load.confirmed + load.tentative) / 60
        capacity = load.capacity / 60
        rows.append(
            {
                "people_id": pid,
                "name": truth.name(pid),
                "department_id": truth.people[pid]["teamId"],
                "capacity": capacity,
                "timeoff": round((contracted - load.capacity) / 60, 2),
                "scheduled": scheduled,
                "billable": scheduled,
                "nonBillable": 0,
                "overtime": max(scheduled - capacity, 0),
                "unscheduled": max(capacity - scheduled, 0),
            }
        )
    return rows
