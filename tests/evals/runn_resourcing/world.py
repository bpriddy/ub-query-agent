"""
world.py — the synthetic agency behind the Runn resourcing eval.

40 people in Los Angeles, New York and London, in Runn's API v1 response
shapes (developer.runn.io: people, contracts, roles, teams, skills,
people-tags, time-offs/leave, holiday-groups, placeholders, projects,
assignments), and the matching UB records (offices, staff, staff metadata).

Everything is written by hand here, then `python -m
tests.evals.runn_resourcing.world` dumps it to `fixtures/` as JSON; the eval
and its tests read the JSON. The dates are fixed: the eval pins "today" to
TODAY (a Wednesday), so "next week" is the week of 2026-11-23, US
Thanksgiving week.

Deliberate traps, each one a question in questions.py:
  * two Janes (LA art director, London strategist);
  * part-timers: a 4-day week, two 3-day weeks, a 4-hour-day contract;
  * tentative projects (`isConfirmed: false`) booking real people;
  * placeholders (unfilled roles) on three projects;
  * leave, parental leave, US and UK public holidays;
  * a designer with the art direction skill, a creative director;
  * contractors whose contracts end mid-window;
  * UB staff with no Runn record, and one person whose Runn email differs
    from UB's;
  * office lives in UB (staff.officeId) and in Runn only as a people tag
    and a custom field (the relay drops custom fields).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

TODAY = date(2026, 11, 18)
FIXTURES = Path(__file__).parent / "fixtures"
STAMP = "2026-10-01T09:00:00.000Z"

OFFICES = {
    "LA": ("0f1c6a2e-1111-4a8e-9c01-000000000001", "Los Angeles", "Los Angeles"),
    "NYC": ("0f1c6a2e-2222-4a8e-9c01-000000000002", "New York", "New York"),
    "London": ("0f1c6a2e-3333-4a8e-9c01-000000000003", "London", "London"),
}

ROLES = {
    "Creative Director": 301,
    "Art Director": 302,
    "Designer": 303,
    "Copywriter": 304,
    "Strategist": 305,
    "Producer": 306,
}
TEAMS = {"Creative": 401, "Design": 402, "Strategy": 403, "Production": 404}
ROLE_TEAM = {
    "Creative Director": "Creative",
    "Art Director": "Creative",
    "Copywriter": "Creative",
    "Designer": "Design",
    "Strategist": "Strategy",
    "Producer": "Production",
}
SKILLS = {
    "Art Direction": 501,
    "Brand Identity": 502,
    "Motion Design": 503,
    "Retouching": 504,
    "Social Content": 505,
    "CGI / 3D": 506,
    "Copywriting": 507,
    "Brand Strategy": 508,
    "Comms Planning": 509,
    "Research": 510,
    "Broadcast Production": 511,
    "Integrated Production": 512,
    "Spanish": 513,
}
TAGS = {"LA": 601, "NYC": 602, "London": 603, "Freelancer": 604, "Part-time": 605}
HOLIDAY_GROUPS = {"US": 701, "UK": 702}
HOLIDAYS = [
    # (id, group, name, date)
    (801, "US", "Thanksgiving Day", "2026-11-26"),
    (802, "US", "Day after Thanksgiving", "2026-11-27"),
    (803, "US", "Christmas Day", "2026-12-25"),
    (804, "US", "New Year's Day", "2027-01-01"),
    (805, "US", "Martin Luther King Jr. Day", "2027-01-18"),
    (806, "US", "Presidents' Day", "2027-02-15"),
    (811, "UK", "Christmas Day", "2026-12-25"),
    (812, "UK", "Boxing Day (substitute day)", "2026-12-28"),
    (813, "UK", "New Year's Day", "2027-01-01"),
]
OFFICE_HOLIDAYS = {"LA": "US", "NYC": "US", "London": "UK"}
# A full day: 8 h in the US offices, 7.5 h in London.
FULL_DAY = {"LA": 480, "NYC": 480, "London": 450}

CLIENTS = {
    "Halcyon Hotels": 201,
    "Vela Motors": 202,
    "Lumen Bank": 203,
    "Orchard Foods": 204,
    "Northwind Air": 205,
    "Solace Health": 206,
    "Kite Mobile": 207,
    "Anomaly": 208,
}
# id: (name, client, confirmed, archived, pricing model)
PROJECTS = {
    101: ("Halcyon Hotels - Winter Campaign", "Halcyon Hotels", True, False, "fp"),
    102: ("Vela Motors - EV Launch", "Vela Motors", True, False, "tm"),
    103: ("Lumen Bank - Brand Refresh", "Lumen Bank", True, False, "fp"),
    104: ("Orchard Foods - Holiday Social", "Orchard Foods", True, False, "fp"),
    105: ("Northwind Air - Q1 Retainer", "Northwind Air", True, False, "tm"),
    106: ("Solace Health - Pitch", "Solace Health", False, False, "nb"),
    107: ("Kite Mobile - Spring Launch", "Kite Mobile", False, False, "fp"),
    108: ("Northwind Air - Always On", "Northwind Air", True, False, "tm"),
    109: ("Anomaly - New Business", "Anomaly", True, False, "nb"),
    110: ("Lumen Bank - Annual Report", "Lumen Bank", True, False, "fp"),
    111: ("Orchard Foods - Spring Packaging", "Orchard Foods", True, True, "fp"),
}

MON_THU = {"monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1, "friday": 0}
MON_WED = {"monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 0, "friday": 0}

# id, first, last, office, role, UB title, skills, contract overrides, extra tags
# Contract overrides: minutes (per day), days (rostered pattern), type, end.
PEOPLE = [
    # ── Los Angeles ──
    (
        1,
        "Jane",
        "Park",
        "LA",
        "Art Director",
        "Art Director",
        ["Art Direction", "Social Content"],
        {},
    ),
    (2, "Marcus", "Bell", "LA", "Art Director", "Art Director", ["Art Direction", "CGI / 3D"], {}),
    (
        3,
        "Sofia",
        "Reyes",
        "LA",
        "Art Director",
        "Senior Art Director",
        ["Art Direction", "Spanish"],
        {"days": MON_THU},
    ),
    (4, "Theo", "Grant", "LA", "Art Director", "Art Director", ["Art Direction"], {}),
    (
        5,
        "Priya",
        "Nair",
        "LA",
        "Art Director",
        "Art Director",
        ["Art Direction", "Brand Identity"],
        {},
    ),
    (6, "Dylan", "Cho", "LA", "Designer", "Senior Designer", ["Brand Identity", "Retouching"], {}),
    (7, "Ava", "Thompson", "LA", "Designer", "Designer", ["Motion Design", "Social Content"], {}),
    (
        8,
        "Leo",
        "Martinez",
        "LA",
        "Designer",
        "Designer",
        ["Art Direction", "Motion Design", "Spanish"],
        {},
    ),
    (9, "Hannah", "Lee", "LA", "Copywriter", "Copywriter", ["Copywriting", "Social Content"], {}),
    (
        10,
        "Omar",
        "Haddad",
        "LA",
        "Copywriter",
        "Freelance Copywriter",
        ["Copywriting"],
        {"type": "contractor", "end": "2026-12-04"},
    ),
    (11, "Grace", "Kim", "LA", "Strategist", "Strategist", ["Brand Strategy", "Research"], {}),
    (12, "Ben", "Carter", "LA", "Producer", "Senior Producer", ["Broadcast Production"], {}),
    (13, "Nina", "Patel", "LA", "Producer", "Producer", ["Integrated Production"], {}),
    (
        14,
        "Ruby",
        "Alvarez",
        "LA",
        "Designer",
        "Designer",
        ["Retouching", "Brand Identity"],
        {"minutes": 240},
    ),
    (
        15,
        "Chris",
        "Doyle",
        "LA",
        "Creative Director",
        "Executive Creative Director",
        ["Art Direction", "Copywriting"],
        {},
    ),
    # ── New York ──
    (
        16,
        "Ethan",
        "Brooks",
        "NYC",
        "Art Director",
        "Art Director",
        ["Art Direction", "Motion Design"],
        {},
    ),
    (17, "Maya", "Singh", "NYC", "Art Director", "Senior Art Director", ["Art Direction"], {}),
    (18, "Lucas", "Wright", "NYC", "Designer", "Designer", ["Motion Design", "CGI / 3D"], {}),
    (19, "Zoe", "Chen", "NYC", "Designer", "Senior Designer", ["Brand Identity"], {}),
    (20, "Isaac", "Moore", "NYC", "Designer", "Designer", ["Brand Identity", "Social Content"], {}),
    (21, "Olivia", "Hayes", "NYC", "Copywriter", "Copywriter", ["Copywriting"], {}),
    (22, "Daniel", "Kim", "NYC", "Strategist", "Senior Strategist", ["Brand Strategy"], {}),
    (
        23,
        "Fatima",
        "Ali",
        "NYC",
        "Strategist",
        "Strategist",
        ["Comms Planning", "Research"],
        {"days": MON_WED},
    ),
    (24, "Sam", "Rivera", "NYC", "Producer", "Senior Producer", ["Integrated Production"], {}),
    (25, "Emma", "Collins", "NYC", "Producer", "Producer", ["Broadcast Production"], {}),
    (26, "Jack", "Turner", "NYC", "Copywriter", "Senior Copywriter", ["Copywriting"], {}),
    (27, "Lily", "Wong", "NYC", "Designer", "Designer", ["Motion Design"], {}),
    (
        28,
        "Ryan",
        "Foster",
        "NYC",
        "Strategist",
        "Head of Strategy",
        ["Brand Strategy", "Comms Planning"],
        {},
    ),
    (
        29,
        "Chloe",
        "Martin",
        "NYC",
        "Art Director",
        "Art Director",
        ["Art Direction", "Retouching"],
        {},
    ),
    # ── London ──
    (
        30,
        "Jane",
        "Okafor",
        "London",
        "Strategist",
        "Strategy Director",
        ["Brand Strategy", "Research"],
        {},
    ),
    (31, "Oliver", "Hughes", "London", "Art Director", "Art Director", ["Art Direction"], {}),
    (
        32,
        "Amelia",
        "Clarke",
        "London",
        "Art Director",
        "Art Director",
        ["Art Direction", "Social Content"],
        {},
    ),
    (
        33,
        "Harry",
        "Evans",
        "London",
        "Designer",
        "Designer",
        ["Brand Identity", "Motion Design"],
        {},
    ),
    (34, "Isla", "Murray", "London", "Designer", "Designer", ["Retouching"], {"days": MON_WED}),
    (35, "George", "Patel", "London", "Copywriter", "Copywriter", ["Copywriting"], {}),
    (36, "Freya", "Lewis", "London", "Strategist", "Strategist", ["Comms Planning"], {}),
    (37, "Noah", "Bennett", "London", "Producer", "Producer", ["Integrated Production"], {}),
    (38, "Poppy", "Shaw", "London", "Producer", "Producer", ["Broadcast Production"], {}),
    (
        39,
        "Arjun",
        "Mehta",
        "London",
        "Strategist",
        "Strategist",
        ["Research", "Brand Strategy"],
        {},
    ),
    (
        40,
        "Ella",
        "Ward",
        "London",
        "Designer",
        "Freelance Designer",
        ["Motion Design", "CGI / 3D"],
        {"type": "contractor", "end": "2026-12-23"},
    ),
]

# Runn's email for a person differs from UB's: the London domain.
RUNN_EMAIL_OVERRIDES = {33: "harry.evans@anomaly.co.uk"}

# UB staff with no Runn person at all.
UB_ONLY = [
    ("Kai", "Nakamura", "LA", "Art Director", "Creative", "2026-11-16", ["Art Direction"]),
    ("Rosa", "Diaz", "NYC", "Producer", "Production", "2026-10-26", ["Integrated Production"]),
]
# UB status other than active.
UB_STATUS = {19: "on_leave"}

# Placeholders: unfilled roles, Runn's `/placeholders/` records.
PLACEHOLDERS = [
    (901, "Designer", "Placeholder", "Designer"),
    (902, "Producer", "Placeholder", "Producer"),
    (903, "Art Director", "Placeholder", "Art Director"),
]

# person, project, start, end, minutes per day
ASSIGNMENTS = [
    # LA
    (1, 101, "2026-11-02", "2026-12-18", 480),
    (2, 104, "2026-11-02", "2026-11-20", 480),
    (2, 105, "2027-01-04", "2027-03-31", 240),
    (3, 102, "2026-10-05", "2027-01-29", 240),
    (4, 102, "2026-10-05", "2026-12-18", 480),
    (5, 106, "2026-11-16", "2026-12-04", 480),
    (6, 101, "2026-11-02", "2026-12-18", 480),
    (6, 104, "2026-11-09", "2026-11-30", 240),
    (7, 104, "2026-11-02", "2026-11-20", 480),
    (7, 105, "2027-01-04", "2027-03-31", 480),
    (8, 109, "2026-11-02", "2026-12-31", 120),
    (9, 101, "2026-11-02", "2026-12-18", 360),
    (10, 106, "2026-11-16", "2026-12-04", 480),
    (11, 102, "2026-10-05", "2027-01-29", 240),
    (11, 105, "2027-01-04", "2027-03-31", 240),
    (12, 101, "2026-11-02", "2026-12-18", 480),
    (12, 102, "2026-10-05", "2027-01-29", 240),
    (13, 104, "2026-11-02", "2026-12-23", 240),
    (14, 102, "2026-10-05", "2027-01-29", 240),
    (15, 101, "2026-11-02", "2026-12-18", 240),
    (15, 102, "2026-10-05", "2027-01-29", 240),
    # NYC
    (16, 108, "2026-09-01", "2026-11-20", 480),
    (16, 107, "2027-01-11", "2027-03-26", 480),
    (17, 103, "2026-11-09", "2027-02-26", 480),
    (17, 108, "2026-09-01", "2026-12-31", 240),
    (18, 103, "2026-11-09", "2027-02-26", 480),
    (20, 108, "2026-09-01", "2026-11-20", 480),
    (20, 105, "2027-01-04", "2027-03-31", 480),
    (21, 103, "2026-11-09", "2027-02-26", 480),
    (22, 103, "2026-11-09", "2027-02-26", 240),
    (22, 105, "2027-01-04", "2027-03-31", 480),
    (23, 108, "2026-09-01", "2026-12-31", 480),
    (23, 107, "2027-01-11", "2027-03-26", 480),
    (24, 103, "2026-11-09", "2027-02-26", 480),
    (24, 108, "2026-09-01", "2026-12-31", 240),
    (24, 106, "2026-11-16", "2026-12-04", 120),
    (25, 108, "2026-09-01", "2026-12-31", 240),
    (26, 108, "2026-09-01", "2026-12-11", 480),
    (27, 107, "2027-01-11", "2027-03-26", 480),
    (27, 106, "2026-11-16", "2026-12-04", 480),
    (28, 103, "2026-11-09", "2027-02-26", 240),
    (28, 109, "2026-11-02", "2027-03-31", 120),
    (29, 108, "2026-09-01", "2026-12-18", 480),
    (29, 105, "2027-01-04", "2027-03-31", 240),
    # London
    (30, 110, "2026-11-23", "2026-12-11", 450),
    (30, 103, "2026-12-14", "2027-02-26", 225),
    (31, 110, "2026-11-23", "2026-12-11", 450),
    (32, 104, "2026-11-02", "2026-11-20", 450),
    (32, 107, "2027-01-11", "2027-03-26", 450),
    (33, 110, "2026-11-23", "2026-12-11", 450),
    (34, 110, "2026-11-23", "2026-12-11", 450),
    (35, 110, "2026-11-23", "2026-12-11", 225),
    (35, 107, "2027-01-11", "2027-03-26", 225),
    (36, 103, "2026-11-09", "2027-02-26", 225),
    (37, 110, "2026-11-23", "2026-12-11", 450),
    (37, 103, "2026-11-09", "2027-02-26", 225),
    (38, 107, "2027-01-11", "2027-03-26", 450),
    (39, 105, "2027-01-04", "2027-03-31", 450),
    (39, 109, "2026-11-02", "2026-11-30", 90),
    # Placeholders: the unfilled roles
    (901, 103, "2026-11-23", "2026-12-18", 480),
    (902, 105, "2027-01-04", "2027-03-31", 480),
    (903, 106, "2026-11-23", "2026-12-04", 480),
]

# person, start, end, note
LEAVE = [
    (4, "2026-11-23", "2026-11-27", "Vacation"),
    (19, "2026-11-02", "2027-01-29", "Parental leave"),
    (26, "2026-12-14", "2026-12-31", "Vacation"),
    (29, "2026-12-21", "2027-01-01", "Vacation"),
    (36, "2026-12-21", "2027-01-04", "Vacation"),
    (13, "2026-12-21", "2026-12-24", "Vacation"),
]


def _email(first: str, last: str) -> str:
    return f"{first}.{last}@anomaly.com".lower()


def _rostered(minutes: int, days: dict | None) -> dict:
    days = days or dict.fromkeys(("monday", "tuesday", "wednesday", "thursday", "friday"), 1)
    return {day: minutes * on for day, on in days.items()}


def build_runn() -> dict:
    people, contracts, competencies = [], [], []
    for pid, first, last, office, role, _title, skills, over in PEOPLE:
        tags = [{"id": TAGS[office], "name": office}]
        if over.get("type") == "contractor":
            tags.append({"id": TAGS["Freelancer"], "name": "Freelancer"})
        if over.get("days") or over.get("minutes"):
            tags.append({"id": TAGS["Part-time"], "name": "Part-time"})
        person_skills = [
            {"id": SKILLS[s], "level": 3 if i == 0 else 2} for i, s in enumerate(skills)
        ]
        people.append(
            {
                "id": pid,
                "firstName": first,
                "lastName": last,
                "email": RUNN_EMAIL_OVERRIDES.get(pid, _email(first, last)),
                "isArchived": False,
                "references": [],
                "teamId": TEAMS[ROLE_TEAM[role]],
                "tags": tags,
                "skills": person_skills,
                "holidaysGroupId": HOLIDAY_GROUPS[OFFICE_HOLIDAYS[office]],
                "customFields": {
                    "select": [
                        {
                            "id": 1,
                            "name": "Office",
                            "values": [{"id": TAGS[office], "name": office}],
                        }
                    ],
                    "text": [],
                    "checkbox": [],
                    "date": [],
                },
                "managers": [],
                "createdAt": STAMP,
                "updatedAt": STAMP,
            }
        )
        minutes = over.get("minutes", FULL_DAY[office])
        contracts.append(
            {
                "id": 1000 + pid,
                "personId": pid,
                "roleId": ROLES[role],
                "jobTitle": None,
                "employmentType": over.get("type", "employee"),
                "minutesPerDay": minutes,
                "rosteredDays": _rostered(minutes, over.get("days")),
                "costPerHour": None,
                "startDate": "2025-01-06",
                "endDate": over.get("end"),
                "createdAt": STAMP,
                "updatedAt": STAMP,
            }
        )
        for s in person_skills:
            competencies.append(
                {
                    "personId": pid,
                    "skillId": s["id"],
                    "level": s["level"],
                    "createdAt": STAMP,
                    "updatedAt": STAMP,
                }
            )

    placeholders = []
    for phid, first, last, role in PLACEHOLDERS:
        placeholders.append(
            {
                "id": phid,
                "firstName": first,
                "lastName": last,
                "isArchived": False,
                "tags": [],
                "references": [],
                "createdAt": STAMP,
                "updatedAt": STAMP,
            }
        )
        contracts.append(
            {
                "id": 1000 + phid,
                "personId": phid,
                "roleId": ROLES[role],
                "jobTitle": None,
                "employmentType": "employee",
                "minutesPerDay": 480,
                "rosteredDays": _rostered(480, None),
                "costPerHour": None,
                "startDate": "2026-11-01",
                "endDate": None,
                "createdAt": STAMP,
                "updatedAt": STAMP,
            }
        )

    role_of = {c["personId"]: c["roleId"] for c in contracts}
    placeholder_ids = {p[0] for p in PLACEHOLDERS}
    assignments = [
        {
            "id": 3000 + i,
            "personId": person,
            "projectId": project,
            "roleId": role_of[person],
            "phaseId": None,
            "workstreamId": None,
            "startDate": start,
            "endDate": end,
            "minutesPerDay": minutes,
            "isBillable": PROJECTS[project][4] != "nb",
            "isNonWorkingDay": False,
            "isPlaceholder": person in placeholder_ids,
            "isTemplate": False,
            "isActive": True,
            "note": None,
            "createdAt": STAMP,
            "updatedAt": STAMP,
        }
        for i, (person, project, start, end, minutes) in enumerate(ASSIGNMENTS, start=1)
    ]

    projects = [
        {
            "id": prid,
            "name": name,
            "isTemplate": False,
            "isArchived": archived,
            "isConfirmed": confirmed,
            "pricingModel": pricing,
            "rateType": "hours",
            "budgetMethod": "ROLE",
            "teamId": None,
            "budget": None,
            "expensesBudget": None,
            "references": [],
            "clientId": CLIENTS[client],
            "rateCardId": 1,
            "customFields": {"select": [], "text": [], "checkbox": [], "date": []},
            "managerIds": [],
            "tags": [],
            "createdAt": STAMP,
            "updatedAt": STAMP,
        }
        for prid, (name, client, confirmed, archived, pricing) in PROJECTS.items()
    ]

    leave = [
        {
            "id": 2000 + i,
            "personId": person,
            "startDate": start,
            "endDate": end,
            "note": note,
            "minutesPerDay": None,
            "createdAt": STAMP,
            "updatedAt": STAMP,
        }
        for i, (person, start, end, note) in enumerate(LEAVE, start=1)
    ]
    holiday_people = {
        g: [p[0] for p in PEOPLE if OFFICE_HOLIDAYS[p[3]] == g] for g in HOLIDAY_GROUPS
    }
    holidays_timeoff = [
        {
            "id": 2500 + n,
            "personId": person,
            "startDate": day,
            "endDate": day,
            "note": name,
            "minutesPerDay": None,
            "holidayId": hid,
            "createdAt": STAMP,
            "updatedAt": STAMP,
        }
        for n, (hid, group, name, day, person) in enumerate(
            (h + (p,) for h in HOLIDAYS for p in holiday_people[h[1]]), start=1
        )
    ]

    return {
        "people": people,
        "placeholders": placeholders,
        "contracts": contracts,
        "roles": [
            {
                "id": rid,
                "name": name,
                "isArchived": False,
                "defaultHourCost": 0,
                "standardRate": 0,
                "references": [],
                "personIds": sorted(c["personId"] for c in contracts if c["roleId"] == rid),
                "createdAt": STAMP,
                "updatedAt": STAMP,
            }
            for name, rid in ROLES.items()
        ],
        "teams": [
            {"id": tid, "name": name, "createdAt": STAMP, "updatedAt": STAMP}
            for name, tid in TEAMS.items()
        ],
        "skills": [
            {"id": sid, "name": name, "createdAt": STAMP, "updatedAt": STAMP}
            for name, sid in SKILLS.items()
        ],
        "people_skills": competencies,
        "people_tags": [
            {"id": tid, "name": name, "archived": False, "createdAt": STAMP, "updatedAt": STAMP}
            for name, tid in TAGS.items()
        ],
        "clients": [{"id": cid, "name": name} for name, cid in CLIENTS.items()],
        "projects": projects,
        "assignments": assignments,
        "time_offs_leave": leave,
        "time_offs_holidays": holidays_timeoff,
        "holiday_groups": [
            {
                "id": gid,
                "name": "United States" if g == "US" else "United Kingdom",
                "countryCode": g if g == "US" else "GB",
                "countryName": "United States" if g == "US" else "United Kingdom",
                "regionName": None,
                "holidayIds": [h[0] for h in HOLIDAYS if h[1] == g],
            }
            for g, gid in HOLIDAY_GROUPS.items()
        ],
        "holidays": [
            {"id": hid, "name": name, "date": day, "observed": None, "type": "public"}
            for hid, _g, name, day in HOLIDAYS
        ],
    }


def _staff_uuid(n: int) -> str:
    return f"5aff0000-0000-4000-8000-{n:012d}"


def build_ub() -> dict:
    offices = [
        {
            "id": oid,
            "name": name,
            "syncCity": city,
            "isActive": True,
            "startedAt": "2015-01-01T00:00:00.000Z",
            "currentState": {},
            "createdAt": STAMP,
            "updatedAt": STAMP,
        }
        for oid, name, city in OFFICES.values()
    ]
    staff, metadata = [], []
    rows = [
        (pid, first, last, office, title, ROLE_TEAM[role], "2023-03-01", skills)
        for pid, first, last, office, role, title, skills, _o in PEOPLE
    ]
    rows += [
        (41 + i, first, last, office, title, dept, started, skills)
        for i, (first, last, office, title, dept, started, skills) in enumerate(UB_ONLY)
    ]
    for n, first, last, office, title, dept, started, skills in rows:
        sid = _staff_uuid(n)
        staff.append(
            {
                "id": sid,
                "fullName": f"{first} {last}",
                "email": _email(first, last),
                "title": title,
                "department": dept,
                "status": UB_STATUS.get(n, "active"),
                "startedAt": f"{started}T00:00:00.000Z",
                "endedAt": None,
                "officeId": OFFICES[office][0],
                "isActive": True,
            }
        )
        for k, label in enumerate(skills):
            metadata.append(
                {
                    "id": f"3e7a0000-0000-4000-8000-{n:06d}{k:06d}",
                    "staffId": sid,
                    "type": "skill",
                    "label": label,
                    "value": "expert" if k == 0 else "proficient",
                    "notes": None,
                    "metadata": None,
                    "isFeatured": k == 0,
                    "createdAt": STAMP,
                    "updatedAt": STAMP,
                }
            )
    return {"offices": offices, "staff": staff, "staff_metadata": metadata}


def build() -> dict:
    return {"today": TODAY.isoformat(), "runn": build_runn(), "ub": build_ub()}


def write() -> None:
    FIXTURES.mkdir(exist_ok=True)
    world = build()
    (FIXTURES / "runn.json").write_text(json.dumps(world["runn"], indent=1) + "\n")
    (FIXTURES / "ub.json").write_text(json.dumps(world["ub"], indent=1) + "\n")


def load() -> dict:
    return {
        "today": TODAY.isoformat(),
        "runn": json.loads((FIXTURES / "runn.json").read_text()),
        "ub": json.loads((FIXTURES / "ub.json").read_text()),
    }


if __name__ == "__main__":
    write()
