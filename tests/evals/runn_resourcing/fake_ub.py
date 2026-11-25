"""
fake_ub.py — a localhost UB serving the eval's fixtures, shaped like UB.

The Runn half is the relay as of gcp-universal-backend#266
(src/modules/relay/runn/): people, projects, assignments, contracts,
time-offs/{leave,rostered-off,holidays}, holiday-groups (+ /:id/holidays),
roles, teams, skills (+ /:id/people), people-tags and custom-fields/select,
with the relay's per-resource field allowlist (`runn.client.ts` FIELDS: no
notes, rates or budgets; of custom fields only select ones), its
query-parameter allowlist (anything else is a 400 RUNN_BAD_QUERY),
`includeArchived=true` by default on projects, placeholders left out of
/people unless `includePlaceholders=true`, and Runn's `{values, nextCursor}`
pages. As in Runn, a LIST of people carries no skills or custom fields; only
GET /people/:id does. Date filters match on overlap.

The Float half is the relay of gcp-universal-backend#264 over the same agency
in Float's shapes (float_world.py), with `/reports/people` computed from the
ground truth. Only the planners in `sources` are configured; the other relay
answers 503 *_NOT_CONFIGURED, as UB does without a key.

The resourcing half is the relay of gcp-universal-backend#301
(`/relay/resourcing/*`) over a Runn leg built from the same Runn records the
way the leg's `aggregate_v1` views are (runn-resourcing-dashboard
`lib/db/migrations/aggregate_v1.sql`): people, projects, assignments,
working_patterns, time_off and source, blended (each row with `source`),
with resourcing's query syntax (tests/helpers.py `resourcing_read`). Capacity
is computed from the ground truth (truth.py), since resourcing computes it;
`capacity=False` answers 503 RESOURCING_CAPACITY_OFF, as UB does today.

The UB half is what the agent's UB tools call: /org/staff (+ :id, :id/metadata),
/org/resourcing (`type` required; `officeId` and `q` ignored, as UB's
controller does), /org/offices, /org/search (staff only) and /org/query for
staff, offices and teams (eq, neq, in, like, similar_to, is_null, between,
gt/gte/lt/lte; group_by + count; sort; limit). Every other path is a 404, or
an empty list for the account and campaign listings.

Each request is logged with the bearer token, which the harness sets per
question (`eval-<id>`), so a question's relay calls can be read back.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from tests.evals.runn_resourcing.truth import DAYS, Truth, d, workdays
from tests.helpers import resourcing_read

TIME_OFF = ["id", "personId", "startDate", "endDate", "minutesPerDay"]
# The relay's FIELDS (runn.client.ts), resource by resource.
RUNN_FIELDS = {
    "people": [
        "id",
        "firstName",
        "lastName",
        "email",
        "isArchived",
        "teamId",
        "holidaysGroupId",
        "tags",
        "skills",
        "customFields",
        "managers",
        "createdAt",
        "updatedAt",
    ],
    "projects": [
        "id",
        "name",
        "isTemplate",
        "isArchived",
        "isConfirmed",
        "teamId",
        "clientId",
        "managerIds",
        "tags",
        "createdAt",
        "updatedAt",
    ],
    "assignments": [
        "id",
        "personId",
        "projectId",
        "roleId",
        "phaseId",
        "workstreamId",
        "startDate",
        "endDate",
        "minutesPerDay",
        "isBillable",
        "isNonWorkingDay",
        "isPlaceholder",
        "isTemplate",
        "createdAt",
        "updatedAt",
    ],
    "contracts": [
        "id",
        "personId",
        "roleId",
        "employmentType",
        "minutesPerDay",
        "rosteredDays",
        "startDate",
        "endDate",
        "createdAt",
        "updatedAt",
    ],
    "time-offs/leave": TIME_OFF,
    "time-offs/rostered-off": TIME_OFF,
    "time-offs/holidays": [*TIME_OFF, "holidayId"],
    "holiday-groups": ["id", "name", "countryCode", "countryName", "regionName", "holidayIds"],
    "holidays": ["id", "name", "date", "observed", "type"],
    "roles": ["id", "name", "isArchived", "personIds", "createdAt", "updatedAt"],
    "teams": ["id", "name", "createdAt", "updatedAt"],
    "skills": ["id", "name", "createdAt", "updatedAt"],
    "people-tags": ["id", "name", "archived", "createdAt", "updatedAt"],
    "custom-fields/select": ["id", "name", "model", "options", "singleSelect"],
}
PAGE_PARAMS = {"cursor", "limit"}
LIST_PARAMS = PAGE_PARAMS | {"modifiedAfter", "sortBy", "order"}
RUNN_PARAMS = {
    "people": LIST_PARAMS | {"includePlaceholders"},
    "projects": LIST_PARAMS | {"includeArchived"},
    "assignments": LIST_PARAMS | {"personId", "projectId", "roleId", "startDate", "endDate"},
    "contracts": LIST_PARAMS,
    "time-offs/leave": LIST_PARAMS | {"personId", "startDate", "endDate"},
    "time-offs/rostered-off": LIST_PARAMS | {"personId"},
    "time-offs/holidays": LIST_PARAMS | {"personId"},
    "holiday-groups": PAGE_PARAMS | {"sortBy", "order"},
    "roles": LIST_PARAMS | {"name"},
    "teams": LIST_PARAMS,
    "skills": LIST_PARAMS,
    "people-tags": LIST_PARAMS,
    "custom-fields/select": PAGE_PARAMS | {"model"},
}
# Records the relay serves one at a time (GET /relay/runn/<resource>/:id).
RUNN_ONE = {"people", "projects", "holiday-groups", "roles", "teams", "skills"}
# Runn's list of people is its CollectionPerson: no skills, no custom fields
# (developer.runn.io); only GET /people/:id carries them.
COLLECTION_PERSON_DROPS = ("skills", "customFields")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
INT = re.compile(r"^[1-9]\d*$")

# The Float relay (float.client.ts): resources, fields kept, params allowed.
FLOAT_FIELDS = {
    "people": [
        "people_id",
        "name",
        "email",
        "job_title",
        "role_id",
        "department",
        "employee_type",
        "work_days_hours",
        "people_type_id",
        "active",
        "tags",
        "region_id",
        "start_date",
        "end_date",
        "created",
        "modified",
    ],
    "departments": ["department_id", "parent_id", "name"],
    "roles": ["id", "name", "people_ids", "people_count", "created", "modified"],
    "projects": [
        "project_id",
        "name",
        "client_id",
        "tags",
        "status",
        "stage_id",
        "non_billable",
        "active",
        "project_manager",
        "start_date",
        "end_date",
        "created",
        "modified",
    ],
    "clients": ["client_id", "name"],
    "tasks": [
        "task_id",
        "root_task_id",
        "parent_task_id",
        "project_id",
        "phase_id",
        "task_meta_id",
        "name",
        "people_id",
        "people_ids",
        "start_date",
        "end_date",
        "start_time",
        "hours",
        "status",
        "billable",
        "repeat_state",
        "repeat_end_date",
        "created",
        "modified",
    ],
    "timeoffs": [
        "timeoff_id",
        "timeoff_type_id",
        "people_ids",
        "start_date",
        "end_date",
        "start_time",
        "hours",
        "full_day",
        "status",
        "repeat_state",
        "repeat_end",
        "created",
        "modified",
    ],
    "timeoff-types": ["timeoff_type_id", "timeoff_type_name", "active"],
    "public-holidays": ["id", "name", "region", "dates"],
    "holidays": ["holiday_id", "name", "date", "end_date"],
}
FLOAT_ID = {
    "people": "people_id",
    "departments": "department_id",
    "roles": "id",
    "projects": "project_id",
    "clients": "client_id",
    "tasks": "task_id",
    "timeoffs": "timeoff_id",
}
FLOAT_PAGE = {"page", "per-page"}
FLOAT_LIST = FLOAT_PAGE | {"sort", "modified_since"}
FLOAT_RANGE = {"start_date", "end_date"}
FLOAT_PARAMS = {
    "people": FLOAT_LIST
    | {"active", "department_id", "email", "people_type_id", "employee_type", "tag_name"},
    "departments": FLOAT_PAGE,
    "roles": FLOAT_PAGE | {"expand"},
    "projects": FLOAT_LIST | {"client_id", "active", "non_billable", "status", "tag_name"},
    "clients": FLOAT_PAGE,
    "tasks": FLOAT_LIST
    | FLOAT_RANGE
    | {"client_id", "project_id", "phase_id", "task_meta_id", "people_id", "billable", "status"},
    "timeoffs": FLOAT_LIST | FLOAT_RANGE | {"full_day", "status", "timeoff_type_id"},
    "timeoff-types": FLOAT_PAGE | {"active"},
    "public-holidays": FLOAT_PAGE | FLOAT_RANGE | {"year", "region"},
    "holidays": FLOAT_PAGE,
}
FLOAT_PAIRED = {"tasks", "timeoffs", "public-holidays"}


def _pick(resource: str, row: dict) -> dict:
    out = {k: row[k] for k in RUNN_FIELDS[resource] if k in row}
    if resource == "people" and "customFields" in out:
        out["customFields"] = {
            "select": [
                {
                    "id": f["id"],
                    "name": f["name"],
                    "values": [{"id": v["id"], "name": v["name"]} for v in f["values"]],
                }
                for f in (out["customFields"] or {}).get("select", [])
            ]
        }
    return out


def _bad(name: str, resource: str) -> tuple[int, dict]:
    return 400, {
        "code": "RUNN_BAD_QUERY",
        "message": f'Runn {resource} does not take the "{name}" parameter here.',
    }


def _float_bad(name: str, resource: str) -> tuple[int, dict]:
    return 400, {
        "code": "FLOAT_BAD_QUERY",
        "message": f'Float {resource} does not take the "{name}" parameter here.',
    }


def _overlaps(row: dict, start: str | None, end: str | None, s="startDate", e="endDate") -> bool:
    if start and (row.get(e) or "9999") < start:
        return False
    return not (end and (row.get(s) or "0000") > end)


NOT_CONFIGURED = {
    "runn": (
        503,
        {
            "code": "RUNN_NOT_CONFIGURED",
            "message": "The Runn relay is not configured in this environment (no Runn API key).",
        },
    ),
    "float": (
        503,
        {
            "code": "FLOAT_NOT_CONFIGURED",
            "message": "The Float relay is not configured in this environment.",
        },
    ),
}


class FakeUb:
    def __init__(
        self, world: dict, sources: tuple[str, ...] = ("runn",), capacity: bool = True
    ) -> None:
        self.world = world
        #: Whether UB serves /relay/resourcing/capacity (RESOURCING_CAPACITY).
        self.capacity = capacity
        self.runn = world["runn"]
        self.float = world.get("float")
        self.ub = world["ub"]
        #: The planners this UB has a key for; the other relay answers 503.
        self.sources = sources
        self.log: dict[str, list[dict]] = defaultdict(list)
        self._server: asyncio.Server | None = None
        self.port: int | None = None
        offices = {o["id"]: o["name"] for o in self.ub["offices"]}
        self.staff = [{**s, "officeName": offices.get(s["officeId"])} for s in self.ub["staff"]]

    # ── server ──
    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                head = (await reader.readuntil(b"\r\n\r\n")).decode()
                request_line, *header_lines = head.split("\r\n")
                method, target, _ = request_line.split(" ")
                headers = {}
                for line in header_lines:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        headers[k.strip().lower()] = v.strip()
                length = int(headers.get("content-length", "0") or 0)
                raw = await reader.readexactly(length) if length else b""
                body = json.loads(raw) if raw else None
                parts = urlsplit(target)
                params = dict(parse_qsl(parts.query))
                status, payload = self.route(method, parts.path, params, body)
                caller = headers.get("authorization", "").removeprefix("Bearer ").strip()
                self.log[caller].append(
                    {
                        "method": method,
                        "path": parts.path,
                        "params": params,
                        "body": body,
                        "status": status,
                    }
                )
                data = json.dumps(payload).encode()
                writer.write(
                    (
                        f"HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n"
                        f"Content-Length: {len(data)}\r\nConnection: keep-alive\r\n\r\n"
                    ).encode()
                    + data
                )
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()

    # ── routing ──
    def route(self, method: str, path: str, params: dict, body: Any) -> tuple[int, Any]:
        segs = [s for s in path.split("/") if s]
        if segs[:2] == ["relay", "runn"] and method == "GET":
            if "runn" not in self.sources:
                return NOT_CONFIGURED["runn"]
            return self.relay(segs[2:], params)
        if segs[:2] == ["relay", "float"] and method == "GET":
            if "float" not in self.sources or self.float is None:
                return NOT_CONFIGURED["float"]
            return self.float_relay(segs[2:], params)
        if segs[:2] == ["relay", "resourcing"] and method == "GET":
            if "runn" not in self.sources:
                return 503, {
                    "code": "RESOURCING_NOT_CONFIGURED",
                    "message": "The resourcing relay is not configured in this environment.",
                }
            return self.resourcing_relay(segs[2:], params)
        if segs[:1] == ["org"]:
            return self.org(method, segs[1:], params, body)
        return 404, {"code": "NOT_FOUND", "message": "not found"}

    # ── Runn relay ──
    def _runn_rows(self, resource: str) -> list[dict]:
        r = self.runn
        if resource == "custom-fields/select":
            offices = sorted(
                {
                    (v["id"], v["name"])
                    for p in r["people"]
                    for f in p["customFields"]["select"]
                    for v in f["values"]
                }
            )
            return [
                {
                    "id": 1,
                    "name": "Office",
                    "description": "free text the relay drops",
                    "model": "PERSON",
                    "options": [{"id": i, "name": n} for i, n in offices],
                    "singleSelect": True,
                }
            ]
        return {
            "people": r["people"],
            "projects": r["projects"],
            "assignments": r["assignments"],
            "contracts": r["contracts"],
            "time-offs/leave": r["time_offs_leave"],
            "time-offs/rostered-off": [],
            "time-offs/holidays": r["time_offs_holidays"],
            "holiday-groups": r["holiday_groups"],
            "holidays": r["holidays"],
            "roles": r["roles"],
            "teams": r["teams"],
            "skills": r["skills"],
            "people-tags": r["people_tags"],
        }[resource]

    def _page(self, resource: str, rows: list[dict], params: dict, drops=()) -> tuple[int, Any]:
        limit = int(params.get("limit", "50"))
        offset = int(params.get("cursor", "0") or 0)
        page = rows[offset : offset + limit]
        more = offset + limit < len(rows)
        values = []
        for r in page:
            row = _pick(resource, r)
            for k in drops:
                row.pop(k, None)
            values.append(row)
        return 200, {"values": values, "nextCursor": str(offset + limit) if more else None}

    def _check(self, resource: str, params: dict, allowed: set[str]) -> tuple[int, dict] | None:
        for name, value in params.items():
            if name not in allowed:
                return _bad(name, resource)
            if name in ("startDate", "endDate") and not DATE.match(value):
                return _bad(name, resource)
            if name in ("personId", "projectId", "roleId") and not INT.match(value):
                return _bad(name, resource)
            if name == "limit" and not (INT.match(value) and int(value) <= 200):
                return _bad(name, resource)
            if name == "model" and value not in ("PERSON", "PROJECT"):
                return _bad(name, resource)
        return None

    def relay(self, segs: list[str], params: dict) -> tuple[int, Any]:
        path = "/".join(segs)
        # A child list: /holiday-groups/:id/holidays, /skills/:id/people.
        if len(segs) == 3 and (segs[0], segs[2]) in (
            ("holiday-groups", "holidays"),
            ("skills", "people"),
        ):
            parent, rid, child = segs
            bad = self._check(child, params, PAGE_PARAMS)
            if bad:
                return bad
            if not INT.match(rid):
                return 404, {"code": "RUNN_NOT_FOUND", "message": f"No Runn {parent} {rid}."}
            if parent == "holiday-groups":
                group = next((g for g in self.runn["holiday_groups"] if g["id"] == int(rid)), None)
                if group is None:
                    return 404, {"code": "RUNN_NOT_FOUND", "message": f"No Runn {parent} {rid}."}
                rows = [h for h in self.runn["holidays"] if h["id"] in group["holidayIds"]]
                return self._page("holidays", rows, params)
            holders = {ps["personId"] for ps in self.runn["people_skills"]}
            holders = {
                ps["personId"] for ps in self.runn["people_skills"] if ps["skillId"] == int(rid)
            } & holders
            if not any(sk["id"] == int(rid) for sk in self.runn["skills"]):
                return 404, {"code": "RUNN_NOT_FOUND", "message": f"No Runn {parent} {rid}."}
            rows = [p for p in self.runn["people"] if p["id"] in holders]
            return self._page("people", rows, params, COLLECTION_PERSON_DROPS)
        if len(segs) == 2 and segs[0] in RUNN_ONE:
            resource, rid = segs
            if not INT.match(rid):
                return 404, {"code": "RUNN_NOT_FOUND", "message": f"No Runn {resource} {rid}."}
            rows = self._runn_rows(resource) + (
                self.runn["placeholders"] if resource == "people" else []
            )
            for row in rows:
                if row["id"] == int(rid):
                    out = _pick(resource, row)
                    if resource == "people" and row["id"] < 900:
                        # GET /people/:id carries the person's skills.
                        out["skills"] = [
                            {"id": ps["skillId"], "level": ps["level"]}
                            for ps in self.runn["people_skills"]
                            if ps["personId"] == row["id"]
                        ]
                    return 200, out
            return 404, {"code": "RUNN_NOT_FOUND", "message": f"No Runn {resource} {rid}."}
        if path not in RUNN_PARAMS:
            return 404, {"code": "NOT_FOUND", "message": "not found"}
        resource = path
        bad = self._check(resource, params, RUNN_PARAMS[resource])
        if bad:
            return bad
        rows = list(self._runn_rows(resource))
        drops: tuple[str, ...] = ()
        if resource == "people":
            if params.get("includePlaceholders") == "true":
                rows += self.runn["placeholders"]
            drops = COLLECTION_PERSON_DROPS
        elif resource == "projects":
            if params.get("includeArchived", "true") != "true":
                rows = [p for p in rows if not p["isArchived"]]
        elif resource == "roles" and params.get("name"):
            rows = [r for r in rows if params["name"].lower() in (r["name"] or "").lower()]
        else:
            for key in ("personId", "projectId", "roleId"):
                if key in params:
                    rows = [a for a in rows if a.get(key) == int(params[key])]
            rows = [a for a in rows if _overlaps(a, params.get("startDate"), params.get("endDate"))]
        return self._page(resource, rows, params, drops)

    # ── resourcing relay (Runn as its one leg) ──
    def _resourcing_views(self) -> dict[str, list[dict]]:
        if hasattr(self, "_views"):
            return self._views
        r = self.runn
        truth = Truth(self.world)
        teams = {t["id"]: t["name"] for t in r["teams"]}
        clients = {c["id"]: c["name"] for c in r["clients"]}
        roles = {x["id"]: x["name"] for x in r["roles"]}
        placeholders = {p["id"] for p in r["placeholders"]}
        people = [
            {
                "id": str(p["id"]),
                "name": f"{p['firstName']} {p['lastName']}",
                "email": (p.get("email") or "").strip().lower() or None,
                "team": teams.get(p.get("teamId")),
                "is_archived": bool(p.get("isArchived")),
                "role": truth.role(p["id"]),
                "tags": [t["name"] for t in p.get("tags") or []],
            }
            for p in r["people"]
        ]
        self._views = {
            "people": people,
            "projects": [
                {
                    "id": str(p["id"]),
                    "name": p["name"],
                    "client": clients.get(p.get("clientId")),
                    "is_archived": bool(p.get("isArchived")),
                    "is_tentative": not p.get("isConfirmed", True),
                }
                for p in r["projects"]
                if not p.get("isTemplate")
            ],
            "assignments": [
                {
                    "id": str(a["id"]),
                    "person_id": str(a["personId"]),
                    "project_id": str(a["projectId"]),
                    "start_date": a["startDate"],
                    "end_date": a["endDate"],
                    "minutes_per_day": a["minutesPerDay"],
                    "is_placeholder": a["personId"] in placeholders,
                    "role": roles.get(a.get("roleId")),
                    "is_billable": a.get("isBillable"),
                }
                for a in r["assignments"]
                if a.get("isActive", True)
                and not a.get("isTemplate")
                and not a.get("isNonWorkingDay")
            ],
            "working_patterns": [
                {
                    "person_id": str(c["personId"]),
                    "start_date": c["startDate"],
                    "end_date": c["endDate"],
                    "minutes_per_day": c["minutesPerDay"],
                    "minutes_by_weekday": c["rosteredDays"],
                }
                for c in r["contracts"]
            ],
            "time_off": [
                {
                    "person_id": str(t["personId"]),
                    "start_date": t["startDate"],
                    "end_date": t["endDate"],
                    "minutes_per_day": t["minutesPerDay"],
                    "type": kind,
                }
                for kind, rows in (
                    ("time_off", r["time_offs_leave"]),
                    ("holiday", r["time_offs_holidays"]),
                )
                for t in rows
            ],
            "source": [
                {
                    "source": "runn",
                    "as_of": f"{self.world['today']}T06:00:00Z",
                    "contract_version": 1,
                }
            ],  # fmt: skip
        }
        self._truth = truth
        return self._views

    def _meta(self, page: dict) -> dict:
        as_of = f"{self.world['today']}T06:00:00Z"
        return {**page, "sources": [{"source": "runn", "asOf": as_of, "contractVersion": 1}]}

    def resourcing_relay(self, segs: list[str], params: dict) -> tuple[int, Any]:
        views = self._resourcing_views()
        if segs == ["views"]:
            data = {"runn": {v: sorted(rows[0]) for v, rows in views.items() if rows}}
            return 200, {"data": data, "meta": self._meta({})}
        if segs == ["capacity"]:
            return self._resourcing_capacity(params)
        if len(segs) == 3 and segs[0] == "legs":
            if segs[1] != "runn" or segs[2] not in views:
                return 404, {"code": "RESOURCING_NOT_FOUND", "message": "no such view"}
            status, body = resourcing_read(views[segs[2]], params)
            return status, ({**body, "meta": self._meta(body["meta"])} if status == 200 else body)
        if len(segs) == 1 and segs[0] in views:
            rows = [{"source": "runn", **row} for row in views[segs[0]]]
            status, body = resourcing_read(rows, params)
            return status, ({**body, "meta": self._meta(body["meta"])} if status == 200 else body)
        return 404, {"code": "RESOURCING_NOT_FOUND", "message": "no such view"}

    def _resourcing_capacity(self, params: dict) -> tuple[int, Any]:
        if not self.capacity:
            return 503, {
                "code": "RESOURCING_CAPACITY_OFF",
                "message": "Resourcing capacity is not served yet in this environment.",
            }
        start, end = params.get("from"), params.get("to")
        if not (start and end and DATE.match(start) and DATE.match(end)) or end < start:
            return 400, {"code": "RESOURCING_BAD_QUERY", "message": "from and to are dates"}
        if (d(end) - d(start)).days + 1 > 100:
            return 400, {"code": "RESOURCING_BAD_QUERY", "message": "longer than 100 days"}
        people = self._resourcing_views()["people"]
        if not any(k.startswith("is_archived") for k in params):
            params = {**params, "is_archived": "false"}
        status, body = resourcing_read(people, {**params, "limit": "1000"}, reserved=("from", "to"))
        if status != 200:
            return status, body
        data = []
        for p in sorted(body["data"], key=lambda p: p["name"]):
            pid = int(p["id"])
            load = self._truth.load(pid, d(start), d(end))
            contracted = sum(
                int((self._truth.contract(pid, day) or {}).get("rosteredDays", {}).get(
                    DAYS[day.weekday()], 0))
                for day in workdays(d(start), d(end))
            )  # fmt: skip
            data.append(
                {
                    "source": "runn",
                    "name": p["name"],
                    "email": p["email"],
                    "team": p["team"],
                    "personId": p["id"],
                    "capacityHours": round(load.capacity / 60, 1),
                    "allocatedHours": round(load.booked / 60, 1),
                    "availableHours": round(load.free / 60, 1),
                    "utilizationPercent": round(100 * load.booked / load.capacity)
                    if load.capacity
                    else None,
                    "timeOffHours": round((contracted - load.capacity) / 60, 1),
                }
            )
        return 200, {"data": data, "meta": self._meta({})}

    # ── Float relay ──
    def float_relay(self, segs: list[str], params: dict) -> tuple[int, Any]:
        fl = self.float
        if segs == ["reports", "people"]:
            for name in params:
                if name not in FLOAT_RANGE | {"people_id"}:
                    return _float_bad(name, "reports/people")
            if not (params.get("start_date") and params.get("end_date")):
                return _float_bad("start_date", "reports/people")
            from .float_world import people_report

            pid = int(params["people_id"]) if params.get("people_id") else None
            rows = people_report(self.world, params["start_date"], params["end_date"], pid)
            return 200, {"values": rows, "pagination": dict.fromkeys(
                ("page", "pageCount", "totalCount", "perPage"))}  # fmt: skip
        if not segs or segs[0] not in FLOAT_PARAMS:
            return 404, {"code": "NOT_FOUND", "message": "not found"}
        resource = segs[0]
        fields = FLOAT_FIELDS[resource]
        if len(segs) == 2 and resource in FLOAT_ID:
            if not INT.match(segs[1]):
                return 404, {"code": "FLOAT_NOT_FOUND", "message": "not found"}
            for row in fl[resource]:
                if row.get(FLOAT_ID[resource]) == int(segs[1]):
                    return 200, {k: row[k] for k in fields if k in row}
            return 404, {"code": "FLOAT_NOT_FOUND", "message": "not found"}
        if len(segs) != 1:
            return 404, {"code": "NOT_FOUND", "message": "not found"}
        for name in params:
            if name not in FLOAT_PARAMS[resource]:
                return _float_bad(name, resource)
        if resource in FLOAT_PAIRED and ("start_date" in params) != ("end_date" in params):
            return _float_bad("end_date", resource)
        rows = list(fl[resource])
        start, end = params.get("start_date"), params.get("end_date")
        if resource == "people":
            if params.get("email"):
                rows = [p for p in rows if (p["email"] or "").lower() == params["email"].lower()]
            if params.get("active"):
                rows = [p for p in rows if str(p["active"]) == params["active"]]
            if params.get("department_id"):
                did = int(params["department_id"])
                rows = [p for p in rows if (p.get("department") or {}).get("department_id") == did]
        elif resource in ("tasks", "timeoffs"):
            if params.get("people_id"):
                pid = int(params["people_id"])
                rows = [
                    t
                    for t in rows
                    if t.get("people_id") == pid or pid in (t.get("people_ids") or [])
                ]
            if params.get("project_id"):
                rows = [t for t in rows if t.get("project_id") == int(params["project_id"])]
            rows = [t for t in rows if _overlaps(t, start, end, "start_date", "end_date")]
        elif resource == "public-holidays" and start:
            rows = [
                {**h, "dates": [x for x in h["dates"] if start <= x <= end]}
                for h in rows
                if any(start <= x <= end for x in h["dates"])
            ]
        per_page = int(params.get("per-page", "50"))
        page_no = int(params.get("page", "1"))
        chunk = rows[(page_no - 1) * per_page : page_no * per_page]
        pages = max(1, -(-len(rows) // per_page))
        return 200, {
            "values": [{k: r[k] for k in fields if k in r} for r in chunk],
            "pagination": {
                "page": page_no,
                "pageCount": pages,
                "totalCount": len(rows),
                "perPage": per_page,
            },
        }

    # ── UB org ──
    def _staff_out(self, s: dict) -> dict:
        keys = ("id", "fullName", "email", "title", "department", "status", "startedAt", "endedAt")
        return {k: s[k] for k in keys}

    def org(self, method: str, segs: list[str], params: dict, body: Any) -> tuple[int, Any]:
        if method == "GET" and segs == ["staff"]:
            rows = self.staff
            status = params.get("status")
            if status and status != "all":
                rows = [s for s in rows if s["status"] == status]
            elif params.get("all") != "true":
                rows = [s for s in rows if s["status"] in ("active", "on_leave")]
            q = params.get("q")
            if q:
                for token in q.lower().split():
                    rows = [
                        s
                        for s in rows
                        if any(
                            token in (s[f] or "").lower()
                            for f in ("fullName", "email", "title", "department")
                        )
                    ]
            if params.get("limit"):
                rows = rows[: int(params["limit"])]
            return 200, [self._staff_out(s) for s in rows]
        if method == "GET" and len(segs) >= 2 and segs[0] == "staff":
            match = next((s for s in self.staff if s["id"] == segs[1]), None)
            if match is None:
                return 404, {"code": "NOT_FOUND", "message": "Staff member not found"}
            if len(segs) == 2:
                return 200, self._staff_out(match)
            if segs[2:] == ["metadata"]:
                rows = [m for m in self.ub["staff_metadata"] if m["staffId"] == match["id"]]
                return 200, rows
        if method == "GET" and segs == ["resourcing"]:
            return self.resourcing(params)
        if method == "GET" and segs == ["offices"]:
            return 200, self.ub["offices"]
        if method == "GET" and len(segs) == 2 and segs[0] == "offices":
            o = next((o for o in self.ub["offices"] if o["id"] == segs[1]), None)
            return (200, o) if o else (404, {"code": "NOT_FOUND", "message": "Office not found"})
        if method == "GET" and segs == ["teams"]:
            return 200, []
        if method == "GET" and segs == ["search"]:
            return 200, self.search(params.get("q", ""))
        if method == "GET" and segs in (["accounts"], ["campaigns"], ["ideas"]):
            return 200, []
        if method == "POST" and segs == ["query"]:
            return self.query(body or {})
        return 404, {"code": "NOT_FOUND", "message": "not found"}

    def resourcing(self, params: dict) -> tuple[int, Any]:
        kind = params.get("type")
        if not kind:
            return 400, {"code": "VALIDATION_ERROR", "message": "?type= is required"}
        rows = [m for m in self.ub["staff_metadata"] if m["type"] == kind]
        if params.get("label"):
            rows = [m for m in rows if params["label"].lower() in m["label"].lower()]
        if params.get("value"):
            rows = [m for m in rows if m["value"] == params["value"]]
        if params.get("featured") in ("true", "false"):
            rows = [m for m in rows if m["isFeatured"] == (params["featured"] == "true")]
        by: dict[str, dict] = {}
        for m in rows:
            s = next(s for s in self.staff if s["id"] == m["staffId"])
            out = by.setdefault(
                s["id"],
                {
                    "staffId": s["id"],
                    "fullName": s["fullName"],
                    "email": s["email"],
                    "title": s["title"],
                    "status": s["status"],
                    "entries": [],
                },
            )
            out["entries"].append(m)
        return 200, sorted(by.values(), key=lambda r: r["fullName"])

    def search(self, q: str) -> list[dict]:
        words = set(re.findall(r"\w+", q.lower()))
        hits = []
        for s in self.staff:
            name_words = set(re.findall(r"\w+", s["fullName"].lower()))
            sim = len(words & name_words) / max(len(words | name_words), 1)
            if sim > 0:
                hits.append(
                    {
                        "type": "staff",
                        "id": s["id"],
                        "name": s["fullName"],
                        "similarity": round(sim, 3),
                        "wholeNameSimilarity": round(sim, 3),
                        "parentId": None,
                    }
                )
        return sorted(hits, key=lambda h: -h["similarity"])[:10]

    # ── /org/query ──
    def query(self, body: dict) -> tuple[int, Any]:
        entity = body.get("entity")
        tables = {"staff": self.staff, "offices": self.ub["offices"], "teams": []}
        if entity not in tables:
            return 200, {"results": [], "total": 0, "truncated": False}
        rows = list(tables[entity])
        fields = set(rows[0]) if rows else set()
        for field, clause in (body.get("filter") or {}).items():
            if field not in fields:
                return 400, {
                    "code": "QUERY_INVALID",
                    "message": f'Unknown field "{field}" on {entity}.',
                }
            if not isinstance(clause, dict) or len(clause) != 1:
                return 400, {"code": "QUERY_INVALID", "message": "One operator per field."}
            ((op, value),) = clause.items()
            rows = [r for r in rows if _match(r.get(field), op, value)]
        group_by = body.get("group_by") or []
        aggregate = body.get("aggregate") or {}
        if group_by or aggregate:
            groups: dict[tuple, list] = defaultdict(list)
            for r in rows:
                groups[tuple(r.get(g) for g in group_by)].append(r)
            out = []
            for key, members in groups.items():
                row = dict(zip(group_by, key, strict=True))
                if "officeId" in group_by and members:
                    row["officeName"] = members[0].get("officeName")
                for name, spec in (aggregate or {"count": {"op": "count"}}).items():
                    if spec.get("op") == "count":
                        row[name] = len(members)
                out.append(row)
            rows = out
        for s in reversed(body.get("sort") or []):
            rows.sort(
                key=lambda r, f=s.get("field"): (r.get(f) is None, r.get(f)),
                reverse=s.get("direction") == "desc",
            )
        total = len(rows)
        limit = min(int(body.get("limit") or 50), 100)
        return 200, {"results": rows[:limit], "total": total, "truncated": total > limit}


def _match(value: Any, op: str, arg: Any) -> bool:
    text = str(value).lower() if value is not None else ""
    if op == "eq":
        return value == arg or (isinstance(arg, str) and text == arg.lower())
    if op == "neq":
        return value is not None and value != arg
    if op == "in":
        return value in (arg or [])
    if op == "like":
        pattern = re.escape(str(arg).lower()).replace("%", ".*").replace("_", ".")
        return re.fullmatch(pattern, text) is not None
    if op == "similar_to":
        words = set(re.findall(r"\w+", str(arg).lower()))
        return bool(words & set(re.findall(r"\w+", text))) or str(arg).lower() in text
    if op == "is_null":
        return (value is None) == bool(arg)
    if op == "between":
        return value is not None and str(arg[0]) <= str(value)[: len(str(arg[1]))] <= str(arg[1])
    if op in ("gt", "gte", "lt", "lte"):
        if value is None:
            return False
        a, b = str(value), str(arg)
        return {"gt": a > b, "gte": a >= b, "lt": a < b, "lte": a <= b}[op]
    return False
