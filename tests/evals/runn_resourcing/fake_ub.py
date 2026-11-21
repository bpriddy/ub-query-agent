"""
fake_ub.py — a localhost UB serving the eval's fixtures, shaped like UB.

The Runn half is TODAY's relay (gcp-universal-backend
src/modules/relay/runn/): only

    GET /relay/runn/people        GET /relay/runn/people/:id
    GET /relay/runn/projects      GET /relay/runn/projects/:id
    GET /relay/runn/assignments

with the relay's per-resource field allowlist (`runn.client.ts` FIELDS: no
custom fields, notes or budgets), its query-parameter allowlist (anything
else is a 400 RUNN_BAD_QUERY), `includeArchived=true` by default on projects,
placeholders left out of /people unless `includePlaceholders=true`, and
Runn's `{values, nextCursor}` pages. Contracts, roles, skills, teams, tags,
time off and holidays exist in the fixtures but are NOT served, exactly as
the relay does not serve them. The assignment date filters match on overlap.

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

RUNN_FIELDS = {
    "people": [
        "id",
        "firstName",
        "lastName",
        "email",
        "isArchived",
        "teamId",
        "tags",
        "skills",
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
}
LIST_PARAMS = {"cursor", "limit", "modifiedAfter", "sortBy", "order"}
RUNN_PARAMS = {
    "people": LIST_PARAMS | {"includePlaceholders"},
    "projects": LIST_PARAMS | {"includeArchived"},
    "assignments": LIST_PARAMS | {"personId", "projectId", "roleId", "startDate", "endDate"},
}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
INT = re.compile(r"^[1-9]\d*$")


def _pick(resource: str, row: dict) -> dict:
    return {k: row[k] for k in RUNN_FIELDS[resource] if k in row}


def _bad(name: str, resource: str) -> tuple[int, dict]:
    return 400, {
        "code": "RUNN_BAD_QUERY",
        "message": f'Runn {resource} does not take the "{name}" parameter here.',
    }


class FakeUb:
    def __init__(self, world: dict) -> None:
        self.runn = world["runn"]
        self.ub = world["ub"]
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
            return self.relay(segs[2:], params)
        if segs[:1] == ["org"]:
            return self.org(method, segs[1:], params, body)
        return 404, {"code": "NOT_FOUND", "message": "not found"}

    # ── Runn relay ──
    def relay(self, segs: list[str], params: dict) -> tuple[int, Any]:
        if len(segs) == 2 and segs[0] in ("people", "projects"):
            resource, rid = segs
            if not INT.match(rid):
                return 404, {"code": "RUNN_NOT_FOUND", "message": f"No Runn {resource} {rid}."}
            rows = self.runn[resource] + (self.runn["placeholders"] if resource == "people" else [])
            for row in rows:
                if row["id"] == int(rid):
                    return 200, _pick(resource, row)
            return 404, {"code": "RUNN_NOT_FOUND", "message": f"No Runn {resource} {rid}."}
        if len(segs) != 1 or segs[0] not in RUNN_PARAMS:
            return 404, {"code": "NOT_FOUND", "message": "not found"}
        resource = segs[0]
        for name, value in params.items():
            if name not in RUNN_PARAMS[resource]:
                return _bad(name, resource)
            if name in ("startDate", "endDate") and not DATE.match(value):
                return _bad(name, resource)
            if name in ("personId", "projectId", "roleId") and not INT.match(value):
                return _bad(name, resource)
            if name == "limit" and not (INT.match(value) and int(value) <= 200):
                return _bad(name, resource)
        if resource == "people":
            rows = list(self.runn["people"])
            if params.get("includePlaceholders") == "true":
                rows += self.runn["placeholders"]
        elif resource == "projects":
            rows = [
                p
                for p in self.runn["projects"]
                if params.get("includeArchived", "true") == "true" or not p["isArchived"]
            ]
        else:
            rows = self.runn["assignments"]
            for key in ("personId", "projectId", "roleId"):
                if key in params:
                    rows = [a for a in rows if a[key] == int(params[key])]
            if "startDate" in params:
                rows = [a for a in rows if a["endDate"] >= params["startDate"]]
            if "endDate" in params:
                rows = [a for a in rows if a["startDate"] <= params["endDate"]]
        limit = int(params.get("limit", "50" if resource != "assignments" else "100"))
        offset = int(params.get("cursor", "0") or 0)
        page = rows[offset : offset + limit]
        more = offset + limit < len(rows)
        return 200, {
            "values": [_pick(resource, r) for r in page],
            "nextCursor": str(offset + limit) if more else None,
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
