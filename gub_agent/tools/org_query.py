"""
org_query.py — structured query tool for the GUB org data.

A single tool that exposes the GUB query engine (POST /org/query) with
discrete operator modules. Use this for any question that requires
filtering, sorting, counting, grouping, or aggregating across campaigns,
accounts, or staff. Prefer this over fetching lists and reasoning over
them with prose — the LLM is bad at counting and sorting; the database
is good at it.
"""

from __future__ import annotations

from typing import Any

from ._client import gub_post


async def org_query(
    entity: str,
    filter: dict | None = None,
    sort: list[dict] | None = None,
    group_by: list[str] | None = None,
    aggregate: dict | None = None,
    limit: int | None = None,
    tool_context: Any = None,
) -> dict:
    """
    Execute a structured query against GUB org data. THE preferred tool for
    any question involving filtering, counting, sorting, ranking, or
    aggregation — the database does these reliably; do NOT do them yourself.

    ## Entities

    - "campaigns" — agency work for a client account
    - "pieces" — campaign pieces: distinct produced/producing executions within
      a campaign (a commercial, social series, tool, activation). Query these to
      RESOLVE a piece by name (`{name: {similar_to: "..."}}`) or list a campaign's
      pieces (`{campaignId: {eq: "..."}}`); the rich status is NOT here — fetch
      it with get_piece(id). Usually easier: use `find` when you don't yet know
      the thing is a piece.
    - "accounts" — clients of the agency
    - "staff" — people at the agency

    Use a separate call per entity. For multi-entity questions, chain
    queries (see "Composition" below).

    ## Results include human-readable names — do NOT re-resolve ids

    Every FK id in a result row comes with a companion `*Name` field,
    resolved server-side in the same query:

      campaigns: accountId → accountName, createdBy → createdByName
      accounts:  parentId → parentName, ownerStaffId → ownerStaffName,
                 accountExecStaffId → accountExecStaffName
      staff:     officeId → officeName
      pieces:    campaignId → campaignName

    This applies to entity rows AND group_by rows. NEVER spend a follow-up
    query (or a round) turning ids into names — the name is already in the
    row. Chain by id only when you need MORE fields from the related entity
    than its name, or need to filter on it.

    A `*Name` is null in two cases you must NOT conflate: the FK id itself is
    null (no related record), OR you lack access to that record (it exists,
    but its name is withheld by your grants). Treat a null name as "unknown /
    not visible" — fall back to the id or say the name is unavailable; never
    report the related entity as nonexistent on the basis of a null name.

    ## Filter operators

    Each filter is `{field: {op: value}}`. ONE operator per field per call.
    AND across multiple fields. For OR/range, use `in` or `between`.

      eq          {status: {eq: "live"}}                equals
      neq         {status: {neq: "ended"}}              not equals
      in          {status: {in: ["live", "awarded"]}}   value in list
      gt, gte     {budget: {gte: 100000}}               >, >=
      lt, lte     {budget: {lt: 50000}}                 <, <=
      between     {awardedAt: {between: ["2025-01-01","2025-12-31"]}}
      like        {name: {like: "nike"}}                case-insensitive substring; plain
                                                        text, no % or _ wildcards
      similar_to  {name: {similar_to: "chevy"}}         pg_trgm fuzzy match —
                                                        matches Chevrolet, Chevy Trucks, etc.
                                                        USE THIS for fuzzy name lookups.
                                                        Needs at least 3 characters.
                                                        v1 limitation: must be the SOLE
                                                        filter; chain follow-ups by `id.in`.
      is_null     {endsAt: {is_null: true}}             field IS / IS NOT NULL

    Status values are ENTITY-SPECIFIC (a value from the wrong entity silently
    matches nothing): campaigns use `pitch`/`awarded`/`live`/`ended`/`lost`;
    accounts use `active`/`inactive`/`prospect`; staff use `active`/`on_leave`/
    `former`.

    ## Sort

    `[{field: "budget", direction: "desc"}, ...]`. Multi-key allowed.
    May also sort by an aggregate output name (see Aggregate).

    ## Group-by + aggregate (analytics)

    `group_by: ["status"]` groups results by that field. Multi-axis grouping
    is supported: `group_by: ["status", "industry"]` → one row per
    combination.
    `aggregate: {<outputName>: {op: "count"|"sum"|"avg"|"min"|"max", field?: "budget"}}`
    MULTIPLE aggregates in one call are supported — one output name each:
    `aggregate: {n: {op: "count"}, spend: {op: "sum", field: "budget"}}`.

    `count` takes no field. `sum/avg/min/max` require a numeric or date field.
    Group-by without explicit aggregate implicitly counts.

    Result rows shape: `{<group_field(s)>: value, <outputName>: number, ...}`,
    and `sort` may reference any aggregate output name (e.g. sort by `spend`
    desc, limit 5 → top-5 ranking computed in the DB). Grouping by an FK id
    also merges in its `*Name` companion — `group_by: ["accountId"]` rows
    carry both `accountId` and `accountName` (null when the id is null or the
    record isn't visible to you).

    RULE: for portfolio/analytics questions, prefer ONE rich org_query
    (multiple aggregates, multi-axis group_by, sort by aggregate) over many
    small calls.

    CLOSURE: the grouped/aggregated rows ARE the answer to a comparison,
    breakdown, or "how do X vs Y differ" question — synthesize directly from
    them. Do NOT then fetch individual accounts, campaigns, or staff to
    "enrich" a comparison: that re-introduces the per-entity fan-out the
    aggregate just replaced. Drill into a single entity only if the user
    asked about that specific one.

    ## Limit + total

    `limit` caps the returned rows (default 50, max 100). The response includes
    `total` — the REAL DB count of matching rows independent of `limit`. USE
    `total` FOR ANY "HOW MANY" QUESTION. Never count items in the `results`
    array yourself.

    ## Composition for multi-entity questions

    org_query handles ONE entity per call. For multi-entity questions, chain
    queries using `in` as the join primitive:

      "Most expensive campaign for Chevy this year"
      1) org_query(entity="accounts", filter={name: {similar_to: "chevy"}}, limit=5)
         → pick the right accountId from the candidates
      2) org_query(
           entity="campaigns",
           filter={
             accountId: {eq: <that account's id>},
             awardedAt: {between: ["2025-01-01","2025-12-31"]}
           },
           sort=[{field: "budget", direction: "desc"}],
           limit=1
         )

      "Staff who led campaigns over $1M for auto accounts"
      1) accounts where industry=auto → ids A
      2) campaigns where accountId in A and budget > 1M → rows already carry
         createdByName; done if only names are needed
      3) staff where id in (distinct createdBy) — ONLY if you need staff
         fields beyond the name (title, office, status)

    ## Worked examples

      # "How many campaigns for Chevy?"
      org_query(entity="accounts", filter={name: {similar_to: "chevy"}}, limit=5)
      # ... pick Chevrolet's id ...
      org_query(entity="campaigns",
                filter={accountId: {eq: "..."}},
                aggregate={count: {op: "count"}})
      # → results: [{count: 7}], total: 1

      # "Top 5 accounts by campaign count" — top-N by aggregate: rank in the
      # DB by sorting on the aggregate OUTPUT NAME. Extra aggregates ride
      # along in the same call.
      org_query(entity="campaigns",
                group_by=["accountId"],
                aggregate={campaignCount: {op: "count"},
                           totalBudget: {op: "sum", field: "budget"}},
                sort=[{field: "campaignCount", direction: "desc"}],
                limit=5)
      # → [{accountId: "...", campaignCount: 9, totalBudget: 4200000,
      #     accountName: "..."}, ...] — accountName rides along; NO
      #     follow-up accounts query just to resolve names.

      # "Who owns the most accounts?" — same top-N pattern on accounts
      org_query(entity="accounts",
                group_by=["ownerStaffId"],
                aggregate={accountCount: {op: "count"}},
                sort=[{field: "accountCount", direction: "desc"}],
                limit=5)
      # → rows carry ownerStaffName; ONE call, not two. A null name means
      #   the owner is unset or not visible to you — fall back to the id.

      # "How do active vs prospect vs inactive accounts differ?" — one
      # multi-axis call gives the whole comparison grid; do NOT query each
      # status separately.
      org_query(entity="accounts",
                group_by=["status", "industry"],
                aggregate={accounts: {op: "count"}},
                sort=[{field: "accounts", direction: "desc"}])
      # → [{status: "active", industry: "auto", accounts: 4}, ...]

      # "Portfolio rollup: volume, spend, and recency per account" — SEVERAL
      # metrics per group in ONE call
      org_query(entity="campaigns",
                group_by=["accountId", "status"],
                aggregate={n: {op: "count"},
                           spend: {op: "sum", field: "budget"},
                           avgBudget: {op: "avg", field: "budget"},
                           latestEnd: {op: "max", field: "endsAt"}},
                sort=[{field: "spend", direction: "desc"}])
      # → one row per (account, status) with all four metrics attached.

      # "Total budget awarded last year"
      org_query(entity="campaigns",
                filter={awardedAt: {between: ["2025-01-01","2025-12-31"]}},
                aggregate={total: {op: "sum", field: "budget"}})

      # "Live campaigns sorted by budget"
      org_query(entity="campaigns",
                filter={status: {eq: "live"}},
                sort=[{field: "budget", direction: "desc"}],
                limit=20)

    ## What this tool deliberately does NOT do

    - No native joins. Use chained queries with `in` as shown above.
    - No text search over status markdown / notes / metadata. Those are
      conversational context — fetch the specific entity by id with the
      existing detail tools (get_campaign / get_account_overview) and read.
    - No staff metadata (skills, certifications). Use
      `find_staff_for_resourcing` for resourcing questions.

    Args:
        entity: One of "campaigns", "accounts", "staff".
        filter: Map of field -> {operator: value}. Optional.
        sort: List of {field, direction: "asc"|"desc"}. Optional.
        group_by: List of fields to group rows by. Optional.
        aggregate: Map of output-name -> {op, field?}. Optional.
        limit: Max rows to return (default 50, max 100). Optional.

    Returns:
        {results: [...], total: int, truncated: bool}
        - results: the row list (entity rows or aggregate/group rows).
        - total: the REAL count of matching rows in the DB, NOT len(results).
        - truncated: True if limit cut off some rows.
    """
    body: dict = {"entity": entity}
    if filter is not None:
        body["filter"] = filter
    if sort is not None:
        body["sort"] = sort
    if group_by is not None:
        body["group_by"] = group_by
    if aggregate is not None:
        body["aggregate"] = aggregate
    if limit is not None:
        body["limit"] = limit
    return await gub_post("/org/query", body, tool_context)
