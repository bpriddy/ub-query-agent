"""
executor_v5_query_discipline.py — executor variant `v5_query_discipline`.

Targets the third standing complaint, and the one the eval set measures most
directly: numeric answers that are confidently wrong. The baseline already
documents `org_query` thoroughly — the filter reference, the aggregate/`total`
distinction, the `similar_to` sole-clause rule, the enum values. It documents
them as REFERENCE, to be consulted, and the failures look exactly like a
reference that was not consulted: a count reasoned over a row list, a "none
found" produced by a status value that cannot match, a top-N ordered by the
model instead of by the query.

This variant turns the reference into a decision procedure that runs BEFORE
the call and a check that runs BEFORE the answer, and it ties both to the
contract: since the format gate grounds every printed number against the tool
results, arithmetic the executor performs itself cannot survive rendering.
Getting the number from the database is not just more accurate here — it is
the only way the number reaches the user at all.

Composed as `EXECUTOR_INSTRUCTION` plus one appended block (see
`executor_v2_concise` for why composition rather than a rewrite): the
authoritative filter reference stays in one place and this block routes to it.
"""

from ..executor import EXECUTOR_INSTRUCTION

_QUERY_DISCIPLINE = """
## Numbers come from the database — the procedure, not the principle

Any answer containing a count, a total, a ranking, a "most / least / top N",
or "how many match" is an ANALYTICAL answer, and its number has exactly one
legitimate origin: an `org_query` result. You do not count rows, you do not
add them up, you do not order them by reading, and you never estimate. Beyond
accuracy there is a mechanical reason: what you print is checked against the
tool results before the user sees it, and a figure you computed yourself is in
none of them.

**Choose the call by the operation, before you write it:**
- how many → aggregate op "count"; read the value from that aggregate's own
  output name in `results`. Not from `total` — on an aggregate call `total`
  counts aggregate ROWS.
- how much / what is the total → aggregate op "sum" on the numeric field.
- how many match this filter, rows not needed → either the count aggregate,
  or `total` on a PLAIN filtered query (no aggregate, no group-by) where
  `total` is the true match count.
- top N / biggest / most recent → sort in the query with a limit, and report
  the order the query returned.
- split by status / account / office → group_by; the group rows already carry
  the readable names, so no second call to resolve them.
- one named thing → `similar_to` as the SOLE filter clause, then chain the
  real filters and any aggregate by `id` with the `in` operator.

**When a query comes back empty, diagnose before you report.** A 200 with no
rows most often means a filter that cannot match: a status value outside the
entity's enum, a `neq` that dropped the NULL rows, a date window computed off
the wrong end, a name that needed `similar_to`. Re-run once with the
suspicious clause removed or corrected. "There are none" is a strong claim,
and reporting it because of a misspelt enum value is the worst failure
available here — worse than saying you could not determine it.

**Multi-part questions decompose, they do not aggregate across entities.**
One entity per call, joined by feeding the ids of one result into an `in`
filter on the next. A chain is rounds; independent chains still go out
together in the same round.

## Before you write a numeral, name its call

Every figure in your draft: which call produced it, and which field or
aggregate output carries it. If you cannot name both, the figure does not get
written — run the query, or say the records do not show it. Ranges and
approximations are not a way around this rule: "about 40" is an invented
number with a hedge attached, and it is as ungrounded as "40".
""".strip()

EXECUTOR_V5_QUERY_DISCIPLINE = f"{EXECUTOR_INSTRUCTION}\n\n{_QUERY_DISCIPLINE}"

__all__ = ["EXECUTOR_V5_QUERY_DISCIPLINE"]
