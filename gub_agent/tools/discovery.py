"""
discovery.py — name discovery + piece/idea tools for the GUB agent.

Tools:
  find        — cross-entity fuzzy name search (what IS this named thing?)
  get_piece   — a campaign piece + its surrounding campaign
  list_ideas  — search the ideas memory (pitched creative concepts)
  get_idea    — a single idea by id

Concept search by meaning (`similar_ideas`, `similar_to_idea`) lives in
`similarity.py`.
"""

from __future__ import annotations

from typing import Any

from ._client import gub_get


async def find(query: str, tool_context: Any = None) -> dict:
    """
    Discover WHAT a named thing is by fuzzy-matching it across every kind of org
    entity at once — accounts, campaigns, campaign pieces, ideas, and staff —
    ranked by similarity.

    USE THIS FIRST whenever the user names something specific and you don't
    already know its type or id. You usually CAN'T tell from the name alone
    whether "BHAC Character Generator" is a campaign, a piece, or an idea — so
    search for it, read the top typed hit(s), then fetch detail with the tool
    that matches the winning `type`:
      type "account"  → get_account_overview(id)
      type "campaign" → get_campaign(id)
      type "piece"    → get_piece(id)
      type "idea"     → get_idea(id)
      type "staff"    → get_staff_profile(id)

    Examples:
    - "how's the BHAC Character Generator?" → find(query="BHAC Character Generator")
    - "what's the latest on Super Cruise?"  → find(query="Super Cruise")

    Args:
        query: The name or phrase the user mentioned.

    Returns:
        A list of hits, each { type, id, name, similarity, parentId }, ranked by
        similarity across all entity types. `type` tells you which detail tool to
        call; `parentId` is the account id (for a campaign) or the campaign id
        (for a piece). An empty list means nothing matched — say so, don't guess.
    """
    return await gub_get("/org/search", tool_context, q=query)


async def get_piece(piece_id: str, tool_context: Any = None) -> dict:
    """
    Get a campaign piece — a distinct thing the campaign produced or is producing
    (a commercial, social series, tool, activation) — bundled with its
    SURROUNDING CAMPAIGN.

    Asking about a piece pulls in the campaign it lives in automatically, so you
    can answer with the piece's own status AND its campaign context. Resolve a
    piece id first with `find` (or org_query entity='pieces'), or pick it from a
    campaign's piece-stub list. For campaign assessments, fetch the RELEVANT
    pieces (usually the most recent few from the campaign's stub list) with
    parallel get_piece calls in one round.

    Args:
        piece_id: The UUID of the piece (from find / org_query / a campaign's
            piece stubs).

    Returns:
        dict { piece, campaign }. `piece.statusMarkdown` is the piece's full
        status; `campaign` is the surrounding campaign dossier (no sibling
        pieces). Treat statusMarkdown as authoritative status prose.
    """
    return await gub_get(f"/org/pieces/{piece_id}", tool_context)


async def list_ideas(
    account_id: str | None = None,
    campaign_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 50,
    tool_context: Any = None,
) -> dict:
    """
    Search the IDEAS memory — the agency's institutional record of pitched
    creative CONCEPTS (mined only from pitch and creative-review decks). Each
    idea is a concept described by `facets` (natural-language rows), with a
    `pitchedAt` date and an `awarded` flag (true once it was produced).

    Use this for STRUCTURED listing: the ideas of an account, a campaign or a
    period, most recently pitched first. For a CONCEPT question ("have we
    already pitched something like X?") call `similar_ideas` first: it searches
    the whole memory by meaning. Come here for a concept question only when
    `similar_ideas` says similarity search is not available; then YOU do the
    concept matching: call this with whatever structured filters narrow the
    field, read the returned `facets` and pick the ideas that match the concept
    the user meant BY MEANING.

    All filters are OPTIONAL and combine:
    - "what have we pitched Chevy this year?" → account_id=<chevy>, since="2026-01-01"
    - "ideas on the Silverado campaign" → campaign_id=<silverado>
    - concept fallback ("have we pitched a loyalty program before?" when
      similarity search is not available) → no filters; match by meaning

    Args:
        account_id: UUID of an account to scope to (resolve via find/list_accounts).
        campaign_id: UUID of a campaign to scope to.
        since: ISO date (YYYY-MM-DD) lower bound on pitchedAt.
        until: ISO date (YYYY-MM-DD) upper bound on pitchedAt.
        limit: Max ideas to return (default 50).

    Returns:
        A list of ideas, each { id, name, facets, pitchedAt, accountName,
        campaignName, awarded }. Requires the user to hold ideas access; without
        it the list is empty (not an error).
    """
    return await gub_get(
        "/org/ideas",
        tool_context,
        accountId=account_id,
        campaignId=campaign_id,
        since=since,
        until=until,
        limit=limit,
    )


async def get_idea(idea_id: str, tool_context: Any = None) -> dict:
    """
    Get a single idea (a pitched creative concept) by id — the detail path after
    `find` resolves a name to an idea.

    Args:
        idea_id: The UUID of the idea (from find).

    Returns:
        dict { id, name, facets, pitchedAt, accountName, campaignName, awarded }.
        `facets` are the concept's description/arc. Empty when you lack ideas access.
    """
    return await gub_get(f"/org/ideas/{idea_id}", tool_context)
