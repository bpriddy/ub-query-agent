"""
GUB agent tool functions.

All tools are plain Python functions — ADK infers their descriptions from
docstrings and their parameter schemas from type hints.

Import pattern in agent.py:
    from .tools import ALL_TOOLS
    agent = Agent(..., tools=ALL_TOOLS)
"""

from .accounts import get_account_overview, get_campaign, list_accounts
from .discovery import find, get_idea, get_piece, list_ideas
from .files import find_files
from .org_query import org_query
from .similarity import similar_ideas, similar_to_idea
from .staff import find_staff_for_resourcing, get_staff_profile, search_staff

ALL_TOOLS = [
    # Discovery — resolve a named thing to a typed id when you don't know what it is
    find,
    # Drive file NAME search (search-01). In the list unconditionally, but
    # offered to the model only while `FILE_SEARCH_ENABLED` is on AND the
    # router classified the turn `file_lookup` — `agents/tool_gate.py`
    # withholds the declaration otherwise, so with the flag off (the default)
    # it is never offered at all.
    find_files,
    # Structured query primitive — preferred for filter/sort/count/aggregate
    org_query,
    # Staff & resourcing
    find_staff_for_resourcing,
    get_staff_profile,
    search_staff,
    # Accounts & campaigns (detail tools)
    list_accounts,
    get_account_overview,
    get_campaign,
    # Pieces & ideas (detail tools)
    get_piece,
    list_ideas,
    get_idea,
    # Idea similarity by meaning (UB D40): weighted facets the model writes
    similar_ideas,
    similar_to_idea,
]

__all__ = [
    "ALL_TOOLS",
    "find",
    "find_files",
    "org_query",
    "find_staff_for_resourcing",
    "get_staff_profile",
    "search_staff",
    "list_accounts",
    "get_account_overview",
    "get_campaign",
    "get_piece",
    "list_ideas",
    "get_idea",
    "similar_ideas",
    "similar_to_idea",
]
