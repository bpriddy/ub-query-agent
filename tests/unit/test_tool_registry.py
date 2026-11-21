"""
tools.ALL_TOOLS — what the executor is offered, pinned.

Nothing asserted the contents of this list until search-01, and until then
nothing needed to: a tool dropped from it would have made answers worse, but
only this repo's answers. `find_files` changed that. gub-gchat-bot reads the
file list straight off the function RESPONSE, keyed on the tool NAME — so a
rename or a removal here empties a section of another repo's UI with no error
raised anywhere, in either repo. That failure deserves a test that fails first.

The set is pinned whole rather than just checking `find_files` is in it: the
point is that changing the offered tools becomes a deliberate edit with a test
diff attached, not a side effect of tidying an import.
"""

from __future__ import annotations

from gub_agent.tools import ALL_TOOLS, find_files

#: Every tool the executor is offered, by the name ADK derives from the
#: function. Adding or removing one is a change to the model's whole
#: capability surface — and, for `find_files`, to another repo's wire.
EXPECTED_TOOLS = {
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
}


def _names() -> list[str]:
    return [tool.__name__ for tool in ALL_TOOLS]


def test_the_offered_tools_are_exactly_these():
    assert set(_names()) == EXPECTED_TOOLS


def test_no_tool_is_registered_twice():
    """ADK builds one declaration per entry; a duplicate would ship the same
    function to the model twice and pay for it on every ReAct round."""
    assert len(_names()) == len(set(_names()))


def test_find_files_is_offered_under_the_name_the_bot_keys_on():
    """The cross-repo contract, at its narrowest: this exact name, on a
    function that is actually in the list the executor is built with."""
    assert find_files in ALL_TOOLS
    assert find_files.__name__ == "find_files"


def test_every_tool_carries_a_docstring():
    """The docstring IS the model-facing tool description — ADK infers it from
    nothing else, so an undocumented tool ships as a bare signature and gets
    called by guesswork."""
    for tool in ALL_TOOLS:
        assert (tool.__doc__ or "").strip(), tool.__name__
