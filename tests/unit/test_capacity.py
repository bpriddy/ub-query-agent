"""
capacity.py — the arithmetic and the matching shared by runn_capacity and
float_capacity. The end-to-end check against the ground truth is in
tests/evals/test_runn_resourcing_eval.py; this pins the pieces.
"""

from __future__ import annotations

from datetime import date

import pytest

from gub_agent.tools.capacity import (
    Booking,
    Directory,
    Person,
    Scope,
    TimeOff,
    Window,
    capacity_answer,
    load_of,
    parse_window,
    term_matches,
    working_days,
)

WEEK = Window(date(2026, 11, 23), date(2026, 11, 27))  # Mon-Fri


def _week(*minutes: float):
    days = [*minutes, *([0] * (7 - len(minutes)))]
    return lambda day: days[day.weekday()]


def _person(**over) -> Person:
    base = {
        "key": 1,
        "name": "Ada Lovelace",
        "email": "ada@anomaly.com",
        "pattern": _week(*[480] * 5),
    }
    base.update(over)
    return Person(**base)


@pytest.mark.parametrize(
    ("query", "text"),
    [
        ("strategy", "Strategist"),
        ("art direction", "Art Director"),
        ("design", "Senior Designer"),
        ("copy", "Copywriter"),
        ("motion", "Motion Design"),
    ],
)
def test_the_words_people_use_match_the_names_planners_hold(query, text):
    assert term_matches(query, text)


@pytest.mark.parametrize(
    ("query", "text"),
    [("art director", "Executive Creative Director"), ("strategy", "Producer"), ("", "Designer")],
)
def test_and_nothing_looser(query, text):
    assert not term_matches(query, text)


def test_an_office_by_its_name_city_alias_or_initials():
    d = Directory(
        offices=[
            {"id": "la", "name": "Los Angeles"},
            {"id": "ny", "name": "New York", "syncCity": "New York"},
            {"id": "ldn", "name": "London"},
        ],
        staff=[],
    )
    for words, office in [
        ("LA", "la"),
        ("los angeles", "la"),
        ("NYC", "ny"),
        ("NY", "ny"),
        ("the London office", "ldn"),
    ]:
        assert d.office_named(words)["id"] == office, words
    assert d.office_named("Paris") is None


def test_working_days_in_words():
    assert working_days(_person(), WEEK.start) == "Mon-Fri 8h"
    assert working_days(_person(pattern=_week(480, 480, 480, 480)), WEEK.start) == "Mon-Thu 8h"
    assert working_days(_person(pattern=_week(240, 0, 240)), WEEK.start) == "Mon,Wed 4h"


def test_holidays_and_leave_cut_capacity_and_work_on_a_day_off_does_not_count():
    person = _person(
        time_off=[
            TimeOff(date(2026, 11, 26), date(2026, 11, 26), "holiday", None, "Thanksgiving"),
            TimeOff(date(2026, 11, 27), date(2026, 11, 27), "leave"),
            TimeOff(date(2026, 11, 23), date(2026, 11, 23), "leave", 240),  # half a day
        ],
        bookings=[
            Booking(date(2026, 11, 1), date(2026, 12, 31), 240, True, "Halcyon"),
            Booking(date(2026, 11, 25), date(2026, 11, 27), 240, False, "Pitch"),
        ],
    )
    load = load_of(person, WEEK)
    assert load.capacity == 240 + 480 + 480  # Mon half, Tue, Wed; Thu holiday, Fri leave
    assert load.confirmed == 3 * 240  # Mon, Tue, Wed only
    assert load.tentative == 240  # Wed only
    assert load.holidays == [(date(2026, 11, 26), "Thanksgiving")]


def test_a_reported_capacity_wins_over_the_computed_one():
    assert load_of(_person(reported_capacity=600), WEEK).capacity == 600


def test_parse_window_refuses_bad_or_huge_ranges():
    assert parse_window("2026-11-27", "2026-11-23")["error"] is True
    assert parse_window("2026-01-01", "2026-12-31")["error"] is True
    assert parse_window("next week", "2026-11-27")["error"] is True
    assert parse_window("2026-11-23", "2026-11-27") == WEEK


def test_the_answer_joins_ub_flags_name_matches_and_lists_no_record_staff_apart():
    directory = Directory(
        offices=[{"id": "la", "name": "Los Angeles"}],
        staff=[
            {"id": "s1", "fullName": "Ada Lovelace", "email": "ada@anomaly.com",
             "officeName": "Los Angeles", "title": "Strategist"},
            {"id": "s2", "fullName": "Alan Turing", "email": "alan@anomaly.com",
             "officeName": "Los Angeles", "title": "Strategist"},
            {"id": "s3", "fullName": "Kai Nakamura", "email": "kai@anomaly.com",
             "officeName": "Los Angeles", "title": "Strategist"},
        ],
    )  # fmt: skip
    people = [
        _person(role="Strategist", bookings=[Booking(WEEK.start, WEEK.end, 600, True, "X")]),
        _person(key=2, name="Alan Turing", email="alan@other.co", role="Strategist"),
    ]
    out = capacity_answer(
        source="Runn",
        people=people,
        directory=directory,
        window=WEEK,
        scope=Scope(office={"id": "la", "name": "Los Angeles"}, role="strategy"),
        calendars={},
        warnings=[],
    )
    assert [r["person"] for r in out["people"]] == ["Alan Turing", "Ada Lovelace"]
    alan, ada = out["people"]
    assert alan["office_from"] == "ub_name_match" and alan["free_h"] == 40.0
    assert "over_capacity" in ada["flags"] and ada["free_h"] == 0.0
    assert out["totals"] == [
        {
            "people": 2,
            "capacity_h": 80.0,
            "confirmed_h": 50.0,
            "tentative_h": 0.0,
            "free_h": 40.0,
            "confirmed_pct": 62,
            "booked_pct": 62,
            "over_capacity": 1,
        }
    ]
    assert [p["person"] for p in out["no_record"]["people"]] == ["Kai Nakamura"]
    assert "NOT free" in out["no_record"]["note"]
    assert any("Alan Turing" in w for w in out["warnings"])
    assert all("_minutes" not in r for r in out["people"])
