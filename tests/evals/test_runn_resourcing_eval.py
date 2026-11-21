"""
The Runn resourcing eval's hermetic half (tests/evals/runn_resourcing/).

The live run (run_eval.py) costs tokens and never runs here. What does:
  * the fixtures on disk are exactly what world.py builds, in Runn's shapes;
  * the ground truth says what the fixtures were built to say;
  * the question set is the size the eval promises, and every name it
    grades on exists;
  * the fake UB serves today's relay and only today's relay, and the real
    `runn_allocations` reads it.
"""

from __future__ import annotations

import json

import pytest

from gub_agent.tools import _client
from gub_agent.tools.runn import runn_allocations
from tests.evals.runn_resourcing import questions, world
from tests.evals.runn_resourcing.fake_ub import FakeUb
from tests.evals.runn_resourcing.truth import Truth, d


@pytest.fixture(scope="module")
def data() -> dict:
    return world.load()


@pytest.fixture(scope="module")
def truth(data) -> Truth:
    return Truth(data)


@pytest.fixture
async def fake(data, monkeypatch):
    server = FakeUb(data)
    await server.start()
    monkeypatch.setattr(_client, "GUB_BASE_URL", server.url)
    monkeypatch.setattr(_client, "GUB_SERVICE_JWT", "eval-test")
    yield server
    await _client.aclose_client()
    await server.stop()


def test_the_fixtures_on_disk_are_what_world_builds(data):
    built = world.build()
    assert json.loads(json.dumps(built["runn"])) == data["runn"]
    assert json.loads(json.dumps(built["ub"])) == data["ub"]


def test_the_fixtures_hold_together(data):
    runn = data["runn"]
    people = {p["id"] for p in runn["people"]} | {p["id"] for p in runn["placeholders"]}
    projects = {p["id"] for p in runn["projects"]}
    roles = {r["id"] for r in runn["roles"]}
    assert len(runn["people"]) == 40
    for a in runn["assignments"]:
        assert a["personId"] in people and a["projectId"] in projects and a["roleId"] in roles
        assert a["startDate"] <= a["endDate"]
    for c in runn["contracts"]:
        assert c["personId"] in people and c["roleId"] in roles
        assert set(c["rosteredDays"]) == {"monday", "tuesday", "wednesday", "thursday", "friday"}
    # Every UB office is used, and all but two UB staff have a Runn person.
    emails = {p["email"] for p in runn["people"]}
    ub_staff = data["ub"]["staff"]
    assert {s["officeId"] for s in ub_staff} == {o["id"] for o in data["ub"]["offices"]}
    assert len([s for s in ub_staff if s["email"] not in emails]) == 3  # 2 UB-only + Harry


def test_the_truth_says_what_the_fixtures_were_built_to_say(truth):
    la_ads = truth.select(office="LA", roles=("Art Director",))
    week = (d("2026-11-23"), d("2026-11-27"))
    avail = truth.availability(la_ads, *week)
    assert avail["core"] == ["Marcus Bell", "Sofia Reyes"]
    assert avail["acceptable"] == ["Priya Nair"]  # tentative work only
    assert set(avail["forbidden"]) == {"Jane Park", "Theo Grant"}
    november = (d("2026-11-01"), d("2026-11-30"))
    assert truth.over_capacity(truth.select(office="NYC"), *november)["core"] == [
        "Maya Singh",
        "Sam Rivera",
    ]
    # Thanksgiving: 3 working days next week in LA, 5 in London.
    assert truth.load(2, *week).capacity == 3 * 480
    assert truth.load(32, *week).capacity == 5 * 450
    # Sofia Reyes works Mon-Thu: 4 h a day is 16 h a week.
    assert truth.load(3, d("2026-11-16"), d("2026-11-20")).confirmed == 16 * 60
    assert truth.understaffed(d("2026-11-18"), d("2027-03-31")) == [
        "Lumen Bank - Brand Refresh",
        "Northwind Air - Q1 Retainer",
        "Solace Health - Pitch",
    ]


def test_the_question_set(truth):
    qs = questions.build(truth)
    assert 25 <= len(qs) <= 40
    assert len({q["id"] for q in qs}) == len(qs)
    names = {truth.name(p) for p in truth.people} | {p["name"] for p in truth.runn["projects"]}
    for q in qs:
        assert q["kind"] in ("names", "value", "clarify")
        assert q["today"] in ("yes", "partial", "no")
        if q["kind"] == "names":
            graded = q["truth"]["core"] + q["truth"]["acceptable"] + q["truth"]["forbidden"]
            assert q["truth"]["core"], q["id"]
            assert set(graded) <= names, q["id"]
        else:
            assert q["truth"]["expected"], q["id"]
    # Follow-ups share a session with the turn they follow.
    chains = [q["chain"] for q in qs]
    assert chains.count("la-ad") == chains.count("jane") == chains.count("over") == 2


async def test_the_fake_serves_todays_relay_and_the_tool_reads_it(fake):
    result = await runn_allocations(
        person_email="sofia.reyes@anomaly.com", start_date="2026-11-23", end_date="2026-11-27"
    )
    assert result["allocations"] == [
        {
            "person": "Sofia Reyes",
            "email": "sofia.reyes@anomaly.com",
            "project": "Vela Motors - EV Launch",
            "start_date": "2026-10-05",
            "end_date": "2027-01-29",
            "hours_per_day": 4.0,
            "billable": True,
        }
    ]
    # Harry Evans's Runn email is not UB's: the tool cannot find him by UB's.
    missing = await runn_allocations(person_email="harry.evans@anomaly.com")
    assert missing["allocations"] == [] and "no person" in missing["note"]


async def test_the_fake_serves_nothing_the_relay_does_not(fake):
    status, page = fake.route("GET", "/relay/runn/people", {"limit": "200"}, None)
    assert status == 200 and len(page["values"]) == 40  # placeholders left out
    assert "customFields" not in page["values"][0] and "holidaysGroupId" not in page["values"][0]
    for path in ("contracts", "roles", "skills", "teams", "people-tags", "time-offs/leave"):
        assert fake.route("GET", f"/relay/runn/{path}", {}, None)[0] == 404
    bad = fake.route("GET", "/relay/runn/people", {"email": "x@y"}, None)
    assert bad == (400, bad[1]) and bad[1]["code"] == "RUNN_BAD_QUERY"
    # A placeholder is readable by id, as Runn's GET /people/{id} is.
    assert fake.route("GET", "/relay/runn/people/901", {}, None)[1]["firstName"] == "Designer"
