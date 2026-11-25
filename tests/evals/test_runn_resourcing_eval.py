"""
The Runn resourcing eval's hermetic half (tests/evals/runn_resourcing/).

The live run (run_eval.py) costs tokens and never runs here. What does:
  * the fixtures on disk are exactly what world.py builds, in Runn's shapes;
  * the ground truth says what the fixtures were built to say;
  * the question set is the size the eval promises, and every name it
    grades on exists;
  * the fake UB serves the relays (resourcing's over a Runn leg, Runn's, and
    Float's when asked), and the real `resourcing_allocations` reads it;
  * `resourcing_capacity` joins UB's offices and filters by role, team and
    person, and `float_capacity` agrees with the ground truth for every
    person over several windows. (Resourcing computes Runn's capacity; the
    fake serves the ground truth's figures for it.)
"""

from __future__ import annotations

import json

import pytest

from gub_agent.tools import _client
from gub_agent.tools.float import float_capacity
from gub_agent.tools.resourcing import resourcing_allocations, resourcing_capacity
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
    assert json.loads(json.dumps(built["float"])) == data["float"]


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


async def test_the_fake_serves_the_relay_and_the_tool_reads_it(fake):
    result = await resourcing_allocations(
        person_email="sofia.reyes@anomaly.com", start_date="2026-11-23", end_date="2026-11-27"
    )
    assert result["allocations"] == [
        {
            "person": "Sofia Reyes",
            "email": "sofia.reyes@anomaly.com",
            "project": "Vela Motors - EV Launch",
            "client": "Vela Motors",
            "start_date": "2026-10-05",
            "end_date": "2027-01-29",
            "hours_per_day": 4.0,
            "billable": True,
            "status": "confirmed",
            "role": "Art Director",
            "source": "runn",
            "works": "Mon-Thu 8h",
        }
    ]
    assert result["sources"] == [
        {"source": "runn", "as_of": "2026-11-18T06:00:00Z", "contract_version": 1}
    ]
    # Harry Evans's Runn email is not UB's: found by the name it spells, and said so.
    harry = await resourcing_allocations(person_email="harry.evans@anomaly.com")
    assert harry["allocations"][0]["email"] == "harry.evans@anomaly.co.uk"
    assert "by name" in harry["notes"][0]
    # Jack Turner's leave comes with his allocations: "when is he back?"
    jack = await resourcing_allocations(
        person_email="jack.turner@anomaly.com", start_date="2026-11-18"
    )
    assert jack["leave"][0]["end_date"] == "2026-12-31"
    # A project's people and hours a day are summed in code.
    vela = await resourcing_allocations(
        project_name="Vela Motors - EV Launch", start_date="2026-11-18"
    )
    assert vela["summary"][0]["people"] == 6
    # Kai Nakamura has no Runn record: the note says that is not "free".
    kai = await resourcing_allocations(person_email="kai.nakamura@anomaly.com")
    assert kai["allocations"] == [] and "do not call them free" in kai["note"]
    # A date range alone lists every booking in it, placeholders marked.
    window = await resourcing_allocations(start_date="2026-11-23", end_date="2026-12-04")
    open_roles = {a["project"] for a in window["allocations"] if a.get("placeholder")}
    assert open_roles == {"Lumen Bank - Brand Refresh", "Solace Health - Pitch"}


def test_the_fake_serves_the_resourcing_relay(fake):
    status, page = fake.route("GET", "/relay/resourcing/people", {"limit": "1000"}, None)
    assert status == 200 and len(page["data"]) == 40  # placeholders are not people
    assert page["data"][0]["source"] == "runn" and page["meta"]["nextOffset"] is None
    assert page["meta"]["sources"][0]["asOf"] == "2026-11-18T06:00:00Z"
    paged = fake.route("GET", "/relay/resourcing/people", {"limit": "30"}, None)[1]
    assert paged["meta"]["nextOffset"] == 30
    leg = fake.route("GET", "/relay/resourcing/legs/runn/projects", {}, None)[1]["data"]
    assert "source" not in leg[0] and {"is_tentative", "client"} <= set(leg[0])
    bad = fake.route("GET", "/relay/resourcing/people", {"salary.gt": "1"}, None)
    assert bad[0] == 400 and bad[1]["code"] == "RESOURCING_BAD_QUERY"
    assert fake.route("GET", "/relay/resourcing/records", {}, None)[0] == 404
    off = FakeUb(fake.world, capacity=False)
    assert off.route("GET", "/relay/resourcing/capacity", {}, None)[1]["code"] == (
        "RESOURCING_CAPACITY_OFF"
    )


async def test_the_fake_serves_what_the_relay_serves(fake):
    status, page = fake.route("GET", "/relay/runn/people", {"limit": "200"}, None)
    assert status == 200 and len(page["values"]) == 40  # placeholders left out
    # Runn's list of people carries no skills or custom fields; one person does.
    assert "customFields" not in page["values"][0] and "skills" not in page["values"][0]
    assert page["values"][0]["holidaysGroupId"] == 701
    one = fake.route("GET", "/relay/runn/people/1", {}, None)[1]
    assert one["customFields"] == {
        "select": [{"id": 1, "name": "Office", "values": [{"id": 601, "name": "LA"}]}]
    }
    assert {s["id"] for s in one["skills"]} == {501, 505}
    for path in (
        "contracts",
        "roles",
        "skills",
        "teams",
        "people-tags",
        "time-offs/leave",
        "time-offs/rostered-off",
        "time-offs/holidays",
        "holiday-groups",
        "custom-fields/select",
    ):
        assert fake.route("GET", f"/relay/runn/{path}", {}, None)[0] == 200, path
    assert fake.route("GET", "/relay/runn/holiday-groups/701/holidays", {}, None)[0] == 200
    skill_people = fake.route("GET", "/relay/runn/skills/503/people", {}, None)[1]["values"]
    assert {p["firstName"] for p in skill_people} >= {"Ava", "Leo", "Lucas"}
    contract = fake.route("GET", "/relay/runn/contracts", {"limit": "1"}, None)[1]["values"][0]
    assert "costPerHour" not in contract and "jobTitle" not in contract
    leave = fake.route("GET", "/relay/runn/time-offs/leave", {}, None)[1]["values"][0]
    assert "note" not in leave
    for path in ("clients", "custom-fields/text", "assignments/1", "people/1/contracts"):
        assert fake.route("GET", f"/relay/runn/{path}", {}, None)[0] == 404, path
    bad = fake.route("GET", "/relay/runn/people", {"email": "x@y"}, None)
    assert bad == (400, bad[1]) and bad[1]["code"] == "RUNN_BAD_QUERY"
    # A placeholder is readable by id, as Runn's GET /people/{id} is.
    assert fake.route("GET", "/relay/runn/people/901", {}, None)[1]["firstName"] == "Designer"
    # Float is not configured unless the run asks for it.
    assert fake.route("GET", "/relay/float/people", {}, None)[0] == 503


WINDOWS = [
    ("2026-11-23", "2026-11-27"),  # Thanksgiving week
    ("2026-11-20", "2026-11-20"),  # one Friday
    ("2026-11-01", "2026-11-30"),
    ("2026-12-01", "2026-12-31"),  # Christmas, leave, contracts ending
    ("2027-01-01", "2027-03-31"),  # Q1
]
OFFICE = {"LA": "Los Angeles", "NYC": "New York", "London": "London"}


async def test_float_capacity_agrees_with_the_ground_truth(data, truth):
    """float_capacity, through the fake Float relay, gives every person exactly
    the capacity, confirmed and tentative hours, and office, that truth.py
    computes from Runn's own records (the same agency in Float's shapes)."""
    server = FakeUb(data, sources=("float",))
    await server.start()
    _client.GUB_BASE_URL, _client.GUB_SERVICE_JWT = server.url, "eval-test"
    try:
        for start, end in WINDOWS:
            result = await float_capacity(start, end)
            rows = {r["person"]: r for r in result["people"]}
            assert len(rows) == 40, start
            for pid in truth.people:
                load = truth.load(pid, d(start), d(end))
                row = rows[truth.name(pid)]
                got = (row["capacity_h"], row["confirmed_h"], row["tentative_h"], row["office"])
                want = (
                    round(load.capacity / 60, 1),
                    round(load.confirmed / 60, 1),
                    round(load.tentative / 60, 1),
                    OFFICE[truth.office(pid)],
                )
                assert got == want, (start, truth.name(pid))
            # UB staff with no planner record are listed apart, never as free.
            assert {p["person"] for p in result["no_record"]["people"]} == {
                "Kai Nakamura",
                "Rosa Diaz",
            }
    finally:
        await _client.aclose_client()
        await server.stop()


@pytest.mark.parametrize("office", sorted(OFFICE))
async def test_resourcing_capacity_places_everyone_in_their_ub_office(fake, truth, office):
    """Office comes from UB's directory joined to the planner's people, by email
    then by name: each office holds exactly the people truth.py puts there."""
    result = await resourcing_capacity("2026-11-23", "2026-11-27", office=office)
    want = {truth.name(p) for p in truth.select(office=office)}
    assert {r["person"] for r in result["people"]} == want
    for r in result["people"]:
        pid = next(p for p in truth.people if truth.name(p) == r["person"])
        load = truth.load(pid, d("2026-11-23"), d("2026-11-27"))
        assert r["capacity_h"] == round(load.capacity / 60, 1)
        assert r["allocated_h"] == round(load.booked / 60, 1)
    if office != "London":
        assert {p["person"] for p in result["no_record"]["people"]} <= {
            "Kai Nakamura",
            "Rosa Diaz",
        }


async def test_capacity_filters_by_office_role_team_and_person(fake):
    week = ("2026-11-23", "2026-11-27")
    la_ads = await resourcing_capacity(*week, office="LA", role="art direction")
    assert [r["person"] for r in la_ads["people"]] == [
        "Marcus Bell",
        "Sofia Reyes",
        "Jane Park",
        "Priya Nair",
        "Theo Grant",
    ]
    assert la_ads["days"] == 5
    assert la_ads["lists"][0]["no_capacity"] == "Theo Grant"  # on leave all week
    # Thanksgiving is out of capacity already: 3 of 5 days for a full-timer.
    assert la_ads["people"][0]["capacity_h"] == 24.0
    assert la_ads["sources"][0]["as_of"] == "2026-11-18T06:00:00Z"

    strategists = await resourcing_capacity("2027-01-01", "2027-03-31", role="strategy")
    assert strategists["totals"][0]["people"] == 7

    london = await resourcing_capacity("2026-12-01", "2026-12-31", office="London", team="design")
    harry = next(r for r in london["people"] if r["person"] == "Harry Evans")
    assert harry["office_from"] == "ub_name_match"
    assert any("Harry Evans" in w for w in london["warnings"])

    isaac = await resourcing_capacity("2026-11-18", "2026-12-24", person="isaac.moore@anomaly.com")
    assert [r["person"] for r in isaac["people"]] == ["Isaac Moore"]
    assert isaac["totals"][0]["people"] == 1 and isaac["lists"][0]["over_capacity"] == "nobody"
    january = await resourcing_capacity("2027-01-01", "2027-01-31")
    assert len(january["lists"][0]["nothing_booked"].split(", ")) >= 7

    unknown = await resourcing_capacity(*week, office="Paris")
    assert unknown["error"] is True and "Los Angeles" in unknown["message"]
    too_long = await resourcing_capacity("2027-01-01", "2027-06-30")
    assert too_long["error"] is True and "100 days" in too_long["message"]


def test_the_float_subset(truth):
    qs = questions.build_float(truth)
    assert len(qs) == len(questions.FLOAT_SUBSET)
    assert all(q["id"].startswith("float-") for q in qs)
    # A follow-up never runs without the turn it follows.
    by_chain: dict[str, int] = {}
    for q in qs:
        by_chain[q["chain"]] = by_chain.get(q["chain"], 0) + 1
    assert by_chain["float-la-ad"] == by_chain["float-over"] == 2
