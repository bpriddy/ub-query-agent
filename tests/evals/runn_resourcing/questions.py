"""
questions.py — what a producer or resource manager asks `/ask` about Runn,
each with its ground truth computed from the fixtures (truth.py).

Today is pinned to Wednesday 2026-11-18 (world.TODAY). So:
  this Friday       2026-11-20
  next week         2026-11-23..27 (US Thanksgiving Thu 26, Fri 27)
  starting Monday   2026-11-23
  the week after    2026-11-30..12-04
  this month        November 2026
  Q1                2027-01-01..03-31

A question is `{id, text, chain?, kind, needs, truth}`:
  chain   questions with the same chain run in one session, in order (a
          follow-up reads the turn before it)
  kind    names    — a list of people/projects: `core` must be named,
                     `acceptable` may be, `forbidden` must not be put forward
                     as an answer (mentioning them as booked is fine)
          value    — a figure or a fact, in `expected`
          clarify  — the right reply asks which one (or answers for each)
  needs   the data the answer needs beyond today's tool (the gap analysis):
          office, role, skills, contracts, time_off, holidays, tentative,
          placeholders, enumerate (every person / every project, not one
          named), identity (a person with no or a different Runn record)
  today   yes / partial / no: can today's relay + tool answer it at all
"""

from __future__ import annotations

from .truth import Truth, d

NEXT_WEEK = (d("2026-11-23"), d("2026-11-27"))
WEEK_AFTER = (d("2026-11-30"), d("2026-12-04"))
THIS_FRIDAY = (d("2026-11-20"), d("2026-11-20"))
NOVEMBER = (d("2026-11-01"), d("2026-11-30"))
DECEMBER = (d("2026-12-01"), d("2026-12-31"))
JANUARY = (d("2027-01-01"), d("2027-01-31"))
Q1 = (d("2027-01-01"), d("2027-03-31"))
THREE_WEEKS = (d("2026-11-23"), d("2026-12-11"))

AD = ("Art Director",)
ART = ("Art Director", "Creative Director")


def _names(core, acceptable=(), forbidden=(), note=""):
    return {
        "core": list(core),
        "acceptable": list(acceptable),
        "forbidden": list(forbidden),
        "note": note,
    }


def build(t: Truth) -> list[dict]:
    q: list[dict] = []

    def add(qid, text, kind, needs, today, truth, chain=None):
        q.append(
            {
                "id": qid,
                "text": text,
                "chain": chain or qid,
                "kind": kind,
                "needs": needs,
                "today": today,
                "truth": truth,
            }
        )

    by_name = {t.name(p): p for p in t.people}

    def plans(pids, window):
        ids = [by_name.get(p, p) for p in pids]
        return " | ".join(t.describe(p, *window) for p in ids)

    la_ads = t.select(office="LA", roles=AD)
    a = t.availability(la_ads, *NEXT_WEEK)
    add(
        "avail-ad-la-next-week",
        "Who is available for art direction in LA next week?",
        "names",
        ["enumerate", "office", "role", "time_off", "holidays", "tentative", "contracts"],
        "no",
        _names(
            a["core"],
            a["acceptable"] + ["Leo Martinez"],
            a["forbidden"],
            "Next week is Mon 23 - Fri 27 Nov; Thu 26 and Fri 27 are US holidays. "
            "Theo Grant is on leave all week. Priya Nair is booked only on the TENTATIVE "
            "Solace pitch. Sofia Reyes works Mon-Thu and is half booked. Leo Martinez is a "
            "designer with the Art Direction skill (fine to suggest). Kai Nakamura (UB) has no "
            "Runn record. " + plans(la_ads, NEXT_WEEK),
        ),
        chain="la-ad",
    )
    a2 = t.availability(la_ads, *WEEK_AFTER)
    add(
        "avail-ad-la-week-after",
        "What about the week after?",
        "names",
        ["enumerate", "office", "role", "tentative", "contracts"],
        "no",
        _names(
            a2["core"],
            a2["acceptable"] + ["Leo Martinez"],
            a2["forbidden"],
            "Follow-up: LA art directors, 30 Nov - 4 Dec. " + plans(la_ads, WEEK_AFTER),
        ),
        chain="la-ad",
    )

    nyc = t.select(office="NYC")
    o = t.over_capacity(nyc, *NOVEMBER)
    add(
        "over-nyc-month",
        "Who is over capacity in NYC this month?",
        "names",
        ["enumerate", "office", "contracts", "time_off"],
        "no",
        _names(
            o["core"],
            [],
            o["forbidden"],
            "Over capacity = booked hours above contracted hours across November. "
            + plans(o["core"], NOVEMBER),
        ),
        chain="over",
    )
    la = t.select(office="LA")
    o2 = t.over_capacity(la, *NOVEMBER)
    add(
        "over-la-month-followup",
        "And in LA?",
        "names",
        ["enumerate", "office", "contracts", "time_off"],
        "no",
        _names(o2["core"], [], o2["forbidden"], plans(o2["core"], NOVEMBER)),
        chain="over",
    )

    designers = t.select(roles=("Designer",))
    a3 = t.availability(designers, *THREE_WEEKS, core_at=0.75, forbid_below=0.5)
    add(
        "free-design-3wk",
        "Who's free to pick up a 3-week design job starting Monday?",
        "names",
        ["enumerate", "role", "time_off", "contracts", "tentative"],
        "no",
        _names(
            a3["core"],
            a3["acceptable"],
            a3["forbidden"],
            "Window 23 Nov - 11 Dec, any office, designers, at least ~75% free. Ella Ward is a "
            "London freelancer whose contract runs to 23 Dec. Zoe Chen is on parental leave. "
            + plans(designers, THREE_WEEKS),
        ),
    )

    add(
        "jane-next-week",
        "What's Jane on next week?",
        "clarify",
        ["identity"],
        "partial",
        {
            "expected": "There are two Janes: Jane Park (LA art director) and Jane Okafor "
            "(London strategist). Either ask which one, or answer for both. "
            + plans([1, 30], NEXT_WEEK)
        },
        chain="jane",
    )
    add(
        "jane-park-followup",
        "Jane Park, in LA.",
        "value",
        ["holidays"],
        "yes",
        {
            "expected": "Jane Park is on Halcyon Hotels - Winter Campaign, 8 h/day, all of next "
            "week's working days (Thu 26 / Fri 27 are US holidays). " + plans([1], NEXT_WEEK)
        },
        chain="jane",
    )

    under = t.understaffed(d("2026-11-18"), Q1[1])
    add(
        "understaffed-projects",
        "Which projects are understaffed?",
        "names",
        ["enumerate", "placeholders"],
        "no",
        _names(
            under,
            [],
            [],
            "Understaffed = a placeholder (unfilled role) is booked on it: a Designer on Lumen "
            "Brand Refresh (23 Nov - 18 Dec), a Producer on Northwind Q1 Retainer (Jan-Mar), an "
            "Art Director on the tentative Solace Health pitch (23 Nov - 4 Dec).",
        ),
    )

    leave_dec = t.on_leave(*DECEMBER)
    add(
        "leave-december",
        "Who's on leave in December?",
        "names",
        ["enumerate", "time_off"],
        "no",
        _names(
            leave_dec,
            [],
            [],
            "Leave records overlapping December (public holidays are not leave). "
            + "; ".join(
                f"{t.name(x['personId'])} {x['startDate']}..{x['endDate']} ({x['note']})"
                for x in t.runn["time_offs_leave"]
            ),
        ),
    )

    strat = t.select(roles=("Strategist",))
    u = t.utilization(strat, *Q1)
    add(
        "strategy-q1",
        "How booked is the strategy team in Q1?",
        "value",
        ["enumerate", "role", "contracts", "holidays", "tentative"],
        "no",
        {
            "expected": f"Strategists across all offices, Jan-Mar 2027: about "
            f"{u['confirmed_pct']}% booked on confirmed work ({u['total_pct']}% counting the "
            f"tentative Kite Mobile launch); {u['confirmed_hours']} h confirmed + "
            f"{u['tentative_hours']} h tentative of {u['capacity_hours']} h capacity. "
            "Daniel Kim is over capacity (Lumen + Northwind Q1). Accept +-10 points. "
            + plans(strat, Q1)
        },
    )

    add(
        "halcyon-team",
        "Who is staffed on the Halcyon winter campaign?",
        "names",
        [],
        "yes",
        _names(t.project_people(101, NOVEMBER[0], DECEMBER[1]), [], [], "Project 101."),
    )
    add(
        "maya-month",
        "What is maya.singh@anomaly.com working on this month?",
        "value",
        [],
        "yes",
        {"expected": plans([17], NOVEMBER) + " She is over capacity (12 h/day booked)."},
    )
    add(
        "sofia-hours-week",
        "How many hours a week is Sofia Reyes booked on Vela Motors?",
        "value",
        ["contracts"],
        "partial",
        {
            "expected": "16 hours a week: 4 h/day on her four working days (she works Mon-Thu). "
            "20 h/week is wrong. " + plans([3], NOVEMBER)
        },
    )
    add(
        "marcus-free",
        "Is Marcus Bell free next week?",
        "value",
        ["holidays"],
        "yes",
        {
            "expected": "Yes: nothing booked 23-27 Nov (Thu 26 and Fri 27 are US holidays, so 3 "
            "working days). His next booking is Northwind Q1 Retainer from 4 Jan. "
            + plans([2], NEXT_WEEK)
        },
    )
    ldn_des = t.select(office="London", roles=("Designer",))
    a4 = t.availability(ldn_des, *DECEMBER)
    add(
        "london-designers-dec",
        "Which London designers have capacity in December?",
        "names",
        ["enumerate", "office", "role", "contracts", "holidays"],
        "no",
        _names(
            a4["core"],
            a4["acceptable"],
            a4["forbidden"],
            "Harry Evans, Isla Murray (Mon-Wed) are booked to 11 Dec then free; Ella Ward's "
            "freelance contract ends 23 Dec. " + plans(ldn_des, DECEMBER),
        ),
    )
    ldn = t.select(office="London")
    tentative_ldn = sorted(
        {
            t.name(p)
            for p in ldn
            for x in t.assignments(p, d("2026-11-18"), Q1[1])
            if not t.projects[x["projectId"]]["isConfirmed"]
        }
    )
    add(
        "london-tentative",
        "Who in London is booked on tentative work?",
        "names",
        ["enumerate", "office", "tentative"],
        "no",
        _names(
            tentative_ldn,
            [],
            [],
            "Kite Mobile - Spring Launch (Jan-Mar 2027) is the tentative project they are on.",
        ),
    )
    add(
        "solace-confirmed",
        "Is the Solace Health pitch confirmed yet?",
        "value",
        ["tentative"],
        "no",
        {"expected": "No: Solace Health - Pitch is tentative (isConfirmed false) in Runn."},
    )
    motion = t.select(skill="Motion Design")
    a5 = t.availability(motion, *NEXT_WEEK)
    add(
        "motion-free-next-week",
        "Who has motion design skills and is free next week?",
        "names",
        ["enumerate", "skills", "time_off", "tentative"],
        "partial",
        _names(
            a5["core"],
            a5["acceptable"],
            a5["forbidden"],
            "Motion Design is a Runn skill (and a UB staff skill). " + plans(motion, NEXT_WEEK),
        ),
    )
    prods = t.select(roles=("Producer",))
    a6 = t.availability(prods, *THIS_FRIDAY, core_at=1.0, forbid_below=0.5)
    a6["acceptable"] = [
        n for n in t.availability(prods, *THIS_FRIDAY)["core"] if n not in a6["core"]
    ]
    add(
        "producers-friday",
        "Which producers are free this Friday?",
        "names",
        ["enumerate", "role"],
        "no",
        _names(
            a6["core"],
            a6["acceptable"],
            a6["forbidden"],
            "This Friday = 20 Nov 2026. Rosa Diaz (UB, NYC producer) has no Runn record. "
            + plans(prods, THIS_FRIDAY),
        ),
    )
    nyc_copy = t.select(office="NYC", roles=("Copywriter",))
    a7 = t.availability(nyc_copy, *JANUARY)
    add(
        "nyc-copy-kite-jan",
        "Do we have a copywriter in New York who could start on Kite Mobile in January?",
        "names",
        ["enumerate", "office", "role", "time_off"],
        "no",
        _names(
            a7["core"],
            a7["acceptable"],
            a7["forbidden"],
            "Kite Mobile - Spring Launch is tentative, 11 Jan - 26 Mar. "
            + plans(nyc_copy, JANUARY),
        ),
    )
    add(
        "northwind-team",
        "Who's working on Northwind?",
        "clarify",
        [],
        "yes",
        {
            "expected": "Two Runn projects match: Northwind Air - Q1 Retainer and Northwind Air "
            "- Always On. Ask which, or answer for both: Always On = "
            + ", ".join(t.project_people(108, d("2026-11-18"), DECEMBER[1]))
            + "; Q1 Retainer = "
            + ", ".join(t.project_people(105, *Q1))
            + " (plus an unfilled Producer placeholder)."
        },
    )
    add(
        "kai-booked",
        "What is Kai Nakamura booked on?",
        "value",
        ["identity"],
        "yes",
        {
            "expected": "Kai Nakamura (LA art director in UB, started 16 Nov) has no Runn "
            "record, so there is no plan to report. Saying he is free/unbooked is WRONG."
        },
    )
    add(
        "harry-next-week",
        "Is Harry Evans available next week?",
        "value",
        ["identity"],
        "partial",
        {
            "expected": "No: Harry Evans is full time on Lumen Bank - Annual Report "
            "(7.5 h/day, 23 Nov - 11 Dec). His Runn email (harry.evans@anomaly.co.uk) "
            "differs from UB's, so an email join misses him; 'not in Runn' or 'free' is wrong. "
            + plans([33], NEXT_WEEK)
        },
    )
    pt = [p for p in t.select(roles=("Designer",)) if t.is_part_time(p)]
    add(
        "part-time-design",
        "Who's part-time in the design team?",
        "names",
        ["enumerate", "role", "contracts"],
        "no",
        _names(
            [t.name(p) for p in pt],
            ["Sofia Reyes"],
            [],
            "Ruby Alvarez: 4 h/day; Isla Murray: Mon-Wed. Sofia Reyes (Mon-Thu) is an art "
            "director, acceptable if labelled.",
        ),
    )
    vela = t.project_people(102, d("2026-11-18"), d("2026-11-18"))
    add(
        "vela-headcount",
        "How many people are on Vela Motors EV Launch right now, and how many hours a day "
        "does that add up to?",
        "value",
        [],
        "yes",
        {
            "expected": f"{len(vela)} people ({', '.join(vela)}); "
            + str(
                sum(
                    x["minutesPerDay"]
                    for x in t.runn["assignments"]
                    if x["projectId"] == 102 and x["startDate"] <= "2026-11-18" <= x["endDate"]
                )
                / 60
            )
            + " h/day in total."
        },
    )
    ads = t.select(roles=AD)
    under_ads = [p for p in ads if t.load(p, *NOVEMBER).share(t.load(p, *NOVEMBER).confirmed) < 0.5]
    add(
        "underbooked-ads-month",
        "Which art directors are underbooked this month?",
        "names",
        ["enumerate", "role", "time_off", "tentative"],
        "no",
        _names(
            [
                t.name(p)
                for p in under_ads
                if t.load(p, *NOVEMBER).capacity > 0 and t.load(p, *NOVEMBER).tentative == 0
            ],
            [t.name(p) for p in under_ads if t.load(p, *NOVEMBER).tentative > 0],
            [t.name(p) for p in ads if p not in under_ads],
            "Under 50% of November capacity on confirmed work (Priya Nair only because her "
            "work is tentative; fine either way if that is said). " + plans(ads, NOVEMBER),
        ),
    )
    e = t.load(16, d("2026-12-01"), d("2026-12-31"))
    add(
        "ethan-3days-dec",
        "Can Ethan Brooks take on three days a week from December 1st?",
        "value",
        ["holidays"],
        "yes",
        {
            "expected": "Yes: Ethan Brooks has nothing booked in December "
            f"({e.capacity / 60:g} h capacity, 0 booked); his next booking is the TENTATIVE "
            "Kite Mobile launch from 11 Jan."
        },
    )
    add(
        "lumen-confirmed",
        "Who's on the Lumen Bank brand refresh, and is that work confirmed?",
        "names",
        ["tentative", "placeholders"],
        "partial",
        _names(
            t.project_people(103, d("2026-11-18"), Q1[1]),
            [],
            [],
            "Lumen Bank - Brand Refresh is confirmed; it also has an unfilled Designer "
            "placeholder 23 Nov - 18 Dec. Not to be confused with Lumen Bank - Annual Report.",
        ),
    )
    add(
        "la-off-thanksgiving",
        "Is anyone in LA off over Thanksgiving week?",
        "value",
        ["enumerate", "office", "time_off", "holidays"],
        "no",
        {
            "expected": "Theo Grant is on leave all week (23-27 Nov); Thu 26 and Fri 27 are "
            "US public holidays for everyone in LA."
        },
    )
    lu = t.utilization(ldn, *NOVEMBER)
    add(
        "london-util-month",
        "What's London's utilization this month?",
        "value",
        ["enumerate", "office", "contracts", "holidays"],
        "no",
        {
            "expected": f"About {lu['confirmed_pct']}% of London's November capacity is booked "
            f"({lu['confirmed_hours']} h of {lu['capacity_hours']} h; no tentative work in "
            "November). Accept +-10 points."
        },
    )
    a8 = t.availability([p for p in la_ads if p != 4], *NEXT_WEEK)
    add(
        "cover-theo",
        "Theo's out next week. Who can cover for him?",
        "names",
        ["enumerate", "office", "role", "time_off", "tentative"],
        "no",
        _names(
            a8["core"],
            a8["acceptable"] + ["Leo Martinez"],
            a8["forbidden"],
            "Theo Grant: LA art director, on leave 23-27 Nov, on Vela Motors - EV Launch. Cover = "
            "free LA art directors that week. " + plans(la_ads, NEXT_WEEK),
        ),
    )
    add(
        "jack-back",
        "When is Jack Turner back from leave?",
        "value",
        ["time_off", "holidays"],
        "no",
        {
            "expected": "Monday 4 January 2027: his leave runs 14-31 Dec and 1 Jan is a public "
            "holiday."
        },
    )
    add(
        "orchard-billable",
        "On Orchard Foods Holiday Social, whose time is billable?",
        "names",
        [],
        "yes",
        _names(
            t.project_people(104, NOVEMBER[0], DECEMBER[1]),
            [],
            [],
            "Every assignment on Orchard Foods - Holiday Social is billable.",
        ),
    )
    nyc_strat = t.select(office="NYC", roles=("Strategist",))
    a9 = t.availability(nyc_strat, *WEEK_AFTER)
    add(
        "nyc-strategy-week-after-next",
        "Who's free for strategy work in New York the week after next?",
        "names",
        ["enumerate", "office", "role", "contracts"],
        "no",
        _names(
            a9["core"],
            a9["acceptable"],
            a9["forbidden"],
            "Week after next = 30 Nov - 4 Dec. " + plans(nyc_strat, WEEK_AFTER),
        ),
    )
    everyone = list(t.people)
    bench = [
        p for p in everyone if t.load(p, *JANUARY).capacity > 0 and t.load(p, *JANUARY).booked == 0
    ]
    add(
        "bench-january",
        "Who has nothing booked at all in January?",
        "names",
        ["enumerate", "time_off", "contracts"],
        "no",
        _names(
            [t.name(p) for p in bench],
            [],
            [],
            "People with January capacity and zero booked hours (tentative counts as booked). "
            "Zoe Chen is on parental leave (no capacity); Omar Haddad's and Ella Ward's "
            "contracts have ended.",
        ),
    )
    add(
        "capacity-definition",
        "Is Isaac Moore overbooked anywhere before Christmas?",
        "value",
        [],
        "yes",
        {
            "expected": "No: Isaac Moore was on Northwind Air - Always On (8 h/day) to 20 Nov and "
            "has nothing booked from 23 Nov to 31 Dec; he is never above capacity."
        },
    )
    add(
        "placeholder-lumen",
        "Is there still an open designer role on the Lumen brand refresh?",
        "value",
        ["placeholders"],
        "partial",
        {
            "expected": "Yes: an unfilled Designer placeholder is booked 8 h/day on Lumen Bank - "
            "Brand Refresh, 23 Nov - 18 Dec."
        },
    )
    return q


# The Float run: the same agency in Float (float_world.py), Runn not
# configured, a representative subset of the questions above — each failure
# type of run 1, by chain so follow-ups keep the turn they follow. The truth
# is the same: the Float fixtures are built from the Runn ones.
FLOAT_SUBSET = (
    "avail-ad-la-next-week",
    "avail-ad-la-week-after",
    "over-nyc-month",
    "over-la-month-followup",
    "free-design-3wk",
    "leave-december",
    "strategy-q1",
    "marcus-free",
    "london-tentative",
    "producers-friday",
    "harry-next-week",
    "kai-booked",
    "part-time-design",
    "london-util-month",
    "cover-theo",
    "nyc-strategy-week-after-next",
)


def build_float(t: Truth) -> list[dict]:
    return [
        {**q, "id": f"float-{q['id']}", "chain": f"float-{q['chain']}"}
        for q in build(t)
        if q["id"] in FLOAT_SUBSET
    ]
