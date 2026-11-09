"""AFL draft history 2005-2025 from Wikipedia (data/sources/wikipedia/afl_drafts_2005_2025.txt).

Run after build.py:  python3 scripts/drafts.py
Output: data/clean/drafts.csv - one row per pick, matched to playerId where the player has played AFL.

draftYear is the year of the draft event (Wikipedia's page year). Picks from the national, rookie and
pre-season drafts first play the following season; mid-season rookie picks play the same season.
Clubs re-listing an existing player through a draft (rookie elevations, re-drafts of delisted players)
get isFirstDraft = False, so career-entry analysis can filter to each player's first selection.
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "data/sources/wikipedia/afl_drafts_2005_2025.txt", ROOT / "data/clean"

CLUB = {"AD": "Adelaide", "BL": "Brisbane", "CA": "Carlton", "CW": "Collingwood", "ES": "Essendon",
        "FR": "Fremantle", "GE": "Geelong", "GC": "Gold Coast", "GW": "Greater Western Sydney", "HA": "Hawthorn",
        "ME": "Melbourne", "NM": "North Melbourne", "PA": "Port Adelaide", "RI": "Richmond", "SK": "St Kilda",
        "SY": "Sydney", "WC": "West Coast", "WB": "Western Bulldogs"}
TYPE = {"N": "national", "R": "rookie", "P": "pre-season", "M": "mid-season rookie", "X": "mini-draft"}
# Junior / U18 pathway leagues vs senior state leagues (for "mature-age" analysis)
JUNIOR = {"TACCup", "NABLeague", "TalentLeague", "SANFL U18s", "APS"}
STATE = {"VFL", "SANFL", "WAFL", "NEAFL", "TSL", "QAFL", "NTFL", "NTFL(Tas)", "TFL"}


def norm(x):
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", x)


NICKNAMES = {frozenset(p) for p in [
    ("jack", "john"), ("jack", "jackson"), ("john", "jonathon"), ("john", "jonathan"), ("jon", "jonathon"),
    ("bill", "william"), ("will", "william"), ("ted", "edward"), ("ned", "edward"), ("ed", "edward"),
    ("bob", "robert"), ("rob", "robert"), ("jim", "james"), ("jimmy", "james"), ("nat", "nathan"),
    ("tom", "thomas"), ("tommy", "thomas"), ("zac", "zachary"), ("zak", "zachary"), ("zach", "zachary"),
    ("matt", "matthew"), ("dan", "daniel"), ("danny", "daniel"), ("sam", "samuel"), ("ben", "benjamin"),
    ("alex", "alexander"), ("max", "maximus"), ("max", "maxwell"), ("nick", "nicholas"), ("mitch", "mitchell"),
    ("josh", "joshua"), ("lachie", "lachlan"), ("cam", "cameron"), ("pat", "patrick"), ("brad", "bradley"),
]}


def first_names_match(a, b):
    """Draft-list first name vs AFL first name (both normalised)."""
    return a[:3] == b[:3] or a.startswith(b) or b.startswith(a) or frozenset((a, b)) in NICKNAMES


def clean_name(n):
    n = re.sub(r"(NSW Scholarship elevation|Queensland priority selection|\[nb \d+\]|\(F/S\)|\+)", "", n)
    n = re.sub(r"\s+x$", "", n.strip())
    return n.replace("'Roy' ", "").strip()


def load():
    rows = []
    for line in SRC.read_text().splitlines():
        if line.startswith("#"):
            continue
        for tok in line.split(";"):
            p = tok.split(",")
            if len(p) < 5:
                continue
            yy, t, pick, club, name = p[:5]
            league = p[5] if len(p) > 5 else ""
            if not pick.isdigit() or name.strip().isdigit() or not name.strip():
                continue  # a handful of 2008 rookie rows came through with the columns shifted
            rows.append({"draftYear": 2000 + int(yy), "draftType": TYPE[t], "pick": int(pick), "club": CLUB[club],
                         "player": clean_name(name), "draftedFrom": league or None})
    d = pd.DataFrame(rows).drop_duplicates(["draftYear", "draftType", "pick", "club"])
    d["firstSeason"] = d.draftYear + (d.draftType != "mid-season rookie").astype(int)
    d["fromLevel"] = d.draftedFrom.map(lambda x: "junior" if x in JUNIOR else "state league" if x in STATE
                                       else "AFL (re-listed)" if x == "AFL" else ("other" if x else None))
    return d


def main():
    d = load()
    stats = pd.read_csv(OUT / "stats.csv", usecols=["playerId", "displayName", "team", "year"], low_memory=False)
    players = pd.read_csv(OUT / "players.csv", low_memory=False, usecols=["playerId", "dob"])
    seasons = stats.groupby(["playerId", "displayName"]).agg(first=("year", "min"), last=("year", "max")).reset_index()
    clubs = stats.groupby("playerId").team.agg(set)
    dob = pd.to_datetime(players.dropna(subset=["playerId"]).drop_duplicates("playerId")
                         .set_index("playerId").dob, errors="coerce")
    dob_year = dob.dt.year + (dob.dt.dayofyear - 1) / 365.25
    sp = seasons.displayName.str.split(", ", n=1)
    seasons["last_n"], seasons["first_n"] = sp.str[0].map(norm), sp.str[1].fillna("").map(norm)

    def match(r):
        toks = [w for w in r.player.split() if not re.fullmatch(r"[A-Z]\.|Jn?r\.?", w)]
        first, last = norm(toks[0]), norm(" ".join(toks[1:]))
        c = seasons[(seasons.last_n == last) | (seasons.last_n == norm(toks[-1]))]
        # must have played after this draft, and (for a first-time pick) not long before it
        c = c[(c["last"] >= r.firstSeason)]
        # first names must be compatible, not just share an initial (Logan vs Lachlan Smith,
        # Scott vs Sam Simpson): same first 3 letters, one a prefix of the other, or a nickname pair
        c = c[c.first_n.map(lambda f: first_names_match(first, f)).astype(bool)]
        # namesakes: never played for the drafting club AND either had already played AFL elsewhere
        # before this pick (and it isn't a re-listing) or would have been an implausible age at the
        # draft (e.g. the 2013 Melbourne and 2025 Sydney Max King picks are not St Kilda's Max King).
        # Late bloomers drafted and debuting years later elsewhere (Freeman, Menegola, Pink) stay linked.
        at_club = c.playerId.map(lambda p: r.club in clubs[p])
        played_before = (c["first"] < r.firstSeason) & (r.fromLevel != "AFL (re-listed)")
        age = (r.draftYear + 11 / 12) - c.playerId.map(dob_year)
        bad_age = ((age < 16.5) | (age > 32)) | (age.isna() & ((c["first"] - r.firstSeason) > 6))
        c = c[at_club | ~(played_before | bad_age)]
        if c.playerId.nunique() > 1:
            c = c[c.first_n.str[:3] == first[:3]]
        if c.playerId.nunique() > 1:
            c = c[c.playerId.map(lambda p: r.club in clubs[p])]
        if c.playerId.nunique() > 1:  # same name, same club: the one whose career is closest to the draft
            c = c.assign(gap=(c["first"] - r.firstSeason).abs()).sort_values("gap").head(1)
        return c.playerId.iloc[0] if c.playerId.nunique() == 1 else None

    d["playerId"] = d.apply(match, axis=1).astype("Int64")
    # a pick of a player who had already played AFL before it is a re-listing, not career entry
    first_game = stats.groupby("playerId").year.min()
    d["playedBefore"] = d.playerId.map(first_game) < d.firstSeason
    d = d.sort_values(["player", "draftYear", "pick"])
    who = d.playerId.astype("string").fillna("name:" + d.player.map(norm))  # unmatched picks: dedupe by name
    d["isFirstDraft"] = ~d.playedBefore & ~d.assign(_w=who).duplicated(subset=["_w"])
    d = d.merge(players.dropna(subset=["playerId"]).astype({"playerId": "Int64"}), on="playerId", how="left")
    # age at the draft (late November; mid-season draft late May)
    draft_day = pd.to_datetime(d.draftYear.astype(str) + d.draftType.map(
        lambda t: "-05-28" if t == "mid-season rookie" else "-11-25"))
    d["ageAtDraft"] = ((draft_day - pd.to_datetime(d.dob)).dt.days / 365.25).round(1)
    d = d.drop(columns=["dob", "playedBefore"]).sort_values(["draftYear", "draftType", "pick"])
    d.to_csv(OUT / "drafts.csv", index=False)

    first = d[d.isFirstDraft]
    print(f"drafts {len(d)} picks; first-time picks {len(first)}; matched to a playerId: "
          f"{first.playerId.notna().mean():.0%} of first-time picks")
    by = first.assign(m=first.playerId.notna()).groupby("draftType").m.agg(["size", "mean"]).round(2)
    print(by.to_string())


if __name__ == "__main__":
    main()
