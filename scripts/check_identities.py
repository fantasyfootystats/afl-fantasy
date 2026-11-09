"""Sense checks for player identity problems: namesakes merged into one ID, or one player split across IDs.

Run after build.py / drafts.py:  python3 scripts/check_identities.py
Writes reports/identity_checks.csv (one row per NEW flag, for a human to review) and prints a summary.
Flags already confirmed correct live in data/reference/identity_confirmed.csv and are hidden; when a
new flag is checked and found fine, copy its check / playerId / detail into that file.

Merged namesakes (two players under one ID) show up as:
  - two clubs in the same season (no mid-season trades in the AFL; a late-season move is rare)
  - two rows for one ID in the same game
  - implausible age at a game (under 16.5 or over 40)
  - a club bounce A -> B -> A
  - a long gap then a return at a different club
  - a sudden role switch (ruck <-> non-ruck) at the same time as a club change
One player split across IDs shows up as:
  - two IDs with compatible names whose careers don't overlap, where one ends and the other starts
    within two seasons (could be a renamed or re-numbered player)
  - same name, same dob, different IDs
Players.csv vs stats:
  - IDs in stats missing from players, or a lastSeason / firstSeason that disagrees with stats
Drafts:
  - an ID linked to two first-time draft picks
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data/clean"
OUT = ROOT / "reports"


def norm(x):
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", x)


def main():
    s = pd.read_csv(CLEAN / "stats.csv", low_memory=False,
                    usecols=["gameId", "team", "year", "round", "playerId", "displayName", "hitOuts", "fantasyScore"])
    g = pd.read_csv(CLEAN / "games.csv", usecols=["gameId", "date"], parse_dates=["date"])
    p = pd.read_csv(CLEAN / "players.csv")
    d = pd.read_csv(CLEAN / "drafts.csv")
    s = s.merge(g, on="gameId", how="left")
    s["hitOuts"] = s.hitOuts.fillna(0)
    pl = p.dropna(subset=["playerId"]).drop_duplicates("playerId").set_index("playerId")
    flags = []

    def flag(kind, pid, detail, severity):
        name = s.loc[s.playerId == pid, "displayName"].iloc[-1] if (s.playerId == pid).any() else pl.displayName.get(pid, "")
        flags.append({"severity": severity, "check": kind, "playerId": pid, "player": name, "detail": detail})

    # season summary per ID
    ys = (s.groupby(["playerId", "year", "team"]).agg(games=("gameId", "size"), first=("date", "min"),
                                                      ho=("hitOuts", "mean")).reset_index())

    # 1. two clubs in one season
    multi = ys.groupby(["playerId", "year"]).team.nunique()
    for (pid, y) in multi[multi > 1].index:
        r = ys[(ys.playerId == pid) & (ys.year == y)]
        flag("two clubs in one season", pid, f"{y}: " + ", ".join(f"{t} ({n})" for t, n in zip(r.team, r.games)), "high")

    # 2. same ID twice in one game
    dup = s[s.duplicated(["gameId", "playerId"], keep=False)]
    for pid, r in dup.groupby("playerId"):
        flag("listed twice in one game", pid, f"{r.gameId.nunique()} games, e.g. {r.gameId.iloc[0]}", "high")

    # 3. implausible age
    dob = pd.to_datetime(pl.dob, errors="coerce")
    s["age"] = (s.date - s.playerId.map(dob)).dt.days / 365.25
    for pid, r in s[(s.age < 16.5) | (s.age > 40)].groupby("playerId"):
        flag("implausible age at a game", pid, f"age {r.age.min():.1f}-{r.age.max():.1f} in {r.year.min()}-{r.year.max()}", "high")

    # 4. club bounce A -> B -> A, and 5. long gap then return at a new club
    main_team = ys.sort_values("games").drop_duplicates(["playerId", "year"], keep="last").sort_values(["playerId", "year"])
    for pid, r in main_team.groupby("playerId"):
        teams, years, ho = r.team.tolist(), r.year.tolist(), r.ho.tolist()
        for i in range(1, len(teams) - 1):
            if teams[i - 1] == teams[i + 1] != teams[i]:
                flag("club bounce A-B-A", pid, f"{years[i-1]} {teams[i-1]} -> {years[i]} {teams[i]} -> {years[i+1]} {teams[i+1]}", "medium")
        for i in range(1, len(teams)):
            gap = years[i] - years[i - 1]
            if gap >= 4 and teams[i] != teams[i - 1]:
                flag("long gap, returns at new club", pid, f"{years[i-1]} {teams[i-1]} -> {years[i]} {teams[i]} ({gap} yrs)", "medium")
            # 6. role switch with club change: ruck (10+ hit-outs) <-> non-ruck (<2)
            if teams[i] != teams[i - 1] and ((ho[i - 1] >= 10 and ho[i] < 2) or (ho[i - 1] < 2 and ho[i] >= 10)):
                flag("role switch with club change", pid,
                     f"{years[i-1]} {teams[i-1]} {ho[i-1]:.1f} HO -> {years[i]} {teams[i]} {ho[i]:.1f} HO", "medium")

    # 7. possibly one player split across IDs: compatible names, careers back-to-back, never overlapping
    car = s.groupby("playerId").agg(name=("displayName", "last"), first=("year", "min"), last=("year", "max"),
                                    clubs=("team", lambda x: ", ".join(dict.fromkeys(x)))).reset_index()
    nm = car.name.str.split(", ", n=1)
    car["last_n"], car["first_n"] = nm.str[0].map(norm), nm.str[1].fillna("").map(norm)
    car["dob"] = car.playerId.map(pl.dob)
    for _, grp in car.groupby("last_n"):
        if len(grp) < 2:
            continue
        rows = grp.to_dict("records")
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                fa, fb = a["first_n"], b["first_n"]
                if not (fa[:3] == fb[:3] or fa.startswith(fb) or fb.startswith(fa)):
                    continue
                same_dob = pd.notna(a["dob"]) and a["dob"] == b["dob"]
                lo, hi = (a, b) if a["first"] <= b["first"] else (b, a)
                back_to_back = lo["last"] < hi["first"] <= lo["last"] + 2
                if same_dob or (back_to_back and (pd.isna(a["dob"]) or pd.isna(b["dob"]))):
                    flag("possibly one player under two IDs", lo["playerId"],
                         f"{lo['name']} {lo['first']}-{lo['last']} ({lo['clubs']}) | ID {hi['playerId']} {hi['name']} "
                         f"{hi['first']}-{hi['last']} ({hi['clubs']})" + (" | same dob" if same_dob else ""),
                         "high" if same_dob else "low")

    # 8. players.csv vs stats
    missing = set(car.playerId) - set(pl.index)
    for pid in missing:
        flag("in stats but not players.csv", pid, "", "high")
    both = car.set_index("playerId").join(pl[["firstSeason", "lastSeason"]], how="inner")
    for pid, r in both[(both["first"] != both.firstSeason) | (both["last"] != both.lastSeason)].iterrows():
        flag("players.csv seasons disagree with stats", pid,
             f"stats {r['first']}-{r['last']}, players {r.firstSeason}-{r.lastSeason}", "low")

    # 9. one ID linked to two first-time draft picks
    fd = d[d.isFirstDraft & d.playerId.notna()]
    for pid, r in fd[fd.duplicated("playerId", keep=False)].groupby("playerId"):
        flag("two first-time draft picks", int(pid),
             "; ".join(f"{x.draftYear} {x.draftType} {x.pick} {x.club} ({x.player})" for x in r.itertuples()), "high")

    # 10. copied stat line: same player, identical stats and TOG in the same or next round
    #     (how the Hibberd and 2022 Taylor Walker phantom rows looked). Coincidental repeats years
    #     apart are common across 180k rows, so only near-adjacent rounds are flagged.
    full = pd.read_csv(CLEAN / "stats.csv", low_memory=False)
    cols = ["kicks", "handballs", "marks", "goals", "behinds", "hitOuts", "tackles", "freesFor", "freesAgainst", "pctPlayed"]
    full = full.fillna({c: 0 for c in cols})
    full["rnd"] = full["round"].str[1:].astype(int)
    rep = full[full.duplicated(["playerId", *cols], keep=False)]
    for _, grp in rep.groupby(["playerId", *cols]):
        grp = grp.sort_values(["year", "rnd"])
        close = (grp.year.diff() == 0) & (grp.rnd.diff() <= 1)
        if close.any():
            flag("copied stat line", grp.playerId.iloc[0],
                 " / ".join(f"{x.gameId} {x.team} v {x.opposition}" for x in grp.itertuples()), "high")

    out = pd.DataFrame(flags)
    # flags a human has already confirmed as correct (data/reference/identity_confirmed.csv) stay quiet
    ok = pd.read_csv(ROOT / "data/reference/identity_confirmed.csv")
    seen = out.set_index(["check", "playerId", "detail"]).index.isin(ok.set_index(["check", "playerId", "detail"]).index)
    print(f"{seen.sum()} previously confirmed flags hidden (data/reference/identity_confirmed.csv)")
    out = out[~seen]
    if out.empty:
        print("no new identity flags")
        (OUT / "identity_checks.csv").unlink(missing_ok=True)
        return
    order = {"high": 0, "medium": 1, "low": 2}
    out = out.sort_values(["severity", "check", "player"], key=lambda c: c.map(order) if c.name == "severity" else c)
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "identity_checks.csv", index=False)
    print(out.groupby(["severity", "check"], sort=False).size().to_string())
    print(f"\n{len(out)} flags -> {OUT / 'identity_checks.csv'}")


if __name__ == "__main__":
    main()
