"""Load DFS Australia pre-season AFL Fantasy spreadsheets (data/sources/dfs/dfs_afl_fantasy_YYYY.xlsx).

Each file is a snapshot at the start of season YYYY: one row per player in the AFL Fantasy game, with
starting price, ownership, position, the previous season's AFL summary (games, average, CBA%, PPM...) and
the previous season's state-league / under-18 averages. DFS excludes AFL games with < 50% TOG.

Run after build.py:  python3 scripts/dfs.py
Outputs: data/clean/dfs_preseason.csv, data/clean/dfs_other_leagues.csv
"""
import re
import unicodedata
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "data/sources/dfs", ROOT / "data/clean"

TEAM = {"ADE": "Adelaide", "BRL": "Brisbane", "CAR": "Carlton", "COL": "Collingwood", "ESS": "Essendon",
        "FRE": "Fremantle", "GEE": "Geelong", "GCS": "Gold Coast", "GWS": "Greater Western Sydney",
        "HAW": "Hawthorn", "MEL": "Melbourne", "NTH": "North Melbourne", "PTA": "Port Adelaide",
        "RIC": "Richmond", "STK": "St Kilda", "SYD": "Sydney", "WCE": "West Coast", "WBD": "Western Bulldogs"}
LEAGUE = {"VFL": "VFL", "WAFL": "WAFL", "SANFL": "SANFL", "SANFL Reserves": "SANFL Reserves",
          "SANFL U18s": "SANFL U18s", "NAB League": "Coates Talent League", "Coates Talent League": "Coates Talent League"}


def norm(x):
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", x)


def read_file(season):
    ws = openpyxl.load_workbook(SRC / f"dfs_afl_fantasy_{season}.xlsx", read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    hi = next(i for i, r in enumerate(rows) if r[0] == "Player")
    group, cols = None, []
    for g, h in zip(rows[hi - 1], rows[hi]):
        group = g or group
        cols.append((group, h))
    recs = []
    for r in rows[hi + 1:]:
        if not r[0]:
            continue
        d = {"season": season}
        other = {}
        for (g, h), val in zip(cols, r):
            if h is None or h == "Link":
                continue
            if g is None:
                d[{"Player": "dfsName", "Champion Data ID": "cdId", "Team": "dfsTeam", "Salary": "price",
                   "Owned": "ownedPct", "Position": "position"}[h]] = val
            elif re.fullmatch(r"\d{4} AFL Fantasy Stats", g):
                d["prev_" + {"GMS": "games", "FP": "avg", "MAX": "max", "100+": "tons", "120+": "over120", "CBA%": "cbaPct",
                             "PPM": "ppm", "REG": "regAvg", "L5": "last5Avg", "FIN": "finalsAvg"}[h]] = val
            elif g == "FP Averages":
                d[f"avg{h}"] = val
            else:
                lg = re.match(r"(\d{4}) (.+) Fantasy Stats", g)
                other.setdefault((int(lg.group(1)), LEAGUE[lg.group(2)]), {})[h] = val
        d["_other"] = other
        recs.append(d)
    return recs


def main():
    stats = pd.read_csv(OUT / "stats.csv", usecols=["playerId", "displayName", "team", "year"], low_memory=False)
    players = pd.read_csv(OUT / "players.csv", low_memory=False)
    cand = stats.drop_duplicates(["playerId", "team", "year"]).copy()
    sp = cand.displayName.str.split(", ", n=1)
    cand["last"], cand["first"] = sp.str[0].map(norm), sp.str[1].fillna("").map(norm)

    recs = [r for s in (2023, 2024, 2025, 2026) for r in read_file(s)]
    pre = pd.DataFrame([{k: v for k, v in r.items() if k != "_other"} for r in recs])
    pre["team"] = pre.dfsTeam.map(TEAM)
    assert pre.team.notna().all(), pre[pre.team.isna()].dfsTeam.unique()

    def match(name, team, season):
        """-> (playerId, level). Level 1 = same club around that season, matching first initial;
        level 2/3 = any club, first name must agree on its first 3 letters (Tom/Thomas still need level 1)."""
        toks = [w for w in name.split() if not re.fullmatch(r"[A-Z]\.|Jn?r\.?", w)]  # drop middle initials, Jr/Jnr
        first, last = norm(toks[0]), " ".join(toks[1:])
        c = cand[cand["last"].isin([norm(last), norm(toks[-1])])]
        levels = [(1, c[(c.team == team) & c.year.between(season - 1, season + 3)], 1),
                  (2, c[c.year.between(season - 3, season + 3)], 3), (3, c, 3)]
        for lvl, cc, n in levels:
            cc = cc[cc["first"].str[:n] == first[:n]]
            if cc.playerId.nunique() > 1:
                cc = cc[cc["first"].str[:3] == first[:3]]
            if cc.playerId.nunique() == 1:
                return cc.playerId.iloc[0], lvl
        return None, None

    res = [match(n, t, s) for n, t, s in zip(pre.dfsName, pre.team, pre.season)]
    pre["playerId"], pre["_lvl"] = [r[0] for r in res], [r[1] for r in res]
    # one DFS row per player per season: keep the best-level match (ties: the row with AFL games)
    pre["_rank"] = pre._lvl.fillna(9) * 10 - pre.prev_games.fillna(0).gt(0)
    dup = pre.dropna(subset=["playerId"]).sort_values("_rank").duplicated(["season", "playerId"])
    pre.loc[dup[dup].index, "playerId"] = None
    # carry a matched ID across seasons through the Champion Data ID (2024+ files have it)
    cd = pre.dropna(subset=["cdId", "playerId"]).groupby("cdId").playerId.agg(lambda x: x.mode()[0])
    pre["playerId"] = pre.playerId.fillna(pre.cdId.map(cd))
    # 2023 file has no CD ID: link by name + team to a 2024 row that has one
    key = pre.dfsName.map(norm) + pre.team
    k2cd = pre[pre.season == 2024].set_index(key[pre.season == 2024]).cdId
    k2cd = k2cd[~k2cd.index.duplicated()]
    pre.loc[pre.season == 2023, "cdId"] = key[pre.season == 2023].map(k2cd)
    pre["playerId"] = pre.playerId.fillna(pre.cdId.map(cd)).astype("Int64")

    other = []
    for r, pid in zip(recs, pre.playerId):
        for (yr, lg), v in r["_other"].items():
            if v.get("GMS"):
                other.append({"playerId": pid, "dfsName": r["dfsName"], "statYear": yr, "league": lg, "games": v["GMS"],
                              "avg": v["FP"], "max": v["MAX"], "tons": v["100+"]})
    oth = pd.DataFrame(other)
    oth["cdId"] = [pre.cdId.iloc[i] for i, r in enumerate(recs) for (yr, lg), v in r["_other"].items() if v.get("GMS")]
    oth = oth.drop_duplicates(["dfsName", "statYear", "league"])

    pre = pre.drop(columns=["_lvl", "_rank"])
    pre = pre[["season", "playerId", "cdId", "dfsName", "team", "position", "price", "ownedPct",
               *[c for c in pre.columns if c.startswith("prev_")], *sorted(c for c in pre.columns if c.startswith("avg"))]]
    pre.to_csv(OUT / "dfs_preseason.csv", index=False)
    oth.to_csv(OUT / "dfs_other_leagues.csv", index=False)

    played = pre.prev_games.fillna(0) > 0
    print(f"dfs_preseason {len(pre)} rows; matched {pre.playerId.notna().mean():.1%} overall, "
          f"{pre[played].playerId.notna().mean():.1%} of players with AFL games the previous season")
    print(f"dfs_other_leagues {len(oth)} player-seasons:", oth.league.value_counts().to_dict())
    un = pre[played & pre.playerId.isna()]
    if len(un):
        print("unmatched with AFL games:", un[["season", "dfsName", "team"]].values.tolist()[:20])


if __name__ == "__main__":
    main()
