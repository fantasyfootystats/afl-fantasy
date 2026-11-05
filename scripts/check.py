"""Integrity checks on data/clean. Run after build:  python3 scripts/check.py"""
from pathlib import Path

import pandas as pd

C = Path(__file__).resolve().parents[1] / "data/clean"
g = pd.read_csv(C / "games.csv")
s = pd.read_csv(C / "stats.csv", low_memory=False)
p = pd.read_csv(C / "players.csv", low_memory=False)
v = pd.read_csv(C / "validGames.csv")

W = {"kicks": 3, "handballs": 2, "marks": 3, "tackles": 4, "freesFor": 1, "freesAgainst": -3, "hitOuts": 1, "goals": 6, "behinds": 1}
checks = {
    "fantasy score = standard AF formula on every row": (sum(s[c].fillna(0) * w for c, w in W.items()).round() == s.fantasyScore).all(),
    "kicks + handballs = disposals": (s.kicks.fillna(0) + s.handballs.fillna(0) == s.disposals.fillna(0)).all(),
    "one row per player per game": not s.duplicated(["gameId", "playerId"]).any(),
    "gameIds unique": g.gameId.is_unique,
    "every stats game is in games": s.gameId.isin(g.gameId).all(),
    "every 2005-2026 game has stats": g[g.year <= 2026].gameId.isin(s.gameId).all(),
    "every stats player is in players": s.playerId.isin(p.playerId).all(),
    "playerIds unique": p.playerId.dropna().is_unique,
    "22-23 players per team-game": s.groupby(["gameId", "team"]).size().between(22, 23).all(),
    # 2025R2410 (Gold Coast v Essendon) is a postponed game played in R24 week, so both clubs appear twice there
    "each team plays once per round": not pd.concat([g[g.gameId != "2025R2410"][["year", "round", c]].set_axis(["y", "r", "t"], axis=1)
                                                     for c in ("homeTeam", "awayTeam")]).duplicated().any(),
    "2027 fixture: 207 games, 23 per team": len(g[g.year == 2027]) == 207 and
        pd.concat([g[g.year == 2027].homeTeam, g[g.year == 2027].awayTeam]).value_counts().eq(23).all(),
    "validGames has no subs and TOG >= 25": v.gameId.isin(s.gameId).all() and (v.pctPlayed >= 25).all(),
}
for name, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + name)
print(f"\ngames {len(g)} ({g.year.min()}-{g.year.max()}), stats rows {len(s)}, validGames {len(v)}, players {len(p)}")
raise SystemExit(0 if all(checks.values()) else 1)
