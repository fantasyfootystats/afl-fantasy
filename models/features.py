"""Shared data loading and pre-season feature building.

Every feature for season N is built from games before season N, plus the
team each player lines up for in season N (known pre-season from trades,
delistings and retirements).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
RUCK_SHARE = 0.35          # share of team hit-outs that makes a player "the ruck"


def load():
    """validGames (with ppm), all stats (with per-game team totals), players (dob)."""
    v = pd.read_csv(CLEAN / "validGames.csv", parse_dates=["date"])
    v = v.sort_values(["date", "gameId"]).reset_index(drop=True)
    v["ppm"] = v.fantasyScore / v.pctPlayed  # points per % of game played

    s = pd.read_csv(CLEAN / "stats.csv", low_memory=False,
                    usecols=["gameId", "team", "year", "playerId", "fantasyScore", "hitOuts", "pctPlayed"])
    s["hitOuts"] = s.hitOuts.fillna(0)
    tot = s.groupby(["gameId", "team"])[["fantasyScore", "hitOuts"]].transform("sum")
    s["share"] = s.fantasyScore / tot.fantasyScore
    s["ho_share"] = np.where(tot.hitOuts > 0, s.hitOuts / tot.hitOuts, 0)

    p = pd.read_csv(CLEAN / "players.csv", usecols=["playerId", "dob"], parse_dates=["dob"])
    p = p.dropna(subset=["playerId"]).drop_duplicates("playerId")
    p["playerId"] = p.playerId.astype("int64")
    return v, s, p


def team_for_season(s, N):
    """Each player's main team in season N (most games)."""
    t = s[s.year == N].groupby(["playerId", "team"]).size().rename("n").reset_index()
    return t.sort_values("n").drop_duplicates("playerId", keep="last")[["playerId", "team"]]


def market_share_features(s, N, team_N=None):
    """Last-season share of team fantasy points, plus how much share left / arrived at the player's
    season-N team.  Departed = players who were at that team in N-1 but are at another club in N or
    never played again (delisted / retired).  Long-term injured players stay on the list, so they
    don't count as departures, which keeps this free of in-season information."""
    prev = s[s.year == N - 1]
    if prev.empty:
        return pd.DataFrame(columns=["playerId"])
    prev_team = team_for_season(s, N - 1).rename(columns={"team": "prev_team"})
    # season share = player's points / team's points over the games the team played
    team_pts = prev.groupby("team").fantasyScore.sum()
    pl = prev.groupby(["playerId", "team"]).fantasyScore.sum().reset_index()
    pl["season_share"] = pl.fantasyScore / pl.team.map(team_pts)
    pl = pl.groupby("playerId").season_share.sum().rename("prev_share")

    if team_N is None:
        team_N = team_for_season(s, N)
    last_year = s.groupby("playerId").year.max()
    moves = prev_team.merge(team_N, on="playerId", how="left")
    moves["prev_share"] = moves.playerId.map(pl)
    played_later = moves.playerId.map(last_year) >= N
    gone = moves.team.isna() & ~played_later        # never played again
    moved = moves.team.notna() & (moves.team != moves.prev_team)
    departed = moves[gone | moved].groupby("prev_team").prev_share.sum().rename("team_departed_share")
    arrived = moves[moved].groupby("team").prev_share.sum().rename("team_arrived_share")

    f = team_N.copy()
    f["prev_share"] = f.playerId.map(pl)
    f["team_departed_share"] = f.team.map(departed).fillna(0)
    f["team_arrived_share"] = f.team.map(arrived).fillna(0)
    pt = f.playerId.map(prev_team.set_index("playerId").prev_team)
    f["changed_team"] = (pt.notna() & (pt != f.team)).astype(int)
    # a mover's own departure shouldn't count as share freed up for himself
    return f.drop(columns="team")


def season_features(v, s, players, N, team_N=None):
    """Pre-season features for every player with valid-game history before season N."""
    hist = v[v.year < N]
    g = hist.groupby("playerId")
    t13 = g.tail(13).groupby("playerId")
    f = pd.DataFrame({
        "avg13": t13.fantasyScore.mean(),
        "avg5": g.tail(5).groupby("playerId").fantasyScore.mean(),
        "career_avg": g.fantasyScore.mean(),
        "career_games": g.size(),
        "seasons": g.year.nunique(),
        "tog13": t13.pctPlayed.mean(),
        "ppm13": t13.ppm.mean(),
    })
    for k in (1, 2, 3):
        y = hist[hist.year == N - k].groupby("playerId")
        f[f"prev{k}_avg"] = y.fantasyScore.mean()
        f[f"prev{k}_games"] = y.size()
        if k == 1:
            f["prev1_tog"] = y.pctPlayed.mean()
            f["prev1_ppm"] = y.ppm.mean()
            f["prev1_max"] = y.fantasyScore.max()
    for hl in (10, 25):
        f[f"ewm{hl}"] = g.fantasyScore.apply(lambda x: x.ewm(halflife=hl).mean().iloc[-1])
    f["ewm_ppm25"] = g.ppm.apply(lambda x: x.ewm(halflife=25).mean().iloc[-1])
    f["ewm_tog25"] = g.pctPlayed.apply(lambda x: x.ewm(halflife=25).mean().iloc[-1])

    # ruck role: share of team hit-outs over the last two seasons
    ho = s[(s.year >= N - 2) & (s.year < N)].groupby("playerId").ho_share.mean()
    f["ruck_share"] = ho
    f = f.reset_index()
    f["ruck_share"] = f.ruck_share.fillna(0)
    f["ruck"] = (f.ruck_share >= RUCK_SHARE).astype(int)

    f = f.merge(players, on="playerId", how="left")
    f["age"] = (pd.Timestamp(f"{N}-03-15") - f.dob).dt.days / 365.25
    f = f.drop(columns="dob")
    for c in ("prev1_games", "prev2_games", "prev3_games"):
        f[c] = f[c].fillna(0)

    ms = market_share_features(s, N, team_N)
    f = f.merge(ms, on="playerId", how="left")
    for c in ("team_departed_share", "team_arrived_share", "changed_team"):
        f[c] = f[c].fillna(0)
    f["season"] = N
    return f


def season_targets(v, N):
    y = v[v.year == N].groupby("playerId")
    return pd.DataFrame({"actual": y.fantasyScore.mean(), "games": y.size()}).reset_index()


# ---------- design matrices ----------
BASE_COLS = ["avg13", "avg5", "career_avg", "career_games", "seasons", "tog13", "ppm13",
             "prev1_avg", "prev1_games", "prev2_avg", "prev2_games", "prev3_avg", "prev3_games",
             "prev1_tog", "prev1_ppm", "prev1_max", "ewm10", "ewm25", "ewm_ppm25", "ewm_tog25", "age"]


def design(d, ruck=False, market=False):
    """Fill gaps and add derived terms.  ruck / market switch on the v2 feature groups."""
    x = d[BASE_COLS].copy()
    for c in ("prev1_avg", "prev2_avg", "prev3_avg"):
        x[c] = x[c].fillna(x.career_avg)
    x["prev1_tog"] = x.prev1_tog.fillna(x.tog13)
    x["prev1_ppm"] = x.prev1_ppm.fillna(x.ppm13)
    x["prev1_max"] = x.prev1_max.fillna(x.career_avg)
    x["age"] = x.age.fillna(25)
    x["age_c"] = x.age - 26
    x["age2"] = x.age_c ** 2
    x = x.drop(columns="age")
    if ruck:
        r = d.ruck.values
        x["ruck"] = r
        x["ruck_share"] = d.ruck_share.values
        x["ruck_age"] = r * x.age_c
        x["ruck_age2"] = r * x.age2
    if market:
        ps = d.prev_share.fillna(0).values
        x["prev_share"] = ps
        x["departed"] = d.team_departed_share.values
        x["arrived"] = d.team_arrived_share.values
        x["departed_x_share"] = x.departed * ps * 10   # do bigger ball-winners absorb more of it?
        x["changed_team"] = d.changed_team.values
    return x
