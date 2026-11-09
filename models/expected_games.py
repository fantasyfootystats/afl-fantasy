"""Expected games played: what share of his team's games will a listed player play (25%+ TOG)?

Season value for the draft = projected average x expected games, with a
replacement-level player filling the rest.

Pool: players with 5+ prior valid games who are still on a list in season N.
Historically "on a list" = plays in season N or any later season. Long-term
injured players who then retired without playing again drop out, so the
backtest slightly flatters the model; for 2027 the pool is active2027.

Run:  python3 models/expected_games.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import ROOT, CLEAN  # noqa: E402

OUT = ROOT / "models" / "output"
TEST_YEARS = range(2012, 2027)
COLS = ["rate1", "rate2", "rate3", "career_rate", "seasons", "age", "prev1_avg", "dnf1", "dnf_career", "ruck", "ever_full"]


def load_games():
    s = pd.read_csv(CLEAN / "stats.csv", low_memory=False,
                    usecols=["gameId", "team", "year", "playerId", "pctPlayed", "subs", "fantasyScore", "hitOuts"])
    s["played"] = s.pctPlayed >= 25
    s["dnf"] = s.subs.eq("Off")                      # left the game (sub off or injured, see README)
    team_games = s.groupby(["year", "team"]).gameId.nunique().rename("team_games")
    p = pd.read_csv(CLEAN / "players.csv", usecols=["playerId", "dob"], parse_dates=["dob"]).dropna(subset=["playerId"])
    p["playerId"] = p.playerId.astype("int64")
    return s, team_games, p.drop_duplicates("playerId")


def player_seasons(s, team_games):
    """One row per player per season he was on a list: games played, team games, rate."""
    main_team = (s.groupby(["playerId", "year", "team"]).size().rename("n").reset_index()
                 .sort_values("n").drop_duplicates(["playerId", "year"], keep="last"))
    ps = s.groupby(["playerId", "year"]).agg(games=("played", "sum"), dnf=("dnf", "sum"),
                                              avg=("fantasyScore", "mean"), ho=("hitOuts", "mean")).reset_index()
    ps = ps.merge(main_team[["playerId", "year", "team"]], on=["playerId", "year"])
    ps["team_games"] = ps.set_index(["year", "team"]).index.map(team_games)
    ps["rate"] = ps.games / ps.team_games
    return ps


def features(ps, players, N):
    """Pre-season availability features for season N from seasons before N.
    A season the player missed entirely (on a list, 0 games) counts as rate 0."""
    hist = ps[ps.year < N]
    first = hist.groupby("playerId").year.min()
    f = pd.DataFrame(index=first.index)
    f["seasons"] = N - first
    for k in (1, 2, 3):
        r = hist[hist.year == N - k].set_index("playerId").rate
        # on a list that year (played before and after) but no games -> 0
        f[f"rate{k}"] = r.reindex(f.index).where(f.seasons >= k, np.nan)
        f.loc[f[f"rate{k}"].isna() & (f.seasons >= k), f"rate{k}"] = 0.0
    g = hist.groupby("playerId")
    f["career_rate"] = g.games.sum() / (f.seasons * 22)
    f["prev1_avg"] = hist[hist.year == N - 1].set_index("playerId").avg.reindex(f.index)
    f["dnf1"] = hist[hist.year == N - 1].set_index("playerId").dnf.reindex(f.index).fillna(0)
    f["dnf_career"] = g.dnf.sum() / g.games.sum().clip(lower=1)
    f["ruck"] = (hist[hist.year >= N - 2].groupby("playerId").ho.mean().reindex(f.index).fillna(0) >= 12).astype(int)
    f["ever_full"] = (g.rate.max() >= 0.9).astype(int)
    f["career_games"] = g.games.sum()
    f = f.reset_index().merge(players, on="playerId", how="left")
    f["age"] = (pd.Timestamp(f"{N}-03-15") - f.dob).dt.days / 365.25
    f["season"] = N
    return f.drop(columns="dob")


def design(d):
    x = d[COLS].copy()
    for c in ("rate2", "rate3"):
        x[c] = x[c].fillna(x.rate1)
    x["prev1_avg"] = x.prev1_avg.fillna(60)
    x["age"] = x.age.fillna(25)
    x["age2"] = (x.age - 27) ** 2
    return x


def main():
    s, team_games, players = load_games()
    ps = player_seasons(s, team_games)
    last_year = ps.groupby("playerId").year.max()
    rows = []
    for N in range(2008, 2027):
        f = features(ps, players, N)
        f = f[(f.career_games >= 5) & (f.playerId.map(last_year) >= N)]   # still on a list in N
        tgt = ps[ps.year == N].set_index("playerId")
        f["games"] = f.playerId.map(tgt.games).fillna(0)
        team = f.playerId.map(tgt.team)
        f["team_games"] = f.playerId.map(tgt.team_games).fillna(team_games.xs(N).median())
        f["rate"] = f.games / f.team_games
        rows.append(f)
    data = pd.concat(rows, ignore_index=True)

    out = []
    for N in TEST_YEARS:
        tr, te = data[(data.season < N) & (data.season != 2020)], data[data.season == N].copy()
        te["naive_last"] = te.rate1.fillna(tr.rate.mean())
        te["naive_mean"] = tr.rate.mean()
        te["ridge"] = Ridge(alpha=1.0).fit(design(tr), tr.rate).predict(design(te)).clip(0, 1)
        hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                            min_samples_leaf=50, random_state=0)
        te["gbm"] = hgb.fit(design(tr), tr.rate).predict(design(te)).clip(0, 1)
        out.append(te)
    res = pd.concat(out, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "expected_games_predictions.csv", index=False)

    ex = res[res.season != 2020]
    draft = ex[ex.prev1_avg >= 80]
    for label, d in (("all listed players", ex), ("prev avg 80+ (draft targets)", draft)):
        print(f"{label}  (n={len(d)}, actual rate {d.rate.mean():.2f})")
        for m in ("naive_mean", "naive_last", "ridge", "gbm"):
            e = (d[m] - d.rate) * d.team_games
            print(f"  {m:11s} MAE {e.abs().mean():.2f} games   bias {e.mean():+.2f}")
    d = draft.assign(bin=pd.cut(draft.gbm, [0, .5, .7, .8, .9, 1]))
    print(d.groupby("bin", observed=True).agg(n=("rate", "size"), predicted=("gbm", "mean"), actual=("rate", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
