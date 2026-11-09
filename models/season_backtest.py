"""Pre-season projection backtest.

For every season N in TEST_YEARS, build each player's features from games
before season N only, predict their season-N average, and score against the
actual average over valid games (TOG >= 25, no sub on/off). Models are
trained only on seasons before N (rolling origin), so nothing leaks.

Run:  python3 models/season_backtest.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "models" / "output"
FIRST_FEATURE_YEAR = 2008   # need a few seasons of history before features are meaningful
TEST_YEARS = range(2012, 2027)
MIN_GAMES_TARGET = 5        # player-season must have this many valid games to be scored
MIN_GAMES_HISTORY = 5       # and this many prior valid games (rookies are a separate model)


def load():
    v = pd.read_csv(CLEAN / "validGames.csv", parse_dates=["date"])
    v = v.sort_values(["date", "gameId"]).reset_index(drop=True)
    v["ppm"] = v.fantasyScore / v.pctPlayed  # points per % of game played
    p = pd.read_csv(CLEAN / "players.csv", usecols=["playerId", "dob"], parse_dates=["dob"])
    p = p.dropna(subset=["playerId"]).drop_duplicates("playerId")
    p["playerId"] = p.playerId.astype("int64")
    return v, p


def split_avg(hist, key):
    """Player's career-to-date average by venue / opposition / home-away."""
    return hist.groupby(["playerId", key]).fantasyScore.mean().rename(f"{key}_avg").reset_index()


def season_features(v, players, N):
    """Features for every player who has history before season N."""
    hist = v[v.year < N]
    g = hist.groupby("playerId")
    f = pd.DataFrame({
        "avg13": g.tail(13).groupby("playerId").fantasyScore.mean(),
        "avg5": g.tail(5).groupby("playerId").fantasyScore.mean(),
        "career_avg": g.fantasyScore.mean(),
        "career_games": g.size(),
        "seasons": g.year.nunique(),
        "tog13": g.tail(13).groupby("playerId").pctPlayed.mean(),
        "ppm13": g.tail(13).groupby("playerId").ppm.mean(),
    })
    for k in (1, 2, 3):
        s = hist[hist.year == N - k].groupby("playerId")
        f[f"prev{k}_avg"] = s.fantasyScore.mean()
        f[f"prev{k}_games"] = s.size()
        if k == 1:
            f["prev1_tog"] = s.pctPlayed.mean()
            f["prev1_ppm"] = s.ppm.mean()
            f["prev1_max"] = s.fantasyScore.max()
    for hl in (10, 25):
        f[f"ewm{hl}"] = g.fantasyScore.apply(lambda x: x.ewm(halflife=hl).mean().iloc[-1])
    f["ewm_ppm25"] = g.ppm.apply(lambda x: x.ewm(halflife=25).mean().iloc[-1])
    f["ewm_tog25"] = g.pctPlayed.apply(lambda x: x.ewm(halflife=25).mean().iloc[-1])
    f = f.reset_index()
    f = f.merge(players, on="playerId", how="left")
    f["age"] = (pd.Timestamp(f"{N}-03-15") - f.dob).dt.days / 365.25
    f = f.drop(columns="dob")
    for c in ("prev1_games", "prev2_games", "prev3_games"):
        f[c] = f[c].fillna(0)
    f["season"] = N
    return f


def season_targets(v, N):
    s = v[v.year == N].groupby("playerId")
    return pd.DataFrame({"actual": s.fantasyScore.mean(), "games": s.size()}).reset_index()


# ---------- Matt's method: per-game regression, inputs frozen at pre-season ----------
MATT_COLS = ["avg13", "avg5", "venue_avg", "opposition_avg", "homeaway_avg"]


def matt_game_rows(v, feats, N):
    """One row per game a player played in season N, with Matt's five inputs as of pre-season."""
    hist = v[v.year < N]
    games = v[v.year == N][["playerId", "venue", "opposition", "homeaway", "fantasyScore"]]
    games = games.merge(feats[["playerId", "avg13", "avg5"]], on="playerId", how="inner")
    for key in ("venue", "opposition", "homeaway"):
        games = games.merge(split_avg(hist, key), on=["playerId", key], how="left")
        games[f"{key}_avg"] = games[f"{key}_avg"].fillna(games.avg13)  # no history there: fall back
    games["season"] = N
    return games


def main():
    v, players = load()
    v = v[v.year >= FIRST_FEATURE_YEAR - 3]
    rows, game_rows = [], []
    for N in range(FIRST_FEATURE_YEAR, 2027):
        f = season_features(v, players, N)
        f = f[f.career_games >= MIN_GAMES_HISTORY]
        t = season_targets(v, N)
        ft = f.merge(t, on="playerId", how="inner")
        ft = ft[ft.games >= MIN_GAMES_TARGET]
        rows.append(ft)
        game_rows.append(matt_game_rows(v, ft, N))
        print(f"features {N}: {len(ft)} player-seasons", flush=True)
    data = pd.concat(rows, ignore_index=True)
    games = pd.concat(game_rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "season_features.csv", index=False)

    season_cols = ["avg13", "avg5", "career_avg", "career_games", "seasons", "tog13", "ppm13",
                   "prev1_avg", "prev1_games", "prev2_avg", "prev2_games", "prev3_avg", "prev3_games",
                   "prev1_tog", "prev1_ppm", "prev1_max", "ewm10", "ewm25", "ewm_ppm25", "ewm_tog25", "age"]

    preds = []
    for N in TEST_YEARS:
        train, test = data[data.season < N], data[data.season == N].copy()

        # Naive references
        test["last_season"] = test.prev1_avg.fillna(test.avg13)
        test["avg13_only"] = test.avg13

        # Matt's method: linear regression on per-game rows, then average each player's games
        gtr, gte = games[games.season < N], games[games.season == N].copy()
        lr = LinearRegression().fit(gtr[MATT_COLS], gtr.fantasyScore)
        gte["p"] = lr.predict(gte[MATT_COLS])
        test = test.merge(gte.groupby("playerId").p.mean().rename("matt").reset_index(), on="playerId", how="left")

        # Ridge on season features (missing filled with a simple fallback)
        def fill(d):
            d = d[season_cols].copy()
            for c in ("prev1_avg", "prev2_avg", "prev3_avg"):
                d[c] = d[c].fillna(d.career_avg)
            d["prev1_tog"] = d.prev1_tog.fillna(d.tog13)
            d["prev1_ppm"] = d.prev1_ppm.fillna(d.ppm13)
            d["prev1_max"] = d.prev1_max.fillna(d.career_avg)
            d["age"] = d.age.fillna(d.age.median() if d.age.notna().any() else 25)
            d["age2"] = (d.age - 25) ** 2
            return d
        rg = Ridge(alpha=1.0).fit(fill(train), train.actual, sample_weight=train.games)
        test["ridge"] = rg.predict(fill(test))

        # Gradient boosting (handles missing values and non-linear age curves itself)
        hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                            min_samples_leaf=40, random_state=0)
        hgb.fit(train[season_cols], train.actual, sample_weight=train.games)
        test["gbm"] = hgb.predict(test[season_cols])
        preds.append(test)
        print(f"tested {N}", flush=True)

    res = pd.concat(preds, ignore_index=True)
    res.to_csv(OUT / "season_backtest_predictions.csv", index=False)
    models = ["last_season", "avg13_only", "matt", "ridge", "gbm"]

    def score(d, label):
        out = []
        for m in models:
            e = d[m] - d.actual
            out.append({
                "subset": label, "model": m, "n": len(d),
                "MAE": np.average(e.abs(), weights=d.games),
                "RMSE": np.sqrt(np.average(e ** 2, weights=d.games)),
                "bias": np.average(e, weights=d.games),
                "spearman": d[[m, "actual"]].corr("spearman").iloc[0, 1],
            })
        return out

    # Draft-relevant: players who averaged 75+ last season or ended up 75+ (the pool that gets drafted)
    draft = res[(res.prev1_avg >= 75) | (res.actual >= 75)]
    summary = pd.DataFrame(score(res, "all") + score(draft, "draft pool (75+)"))
    by_year = (res.assign(**{m + "_ae": (res[m] - res.actual).abs() for m in models})
                 .groupby("season").apply(lambda d: pd.Series({m: np.average(d[m + "_ae"], weights=d.games) for m in models})))
    summary.to_csv(OUT / "season_backtest_summary.csv", index=False)
    by_year.to_csv(OUT / "season_backtest_by_year.csv")
    pd.set_option("display.width", 160)
    print(summary.round(3).to_string(index=False))
    print(by_year.round(2).to_string())


if __name__ == "__main__":
    main()
