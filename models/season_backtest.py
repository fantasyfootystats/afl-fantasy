"""Pre-season projection backtest.

For every season N in TEST_YEARS, build each player's features from games
before season N only, predict their season-N average, and score against the
actual average over valid games (TOG >= 25, no sub on/off). Models are
trained only on seasons before N (rolling origin), so nothing leaks.

Models compared:
  last_season  last season's average (naive)
  matt         Matt's method: per-game regression on 13-game, 5-game, venue,
               opposition and home/away averages, frozen at pre-season
  v1           ridge on season features
  v2_ruck      v1 + ruck role and ruck-specific age curve, 2020 down-weighted
  v2_market    v2_ruck + market share and team departures/arrivals

Run:  python3 models/season_backtest.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import load, season_features, season_targets, design, ROOT  # noqa: E402

OUT = ROOT / "models" / "output"
FIRST_FEATURE_YEAR = 2008
TEST_YEARS = range(2012, 2027)
MIN_GAMES_TARGET = 5        # player-season must have this many valid games to be scored
MIN_GAMES_HISTORY = 5       # and this many prior valid games (young players have their own model)
WEIGHT_2020 = 0.4           # 2020 scores were scaled up from 16-minute quarters: noisier
MATT_COLS = ["avg13", "avg5", "venue_avg", "opposition_avg", "homeaway_avg"]


def matt_game_rows(v, feats, N):
    """One row per game a player played in season N, with Matt's five inputs as of pre-season."""
    hist = v[v.year < N]
    games = v[v.year == N][["playerId", "venue", "opposition", "homeaway", "fantasyScore"]]
    games = games.merge(feats[["playerId", "avg13", "avg5"]], on="playerId", how="inner")
    for key in ("venue", "opposition", "homeaway"):
        split = hist.groupby(["playerId", key]).fantasyScore.mean().rename(f"{key}_avg").reset_index()
        games = games.merge(split, on=["playerId", key], how="left")
        games[f"{key}_avg"] = games[f"{key}_avg"].fillna(games.avg13)
    games["season"] = N
    return games


def build(v, s, players):
    rows, game_rows = [], []
    for N in range(FIRST_FEATURE_YEAR, 2027):
        f = season_features(v, s, players, N)
        f = f[f.career_games >= MIN_GAMES_HISTORY]
        ft = f.merge(season_targets(v, N), on="playerId", how="inner")
        ft = ft[ft.games >= MIN_GAMES_TARGET]
        rows.append(ft)
        game_rows.append(matt_game_rows(v, ft, N))
        print(f"features {N}: {len(ft)}", flush=True)
    return pd.concat(rows, ignore_index=True), pd.concat(game_rows, ignore_index=True)


def fit_ridge(train, ruck, market, down_2020):
    w = train.games * np.where((train.season == 2020) & down_2020, WEIGHT_2020, 1.0)
    return Ridge(alpha=1.0).fit(design(train, ruck, market), train.actual, sample_weight=w)


VARIANTS = {"v1": (False, False, False), "v2_ruck": (True, False, True), "v2_market": (True, True, True)}


def main():
    v, s, players = load()
    data, games = build(v, s, players)
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "season_features.csv", index=False)

    preds = []
    for N in TEST_YEARS:
        train, test = data[data.season < N], data[data.season == N].copy()
        test["last_season"] = test.prev1_avg.fillna(test.avg13)
        gtr, gte = games[games.season < N], games[games.season == N].copy()
        gte["p"] = LinearRegression().fit(gtr[MATT_COLS], gtr.fantasyScore).predict(gte[MATT_COLS])
        test = test.merge(gte.groupby("playerId").p.mean().rename("matt").reset_index(), on="playerId", how="left")
        for name, (ruck, market, d20) in VARIANTS.items():
            test[name] = fit_ridge(train, ruck, market, d20).predict(design(test, ruck, market))
        preds.append(test)
    res = pd.concat(preds, ignore_index=True)
    res.to_csv(OUT / "season_backtest_predictions.csv", index=False)
    summarise(res)


MODELS = ["last_season", "matt", "v1", "v2_ruck", "v2_market"]


def score(d, label):
    out = []
    for m in MODELS:
        e = d[m] - d.actual
        out.append({"subset": label, "model": m, "n": len(d),
                    "MAE": np.average(e.abs(), weights=d.games),
                    "RMSE": np.sqrt(np.average(e ** 2, weights=d.games)),
                    "bias": np.average(e, weights=d.games),
                    "spearman": d[[m, "actual"]].corr("spearman").iloc[0, 1]})
    return out


def summarise(res):
    ex20 = res[res.season != 2020]
    draft = ex20[(ex20.prev1_avg >= 75) | (ex20.actual >= 75)]
    rucks = ex20[ex20.ruck == 1]
    big_change = ex20[ex20.team_departed_share >= ex20.team_departed_share.quantile(0.8)]
    summary = pd.DataFrame(score(ex20, "all (ex 2020)") + score(draft, "draft pool 75+")
                           + score(rucks, "rucks") + score(big_change, "teams losing most share"))
    summary.to_csv(OUT / "season_backtest_summary.csv", index=False)
    pd.set_option("display.width", 160)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
