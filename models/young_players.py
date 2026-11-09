"""Young / new players: draftees from the last three drafts with fewer than 5 valid AFL games.

Two questions, both answered pre-season and backtested 2012-2026:
  1. Will he play at least 5 valid games this season?          (logistic regression)
  2. If he does, what will he average?                            (ridge regression)
Expected value = P(plays) x projected average, used for late-draft fliers.

Draftees who never played AFL have no playerId; they stay in the pool for
question 1 as "didn't play", so the playing rate isn't inflated.

Run:  python3 models/young_players.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import load, market_share_features, team_for_season, ROOT, CLEAN  # noqa: E402

OUT = ROOT / "models" / "output"
TEST_YEARS = range(2012, 2027)
MIN_GAMES = 5


def draft_pool(drafts, v, N):
    """Each draftee's most recent draft before season N, if within 3 drafts and < 5 valid games before N."""
    d = drafts[(drafts.draftYear <= N - 1) & (drafts.draftYear >= N - 3)
               & (drafts.fromLevel != "AFL (re-listed)")].copy()
    d["key"] = d.playerId.astype("Int64").astype(str).where(d.playerId.notna(), d.player + "|" + d.draftYear.astype(str))
    d = d.sort_values("draftYear").drop_duplicates("key", keep="last")
    prior = v[v.year < N].groupby("playerId").agg(prior_games=("fantasyScore", "size"), prior_avg=("fantasyScore", "mean"))
    d = d.merge(prior, left_on="playerId", right_index=True, how="left")
    d["prior_games"] = d.prior_games.fillna(0)
    d = d[d.prior_games < MIN_GAMES]
    d["years_since"] = N - d.draftYear
    d["season"] = N
    return d


def design(d):
    x = pd.DataFrame(index=d.index)
    x["log_pick"] = np.log(d.pick)
    x["national"] = (d.draftType == "national").astype(int)
    x["rookie"] = d.draftType.str.contains("rookie").astype(int)
    x["state_league"] = (d.fromLevel == "state league").astype(int)
    x["years_since"] = d.years_since
    x["yr1"] = (d.years_since == 1).astype(int)
    x["prior_games"] = d.prior_games
    x["prior_avg"] = d.prior_avg.fillna(0)
    x["has_prior"] = (d.prior_games > 0).astype(int)
    x["team_departed"] = d.team_departed_share.fillna(d.team_departed_share.median())
    return x


def design_avg(d):
    x = design(d)
    x["age"] = d.age.fillna(d.age.median())
    x["prior_avg_w"] = d.prior_avg.fillna(0) * d.prior_games / 4   # trust prior average more with more games
    return x


OTHER_LEAGUES = ["VFL", "SANFL", "WAFL", "Coates Talent League"]


def other_league(season_rows):
    """Previous-season state league / VFL / Talent League average (DFS: 2022-2025, so seasons 2023-2026)."""
    o = pd.read_csv(CLEAN / "dfs_other_leagues.csv")
    o = o[o.league.isin(OTHER_LEAGUES) & (o.games >= 4)].copy()
    o["season"] = o.statYear + 1
    o = o.sort_values("games").drop_duplicates(["playerId", "season"], keep="last")
    o = o.rename(columns={"avg": "ol_avg", "games": "ol_games"})[["playerId", "season", "ol_avg", "ol_games", "league"]]
    return season_rows.merge(o, on=["playerId", "season"], how="left")


def ol_design(d):
    return pd.DataFrame({"proj": d.proj_avg, "ol": d.ol_avg,
                         "ol_talent": d.ol_avg * (d.league == "Coates Talent League")})


def add_other_league(res):
    """Blend the draft-based projection with the other-league average where we have one.
    Rolling: each season's blend is fitted only on earlier seasons that had the data."""
    res = other_league(res)
    res["proj_avg_ol"] = res.proj_avg
    for N in sorted(res.season.unique()):
        tr = res[(res.season < N) & (res.played == 1) & res.ol_avg.notna()]
        te = (res.season == N) & res.ol_avg.notna()
        if len(tr) < 40 or not te.any():
            continue
        m = Ridge(alpha=1.0).fit(ol_design(tr), tr.actual, sample_weight=tr.games)
        res.loc[te, "proj_avg_ol"] = m.predict(ol_design(res[te]))
    return res


def main():
    v, s, players = load()
    drafts = pd.read_csv(CLEAN / "drafts.csv")
    rows = []
    for N in range(2008, 2027):
        d = draft_pool(drafts, v, N)
        tgt = v[v.year == N].groupby("playerId").fantasyScore.agg(games="size", actual="mean")
        d = d.merge(tgt, left_on="playerId", right_index=True, how="left")
        d["games"] = d.games.fillna(0)
        d["played"] = (d.games >= MIN_GAMES).astype(int)
        # opportunity at the club he's listed at: share of points that left the club before season N
        tN = team_for_season(s, N)
        ms = market_share_features(s, N, tN).merge(tN, on="playerId")
        d["team_departed_share"] = d.club.map(ms.groupby("team").team_departed_share.first())
        d = d.merge(players, on="playerId", how="left")
        d["age"] = (pd.Timestamp(f"{N}-03-15") - d.dob).dt.days / 365.25
        rows.append(d.drop(columns="dob"))
    data = pd.concat(rows, ignore_index=True)

    out = []
    for N in TEST_YEARS:
        tr, te = data[data.season < N], data[data.season == N].copy()
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(design(tr), tr.played)
        te["p_play"] = clf.predict_proba(design(te))[:, 1]
        trp = tr[tr.played == 1]
        reg = Ridge(alpha=1.0).fit(design_avg(trp), trp.actual, sample_weight=trp.games)
        te["proj_avg"] = reg.predict(design_avg(te))
        te["naive_avg"] = np.average(trp.actual, weights=trp.games)
        te["naive_p"] = tr.played.mean()
        out.append(te)
    res = pd.concat(out, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "young_backtest_predictions.csv", index=False)

    res = add_other_league(res)
    res.to_csv(OUT / "young_backtest_predictions.csv", index=False)
    ex = res[res.season != 2020]
    pl = ex[ex.played == 1]
    print(f"pool: {len(ex)} draftee-seasons, {ex.played.mean():.0%} played {MIN_GAMES}+ valid games")
    print(f"P(play)  AUC {roc_auc_score(ex.played, ex.p_play):.3f}   Brier {brier_score_loss(ex.played, ex.p_play):.4f}"
          f"  (naive Brier {brier_score_loss(ex.played, ex.naive_p):.4f})")
    for label, sub in (("all seasons", pl), ("2024-26 (other-league data)", pl[pl.season >= 2024])):
        print(label)
        for m in ("naive_avg", "proj_avg", "proj_avg_ol"):
            e = sub[m] - sub.actual
            print(f"  avg {m:11s} MAE {np.average(e.abs(), weights=sub.games):.2f}  "
                  f"spearman {sub[[m, 'actual']].corr('spearman').iloc[0, 1]:.3f}")
    # calibration: does a 60% chance really play 60% of the time?
    ex = ex.assign(bin=pd.cut(ex.p_play, [0, .1, .2, .35, .5, .7, 1]))
    print(ex.groupby("bin", observed=True).agg(n=("played", "size"), predicted=("p_play", "mean"), actual=("played", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
