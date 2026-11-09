"""2027 pre-season projections: average, expected games and season total for every listed player.

Established players (5+ valid games): season model v2_market + expected games model.
Young / new players (< 5 valid games, drafted 2023-2025): P(plays 5+ games) and projected average.
2026 national / rookie draftees are added once the drafts are in data/clean/drafts.csv.

Fantasy positions are the 2026 start-of-season positions (DFS) until 2027 positions are released.

Run:  python3 models/project_2027.py   ->  models/output/projections_2027.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge, LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F                                         # noqa: E402
import season_backtest as SB                                 # noqa: E402
import expected_games as EG                                  # noqa: E402
import young_players as YP                                   # noqa: E402

N = 2027
GAMES_2027 = 23


def listed_2027():
    p = pd.read_csv(F.CLEAN / "players.csv")
    p = p[p.active2027 == True].copy()  # noqa: E712
    p["playerId"] = p.playerId.astype("Int64")
    return p


def established(v, s, players, listed):
    feats_path = F.ROOT / "models" / "output" / "season_features.csv"
    train = pd.read_csv(feats_path) if feats_path.exists() else SB.build(v, s, players)[0]
    model = SB.fit_ridge(train, ruck=True, market=True, down_2020=True)

    team_N = listed.dropna(subset=["playerId"])[["playerId", "team2027"]].rename(columns={"team2027": "team"})
    team_N["playerId"] = team_N.playerId.astype("int64")
    f = F.season_features(v, s, players, N, team_N=team_N)
    f = f[(f.career_games >= SB.MIN_GAMES_HISTORY) & f.playerId.isin(team_N.playerId)].copy()
    f["proj_avg"] = model.predict(F.design(f, ruck=True, market=True))

    # expected games
    gs, team_games, gp = EG.load_games()
    ps = EG.player_seasons(gs, team_games)
    eg_rows = []
    last_year = ps.groupby("playerId").year.max()
    for y in range(2008, 2027):
        e = EG.features(ps, gp, y)
        e = e[(e.career_games >= 5) & (e.playerId.map(last_year) >= y) & (e.season != 2020)]
        t = ps[ps.year == y].set_index("playerId")
        e["rate"] = (e.playerId.map(t.games).fillna(0) / e.playerId.map(t.team_games)).fillna(0)
        eg_rows.append(e)
    eg_train = pd.concat(eg_rows)
    eg_model = Ridge(alpha=1.0).fit(EG.design(eg_train), eg_train.rate)
    e27 = EG.features(ps, gp, N)
    e27["avail"] = eg_model.predict(EG.design(e27)).clip(0, 1)
    f = f.merge(e27[["playerId", "avail"]], on="playerId", how="left")
    f["exp_games"] = (f.avail * GAMES_2027).round(1)
    f["group"] = "established"
    return f


def young(v, s, players, listed):
    drafts = pd.read_csv(F.CLEAN / "drafts.csv")
    hist = []
    for y in range(2008, 2027):
        d = YP.draft_pool(drafts, v, y)
        tgt = v[v.year == y].groupby("playerId").fantasyScore.agg(games="size", actual="mean")
        d = d.merge(tgt, left_on="playerId", right_index=True, how="left")
        d["games"] = d.games.fillna(0)
        d["played"] = (d.games >= YP.MIN_GAMES).astype(int)
        tN = F.team_for_season(s, y)
        ms = F.market_share_features(s, y, tN).merge(tN, on="playerId")
        d["team_departed_share"] = d.club.map(ms.groupby("team").team_departed_share.first())
        d = d.merge(players, on="playerId", how="left")
        d["age"] = (pd.Timestamp(f"{y}-03-15") - d.dob).dt.days / 365.25
        hist.append(d.drop(columns="dob"))
    hist = pd.concat(hist)
    clf = LogisticRegression(max_iter=2000).fit(YP.design(hist), hist.played)
    pl = hist[hist.played == 1]
    reg = Ridge(alpha=1.0).fit(YP.design_avg(pl), pl.actual, sample_weight=pl.games)

    d = YP.draft_pool(drafts, v, N)
    # undebuted draftees have no playerId: match to the 2027 list by "Last, First"
    name = d.player.str.split(" ", n=1).map(lambda x: f"{x[1]}, {x[0]}" if len(x) == 2 else x[0])
    undebuted = listed[listed.playerId.isna()].set_index("displayName")
    d["onList"] = d.playerId.isin(listed.playerId.dropna()) | (d.playerId.isna() & name.isin(undebuted.index))
    d = d[d.onList].copy()
    d["displayName"] = name
    d["team2027"] = d.playerId.map(listed.dropna(subset=["playerId"]).set_index("playerId").team2027)
    d["team2027"] = d.team2027.fillna(d.displayName.map(undebuted.team2027))
    tN = listed.dropna(subset=["playerId"])[["playerId", "team2027"]].rename(columns={"team2027": "team"})
    tN["playerId"] = tN.playerId.astype("int64")
    ms = F.market_share_features(s, N, tN).merge(tN, on="playerId")
    d["team_departed_share"] = d.team2027.map(ms.groupby("team").team_departed_share.first())
    d = d.merge(players, on="playerId", how="left")
    d["age"] = (pd.Timestamp(f"{N}-03-15") - d.dob).dt.days / 365.25
    d["p_play"] = clf.predict_proba(YP.design(d))[:, 1]
    d["proj_avg"] = reg.predict(YP.design_avg(d))
    # no 2026 state-league averages yet (DFS releases these pre-season); blend added when they arrive
    d["exp_games"] = (d.p_play * 12).round(1)     # rough: a young player who plays gets ~12 games
    d["group"] = "young"
    return d


def main():
    v, s, players = F.load()
    listed = listed_2027()
    est = established(v, s, players, listed)
    yng = young(v, s, players, listed)
    names = listed.dropna(subset=["playerId"]).set_index("playerId")
    est["displayName"] = est.playerId.map(names.displayName)
    est["team2027"] = est.playerId.map(names.team2027)

    pos = pd.read_csv(F.CLEAN / "dfs_preseason.csv")
    pos = pos[(pos.season == 2026) & pos.playerId.notna()].drop_duplicates("playerId").set_index("playerId").position

    cols = ["playerId", "displayName", "team2027", "group", "age", "prev1_avg", "proj_avg", "exp_games", "p_play"]
    est["p_play"] = np.nan
    yng["prev1_avg"] = yng.prior_avg
    out = pd.concat([est[cols], yng[cols]], ignore_index=True)
    out["playerId"] = out.playerId.astype("Int64")
    out["position_2026"] = out.playerId.map(pos)
    out["season_total"] = (out.proj_avg * out.exp_games).round(0)
    out = out.round({"age": 1, "prev1_avg": 1, "proj_avg": 1, "p_play": 2})
    out = out.sort_values("season_total", ascending=False)
    path = F.ROOT / "models" / "output" / "projections_2027.csv"
    out.to_csv(path, index=False)
    pd.set_option("display.width", 160)
    print(f"{len(out)} players ({(out.group == 'established').sum()} established, {(out.group == 'young').sum()} young)")
    print(out.head(25).to_string(index=False))
    for p in ("RUC", "DEF", "FWD"):
        print(f"\ntop {p}-eligible by average")
        print(out[out.position_2026.fillna("").str.contains(p)].sort_values("proj_avg", ascending=False)
              .head(8)[["displayName", "position_2026", "age", "prev1_avg", "proj_avg", "exp_games"]].to_string(index=False))


if __name__ == "__main__":
    main()
