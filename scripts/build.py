"""Build the clean AFL fantasy tables from Matt's raw exports plus back-fill sources.

Run from the repo root:  python3 scripts/build.py

Inputs  (never edited):  data/raw/*.csv, data/sources/*.csv
Outputs (regenerated):   data/clean/{players,games,stats,validGames,player_id_aliases}.csv
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, SRC, OUT = ROOT / "data/raw", ROOT / "data/sources", ROOT / "data/clean"
REF = ROOT / "data/reference"

# validGames rule; TOG threshold will become per-game-version (Classic / WAFL / Shiva) once 2027 rules are known
VALID_MIN_TOG = 25
OUT.mkdir(parents=True, exist_ok=True)

TEAM_FIX = {"Brisbane Lions": "Brisbane", "Kangaroos": "North Melbourne"}

# Same player recorded under two IDs, or a mistyped ID -> the ID we keep.
ID_ALIASES = {
    2022020701: 2014765737,  # Angus Litherland -> Angus Dewar (name change, delisted & re-drafted)
    2023010801: 2018719221,  # Fergus Greene (Western Bulldogs 2018, Hawthorn 2023)
    2022020502: 2017833334,  # Sam Skinner (Brisbane, then Port Adelaide)
    20251780201: 2025170201,  # Malakai Champion, mistyped 11-digit ID
    2026090901: 2026090101,  # Bodie Ryan: debuted game 1 of R09 2026 (players table had 09)
}
PREFERRED_NAME = {2014765737: "Dewar, Angus"}

# Rows filed under the wrong player (two players sharing a name): (old id, team, year) -> correct id
ROW_REASSIGN = [
    (2010763943, "Richmond", 2022, 2011769877),  # Tom Lynch (Richmond, b.1992) had 2022 games under Tom Lynch (Adelaide, b.1990)
    (2011665405, "West Coast", 2022, 2022020704),  # Aaron Black (West Coast, b.1992), 1 game; not the North/Geelong Aaron Black
]
NEW_PLAYERS = [  # players created by ROW_REASSIGN
    {"playerId": 2022020704, "displayName": "Black, Aaron", "team": "West Coast", "dob": "1992-12-25", "active": False},
]

STAT_COLS = {
    "Disposals": "disposals", "Kicks": "kicks", "Marks": "marks", "Handballs": "handballs",
    "Goals": "goals", "Behinds": "behinds", "Hit Outs": "hitOuts", "Tackles": "tackles",
    "Frees": "freesFor", "Frees Against": "freesAgainst",
}
AF_WEIGHTS = {"kicks": 3, "handballs": 2, "marks": 3, "tackles": 4, "freesFor": 1,
              "freesAgainst": -3, "hitOuts": 1, "goals": 6, "behinds": 1}


def af_score(df):
    return sum(df[c].fillna(0) * w for c, w in AF_WEIGHTS.items())


def norm(x):
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", x)


def fix_id(series):
    return series.map(lambda i: ID_ALIASES.get(i, i))


# --------------------------------------------------------------------------- games
def build_games():
    g = pd.read_csv(RAW / "games_2026.csv")
    iso = g["date"].str.match(r"^\d{4}-\d{2}-\d{2}$")  # 2025+ rows are ISO, earlier rows are d-Mon-YYYY
    g["date"] = pd.to_datetime(g["date"].where(iso), format="%Y-%m-%d").fillna(
        pd.to_datetime(g["date"].where(~iso), format="%d-%b-%Y")).dt.strftime("%Y-%m-%d")
    g["attendance"] = pd.to_numeric(g["attendance"].astype(str).str.replace(",", ""), errors="coerce")
    g = g.rename(columns={"rainfall (mm)": "rainfallMm"})
    g["source"] = "matt"

    # back-fill 2005-2011 regular-season games that are missing
    m = pd.read_csv(SRC / "afltables_matches_debut_years.csv")
    m = m[m.year.between(2005, 2011)]
    m = m[pd.to_numeric(m.round_num, errors="coerce").notna()].copy()
    m["rnd"] = m.round_num.astype(int)
    for t in ("team_1_team_name", "team_2_team_name"):
        m[t] = m[t].replace(TEAM_FIX)
    m["kickoff"] = pd.to_datetime(m["date"])
    m = m.sort_values(["year", "rnd", "kickoff"], kind="stable")
    have = g[g.year <= 2011].assign(rnd=lambda d: d["round"].str[1:].astype(int))
    have_keys = set(zip(have.year, have.rnd, have.homeTeam))
    new = []
    for (yr, rnd), grp in m.groupby(["year", "rnd"], sort=False):
        used = set(have[(have.year == yr) & (have.rnd == rnd)].game)
        free = (n for n in range(1, 20) if n not in used)
        for _, r in grp.iterrows():
            if (yr, rnd, r.team_1_team_name) in have_keys:
                continue
            n = next(free)
            new.append({
                "gameId": f"{yr}R{rnd:02d}{n:02d}", "year": yr, "round": f"R{rnd:02d}", "game": n,
                "date": r.kickoff.strftime("%Y-%m-%d"), "venue": r.venue, "startTime": r.kickoff.strftime("%H:%M"),
                "attendance": pd.to_numeric(str(r.attendance).replace(",", ""), errors="coerce"),
                "homeTeam": r.team_1_team_name,
                "homeTeamScore": r.team_1_final_goals * 6 + r.team_1_final_behinds,
                "awayTeam": r.team_2_team_name,
                "awayTeamScore": r.team_2_final_goals * 6 + r.team_2_final_behinds,
                "source": "afltables",
            })
    g = pd.concat([g, pd.DataFrame(new)], ignore_index=True)
    g = g.sort_values(["year", "round", "game"]).reset_index(drop=True)
    # attendance in the afltables match file is actually a local time for some rows; drop non-numeric
    return g


# --------------------------------------------------------------------------- stats
def load_matt_stats():
    s = pd.read_csv(RAW / "stats_2026.csv", low_memory=False)
    s = s.drop(columns=["Active"]).rename(columns=STAT_COLS).rename(columns={
        "% Played": "pctPlayed", "Subs": "subs", "AFL Fantasy Score": "fantasyScore"})
    s["playerId"] = fix_id(s.playerId)
    for old, team, yr, new in ROW_REASSIGN:
        s.loc[(s.playerId == old) & (s.team == team) & (s.year == yr), "playerId"] = new
    for pid, nm in PREFERRED_NAME.items():
        s.loc[s.playerId == pid, "displayName"] = nm
    s["homeaway"] = s.homeaway.str.lower()
    s["subs"] = s.subs.replace({"-": None})
    s["source"] = "matt"
    return s


def mint_ids(games, idmap):
    """Matt's ID format YYYYRRGGPP from each new player's debut game."""
    deb = pd.read_csv(SRC / "afltables_debut_games.csv")
    deb = deb[~deb.key.isin(idmap.key)].copy()
    m = pd.read_csv(SRC / "afltables_matches_debut_years.csv")
    m = m[pd.to_numeric(m.round_num, errors="coerce").notna()].copy()
    m["rnd"] = m.round_num.astype(int)
    for t in ("team_1_team_name", "team_2_team_name"):
        m[t] = m[t].replace(TEAM_FIX)
    m["kickoff"] = pd.to_datetime(m["date"])
    m = m.sort_values(["year", "rnd", "kickoff"], kind="stable")
    m["seq"] = m.groupby(["year", "rnd"]).cumcount() + 1
    # prefer Matt's own game numbers where the game is in games.csv
    gm = games.assign(rnd=games["round"].str[1:].astype(int))[["year", "rnd", "homeTeam", "awayTeam", "game"]]
    lookup = {}
    for _, r in m.iterrows():
        for t in (r.team_1_team_name, r.team_2_team_name):
            lookup[(r.year, r.rnd, t)] = r.seq
    for _, r in gm.iterrows():
        for t in (r.homeTeam, r.awayTeam):
            lookup[(r.year, r.rnd, t)] = r.game
    deb["rnd"] = pd.to_numeric(deb["round"], errors="coerce")
    deb["gameNo"] = [lookup.get((y, r, t)) for y, r, t in zip(deb.year, deb.rnd, deb.team)]
    assert deb.gameNo.notna().all(), deb[deb.gameNo.isna()]
    deb["base"] = deb.year.astype(str) + deb.rnd.astype(int).map("{:02d}".format) + deb.gameNo.astype(int).map("{:02d}".format)
    existing = set(idmap.playerId.astype(str))
    out = {}
    for base, grp in deb.sort_values("key").groupby("base"):
        n = 1
        for k in grp.key:
            while f"{base}{n:02d}" in existing:
                n += 1
            pid = f"{base}{n:02d}"
            existing.add(pid)
            out[k] = int(pid)
            n += 1
    return out


def opening_round_2026(games):
    """2026 Opening Round was missing from stats; AFL Tables (files it as Round 1). Matches Matt's validGames
    scores on 226 of 228 rows; McCartin and McInerney differ by 2 (one handball) from his source."""
    r = pd.read_csv(SRC / "afltables_2026_opening_round.csv")
    g = games.set_index("gameId")
    r["year"], r["round"] = 2026, "R00"
    r["game"] = r.gameId.str[-2:].astype(int)
    home = r.team == r.gameId.map(g.homeTeam)
    r["homeaway"] = home.map({True: "home", False: "away"})
    r["opposition"] = r.gameId.map(g.awayTeam).where(home, r.gameId.map(g.homeTeam))
    r["fantasyScore"] = af_score(r).astype(int)
    r["subs"] = None
    r["source"] = "afltables"
    return r


def build_stats(games):
    s = load_matt_stats()
    idmap = pd.read_csv(SRC / "afltables_id_map.csv")
    idmap["playerId"] = fix_id(idmap.playerId)
    idmap = idmap.drop_duplicates("key")
    new_ids = mint_ids(games, idmap)
    key2id = dict(zip(idmap.key, idmap.playerId)) | new_ids

    det = pd.read_csv(SRC / "afltables_player_details.csv")
    det["displayName"] = det.last_name + ", " + det.first_name
    key2name = dict(zip(det.key, det.displayName))

    a = pd.read_csv(SRC / "afltables_player_games_2005_2011.csv")
    a = a[pd.to_numeric(a["round"], errors="coerce").notna()].copy()
    a["rnd"] = a["round"].astype(int)
    gm = games[games.year <= 2011].assign(rnd=lambda d: d["round"].str[1:].astype(int))
    home = gm[["gameId", "year", "rnd", "homeTeam", "venue"]].rename(columns={"homeTeam": "team"}).assign(homeaway="home")
    away = gm[["gameId", "year", "rnd", "awayTeam", "venue"]].rename(columns={"awayTeam": "team"}).assign(homeaway="away")
    a = a.merge(pd.concat([home, away]), on=["year", "rnd", "team"], how="left")
    assert a.gameId.notna().all(), a[a.gameId.isna()].head()
    a["playerId"] = a.key.map(key2id).astype("int64")
    a["displayName"] = a.key.map(key2name)
    a = a.rename(columns={"hit_outs": "hitOuts", "free_kicks_for": "freesFor", "free_kicks_against": "freesAgainst",
                          "percentage_of_game_played": "pctPlayed", "opponent": "opposition"})
    a["round"] = "R" + a.rnd.map("{:02d}".format)
    a["game"] = a.gameId.str[-2:].astype(int)
    a["fantasyScore"] = af_score(a).astype(int)
    # AFL Tables marks substitutes on the games-played count: up-arrow = sub on, down-arrow = sub off
    mark = a.games_played.astype(str).str.extract("([\u2191\u2193])")[0]
    a["subs"] = mark.map({"\u2191": "On", "\u2193": "Off"})
    a["source"] = "afltables"
    keep = ["gameId", "team", "year", "round", "game", "playerId", "displayName", *STAT_COLS.values(),
            "pctPlayed", "subs", "fantasyScore", "opposition", "homeaway", "source"]
    a = a[keep]

    # 2005-2011: drop Matt's back-filled rows the source doesn't confirm
    # (39 phantom Michael Hibberd rows 2009-11 that copy Michael Hurley's games)
    src_idx = a.set_index(["gameId", "playerId"]).index
    phantom = (s.year <= 2011) & ~s.set_index(["gameId", "playerId"]).index.isin(src_idx)
    print(f"dropped {phantom.sum()} unconfirmed 2005-2011 rows: {s[phantom].displayName.unique().tolist()}")
    s = s[~phantom]

    # hand-checked bad rows in Matt's data from 2012 on (data/reference/drop_rows.csv), e.g. a copied
    # stat line in a game the player didn't play. Found by scripts/check_identities.py
    drop = pd.read_csv(ROOT / "data/reference/drop_rows.csv", dtype={"gameId": str, "playerId": "int64"})
    bad = s.set_index(["gameId", "playerId"]).index.isin(drop.set_index(["gameId", "playerId"]).index)
    print(f"dropped {bad.sum()} hand-checked bad rows: {s[bad].displayName.tolist()}")
    s = s[~bad]

    # Matt's existing rows win; check they agree with the source where both exist
    both = s.merge(a, on=["gameId", "playerId"], suffixes=("", "_a"))
    mism = (both.fantasyScore != both.fantasyScore_a).sum()
    print(f"back-fill overlap with Matt's rows: {len(both)} rows, {mism} fantasy-score mismatches")
    # source sub flags are authoritative for 2005-2011 (they agree with all of Matt's real 2011 flags)
    flags = a.set_index(["gameId", "playerId"]).subs
    old = s.year <= 2011
    s.loc[old, "subs"] = flags.reindex(pd.MultiIndex.from_frame(s.loc[old, ["gameId", "playerId"]])).values
    a = a[~a.set_index(["gameId", "playerId"]).index.isin(s.set_index(["gameId", "playerId"]).index)]
    out = pd.concat([s, a, opening_round_2026(games)], ignore_index=True)
    out["pctPlayed"] = out.pctPlayed.astype("Int64")
    assert (af_score(out).round() == out.fantasyScore).all(), "fantasy score formula mismatch"
    return out.sort_values(["year", "round", "game", "team", "displayName"]).reset_index(drop=True), new_ids


# --------------------------------------------------------------------------- players
def build_players(stats, new_ids):
    p = pd.read_csv(RAW / "players_2026.csv", low_memory=False)
    p = p.drop(columns=["Lookup", "adjusted", "playerId.1"], errors="ignore")
    p["playerId"] = p.playerId.astype("Int64").map(lambda i: ID_ALIASES.get(i, i) if pd.notna(i) else i).astype("Int64")
    p = p.rename(columns={"2025 best23": "best23_2025", "2026 best23": "best23_2026", "2025 marketShare": "marketShare2025"})
    for pid, nm in PREFERRED_NAME.items():
        p.loc[p.playerId == pid, "displayName"] = nm

    # add a record for every player in stats who isn't in the table (mostly retired players)
    det = pd.read_csv(SRC / "afltables_player_details.csv")
    idmap = pd.read_csv(SRC / "afltables_id_map.csv")
    idmap["playerId"] = fix_id(idmap.playerId)
    key2id = dict(zip(idmap.key, idmap.playerId)) | new_ids
    det["playerId"] = det.key.map(key2id)
    dob = dict(zip(det.playerId, det.born_date))
    last = stats.sort_values(["year", "round"]).groupby("playerId").agg(
        displayName=("displayName", "last"), team=("team", "last"))
    p = pd.concat([p, pd.DataFrame(NEW_PLAYERS)], ignore_index=True)
    miss = last[~last.index.isin(p.playerId.dropna())].reset_index()
    miss["active"] = False
    miss["dob"] = miss.playerId.map(dob)
    p = pd.concat([p, miss], ignore_index=True)
    p["dob"] = p.dob.fillna(p.playerId.map(dob))

    # career average and games from the valid set happen downstream; here just debut/last season
    span = stats.groupby("playerId").year.agg(["min", "max"]).rename(columns={"min": "firstSeason", "max": "lastSeason"})
    p = p.merge(span, left_on="playerId", right_index=True, how="left")
    p["playerId"] = p.playerId.astype("Int64")
    assert p.playerId.dropna().is_unique
    return p


# --------------------------------------------------------------------------- valid games
def build_valid(stats, games):
    v = stats[(~stats.subs.isin(["On", "Off"])) & (stats.pctPlayed >= VALID_MIN_TOG)]
    v = v.merge(games[["gameId", "venue", "date"]], on="gameId", how="left")
    return v[["gameId", "year", "round", "playerId", "displayName", "team", "opposition", "homeaway",
              "venue", "date", "pctPlayed", "fantasyScore"]]


ZH_TEAMS = {"ADEL": "Adelaide", "BL": "Brisbane", "CARL": "Carlton", "COLL": "Collingwood", "ESS": "Essendon",
            "FRE": "Fremantle", "GEEL": "Geelong", "GC": "Gold Coast", "GWS": "Greater Western Sydney",
            "HAW": "Hawthorn", "MELB": "Melbourne", "NMFC": "North Melbourne", "PORT": "Port Adelaide",
            "RICH": "Richmond", "STK": "St Kilda", "SYD": "Sydney", "WCE": "West Coast", "WB": "Western Bulldogs"}
ZH_VENUES = ["SCG", "Optus Stadium", "MCG", "Adelaide Oval", "Marvel Stadium", "Gabba", "Corroboree Group Oval Manuka",
             "People First Stadium", "GMHBA Stadium", "Barossa Park", "Norwood Oval", "TIO Traeger Park", "ENGIE Stadium",
             "Mars Stadium", "Ninja Stadium", "UTAS Stadium", "TIO Stadium", "Hands Oval"]


def venue_lookup():
    v = pd.read_csv(REF / "venues.csv").fillna("")
    alias = {}
    for _, r in v.iterrows():
        for a in [r.venue, *[x for x in r.aliases.split("|") if x]]:
            alias[a.lower()] = r.venue
    return v.set_index("venue"), alias


def fixture_2027():
    """2027 fixture (Zero Hanger, read 6 Nov 2026). Times are converted from UTC to venue-local, matching how
    startTime is stored for earlier seasons. Rounds 17-24 had dates/times TBC when read."""
    vinfo, alias = venue_lookup()
    rows = []
    for line in (SRC / "zerohanger_fixture_2027.txt").read_text().splitlines():
        if line.startswith("#"):
            continue
        for tok in line.split():
            rnd, vc, h, a, ko = tok.split(",")
            venue = alias[ZH_VENUES[int(vc)].lower()]
            dt = None
            if ko:
                utc = pd.Timestamp(f"2027-{ko[:2]}-{ko[2:4]} {ko[4:6]}:{ko[6:]}", tz="UTC")
                dt = utc.tz_convert(vinfo.at[venue, "timezone"])
            rows.append({"round": int(rnd), "venue": venue, "homeTeam": ZH_TEAMS[h], "awayTeam": ZH_TEAMS[a],
                         "kickoffUtc": utc if ko else pd.NaT, "local": dt})
    f = pd.DataFrame(rows)
    f = f.sort_values(["round", "kickoffUtc"], kind="stable", na_position="last")
    f["game"] = f.groupby("round").cumcount() + 1
    f["year"] = 2027
    f["gameId"] = "2027R" + f["round"].map("{:02d}".format) + f.game.map("{:02d}".format)
    f["date"] = f.local.map(lambda d: d.strftime("%Y-%m-%d") if d is not None else None)
    f["startTime"] = f.local.map(lambda d: d.strftime("%H:%M") if d is not None else None)
    f["round"] = "R" + f["round"].map("{:02d}".format)
    f["source"] = "zerohanger"
    return f[["gameId", "year", "round", "game", "date", "venue", "startTime", "homeTeam", "awayTeam", "source"]]


def apply_list_changes(p):
    """2027 list changes (AFL.com.au retirements/delistings table + trade tracker, read 6 Nov 2026).
    Adds team2027 / status2027 / active2027; Matt's original team and active columns are left as they were."""
    lc = pd.read_csv(REF / "list_changes_2027.csv").fillna("")
    sp = p.displayName.str.split(", ", n=1)
    last, first = sp.str[0].map(norm), sp.str[1].fillna("").map(norm)
    p["team2027"] = p.team.where(p.active == True)
    p["status2027"] = p.active.map({True: "listed"}).astype("object")
    unmatched = []
    for _, r in lc.iterrows():
        fn, ln = r.player.replace(" jnr", "").split(" ", 1)
        c = p[(last == norm(ln)) & (p.team == r.fromTeam)]
        if len(c) > 1 and (c.active == True).sum() >= 1:
            c = c[c.active == True]
        if len(c) > 1:
            c = c[first[c.index].str[:1] == norm(fn)[:1]]
        if len(c) > 1:
            c = c[first[c.index].str[:3] == norm(fn)[:3]]
        if len(c) != 1:
            unmatched.append(r.player)
            continue
        i = c.index[0]
        status = r.change + (" (redraft pledged)" if r.redraftPledged else "")
        p.at[i, "status2027"] = status
        p.at[i, "team2027"] = r.toTeam or (r.fromTeam if r.redraftPledged else None)
    p["active2027"] = p.team2027.notna()
    print(f"list changes applied: {len(lc) - len(unmatched)}; not in players table (no AFL games): {unmatched}")
    return p


def main():
    games = build_games()
    stats, new_ids = build_stats(games)
    players = apply_list_changes(build_players(stats, new_ids))
    valid = build_valid(stats, games)
    games = pd.concat([games, fixture_2027()], ignore_index=True)
    _, alias = venue_lookup()
    unknown = set(games.venue.str.lower()) - set(alias)
    assert not unknown, f"venues missing from data/reference/venues.csv: {unknown}"

    legacy = pd.read_csv(RAW / "validGames_2026.csv", low_memory=False)
    legacy["playerId"] = fix_id(legacy.playerId)
    legacy["homeaway"] = legacy.homeaway.str.lower()
    legacy.to_csv(OUT / "validGames_legacy.csv", index=False)

    pd.Series(ID_ALIASES, name="playerId").rename_axis("oldId").to_csv(OUT / "player_id_aliases.csv")
    games.to_csv(OUT / "games.csv", index=False)
    stats.to_csv(OUT / "stats.csv", index=False)
    players.to_csv(OUT / "players.csv", index=False)
    valid.to_csv(OUT / "validGames.csv", index=False)
    print(f"games {len(games)}, stats {len(stats)}, players {len(players)}, validGames {len(valid)}, new IDs {len(new_ids)}")


if __name__ == "__main__":
    main()
