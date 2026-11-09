# afl-fantasy

Matt's AFL fantasy data and projections, the AFL counterpart to Rugby Per Minute. Personal use for now.
Games played: Classic (fantasy.afl.com.au), WAFL (draft league), Shiva (keeperfantasy.com). All three use the standard AFL Fantasy scoring.

## Rebuild

```
python3 scripts/build.py   # data/raw + data/sources + data/reference -> data/clean
python3 scripts/dfs.py     # DFS Australia pre-season sheets -> dfs_preseason, dfs_other_leagues
python3 scripts/drafts.py   # Wikipedia draft history 2005-2025 -> drafts
python3 scripts/check.py   # integrity checks
```

Never edit `data/clean` by hand; fix the inputs or the build and rerun.

## Layout

| Folder | What's in it |
|---|---|
| `data/raw/` | Matt's master exports as of end of 2026, untouched |
| `data/sources/` | Outside data: `wikipedia/` draft history 2005–25; `dfs/` DFS Australia AFL Fantasy spreadsheets 2023–26; AFL Tables via the [AFL-Data-Analysis](https://github.com/akareen/AFL-Data-Analysis) scrape, AFL Tables 2026 Opening Round pages, Zero Hanger 2027 fixture) |
| `data/reference/` | Hand-maintained lookups: `venues.csv` (canonical venue names, sponsor aliases, timezones), `list_changes_2027.csv`, `draft_links.csv` (hand-checked draft pick → playerId corrections; blank = not an AFL player) |
| `data/clean/` | The tables to use |

## Clean tables

- **games** — every regular-season game 2005–2027. `gameId` = `YYYYRrrgg`. `startTime` is venue-local. 2027 rounds 17–24 have no date/time yet (TBC when read). No finals.
- **stats** — one row per player per game, regular season, 2005–2026. Full league for every season. `subs` is On/Off (2011–15 and 2021–25). In 2023–25 Matt also marked "Off" for players who left injured and didn't return (about 70 a season), so treat Off as "didn't finish". 2020 is scaled up from 16-minute quarters.
- **validGames** — stats filtered by: no sub on/off, `pctPlayed >= VALID_MIN_TOG` (25, set in `build.py`). All players, including retired. To become per game version (Classic / WAFL / Shiva) once 2027 rules are set.
- **validGames_legacy** — Matt's original validGames for comparison (mostly current players, mixed historical rules).
- **players** — every player in stats plus undebuted squad players (blank `playerId`). Matt's columns kept; adds `firstSeason`, `lastSeason`, `team2027`, `status2027`, `active2027`.
- **player_id_aliases** — old IDs that now point to another ID.
- **dfs_preseason** — DFS Australia AFL Fantasy pre-season sheets 2023–2026: one row per player per season with starting price, ownership, position, Champion Data ID, and the previous AFL season (games incl. finals, average, regular-season average, CBA%, PPM). DFS excludes games under 50% TOG. Matched to playerId for every player with AFL games (regular-season averages agree with stats 99.96%).
- **drafts** — every national, rookie, pre-season, mid-season and mini-draft pick 2005–2025 (Wikipedia), with club, pick, league drafted from (junior / state league / re-listed AFL), `firstSeason`, `ageAtDraft` (players who debuted), `playerId` where the player has played AFL, and `isFirstDraft` (false for re-listings and later re-drafts). A few 2008 rookie picks were unreadable on Wikipedia and are missing.
- **dfs_other_leagues** — previous-season averages in VFL, WAFL, SANFL, SANFL Reserves/U18s and NAB/Coates Talent League (2022–2025), one row per player per league per year.

## Conventions

- Fantasy score = 3×kicks + 2×handballs + 3×marks + 4×tackles + frees for − 3×frees against + hit-outs + 6×goals + behinds. Holds on every row.
- New player IDs follow Matt's format: debut `YYYY RR GG PP` (year, round, game, nth debutant in that game). Older IDs from a previous source are kept as-is.
- Venue names follow Matt's historic (AFL Tables) names, e.g. Docklands not Marvel Stadium. New sponsor names go in `data/reference/venues.csv` as aliases.
- Team names: Brisbane (not Brisbane Lions), Greater Western Sydney, Western Bulldogs.

## Data fixes made on import (Nov 2026)

- 2005–2011 back-filled for the whole league from AFL Tables (≈52k rows, 564 new player IDs). Overlap with Matt's rows agreed on 2,166 of 2,172 scores.
- 39 phantom Michael Hibberd rows (2009–11, copies of Michael Hurley's games) dropped.
- 2011 sub flags taken from AFL Tables markers (agree with every real flag Matt had).
- 2026 Opening Round (5 games) added from AFL Tables; matches Matt's validGames on 226/228 rows (McCartin, McInerney differ by 2).
- IDs merged: Angus Litherland→Dewar, Fergus Greene, Sam Skinner (two IDs each); Malakai Champion 11-digit typo; Bodie Ryan is 2026090101.
- Rows reassigned: Tom Lynch (Richmond) 2022 games were under the other Tom Lynch; Aaron Black (West Coast, 2022) is a different player from Aaron Black (North/Geelong), new ID 2022020704.
- 2025–26 dates in games were ISO; now all dates are ISO `YYYY-MM-DD`.

## Still to do

- Add draftees after the national (19–20 Nov) and rookie (23 Nov) drafts; supplemental signings to March.
- 2027 rounds 17–24 dates/times once confirmed.
- Team scores and attendance missing for 2023–2026.
- Per-game-version valid rules (Classic / WAFL / Shiva).
- A consistent injury / didn't-finish flag for projections, separate from subs.
- Weekly 2027 updates from DFS Australia downloads (adds centre bounce attendances and kick-ins), cross-checked against AFL Tables.
