# NFL Oracle — October 16, 2026 percentage-only as-of personnel replay

Successful run: https://github.com/joestella99/NFL_Oracle/actions/runs/38954814076
Commit: https://github.com/joestella99/NFL_Oracle/commit/e9434254563badad3fde5b09c32659690d5a75fe

Official percent-only rules: `config/parlay_percentage_policy.json`, `scripts/parlay_2026_review.py`. No stake, dollar or bankroll selection. Minimum conservative combined-ticket EV +2%, 1% per-leg gross payout stress, min 60% individual favored-team probability, min 15% conservative ticket probability; no individual-leg EV hard veto.

The completed-game check covers **79 games**, 2026 Weeks 1–5 and **one Thursday Week 6 game on Oct 15**. We neither use uncompleted games as labels nor infer future Sunday line availability. `compare` runs winner-model training strictly before each evaluated week, and the injury-augmented probability models train on prior seasons only. Model variant selection was fixed by 2023–2025 Brier before scoring 2026; the **market** won this prior-year comparison.

| 2026 completed-game metric | No-vig market | Football QB/EPA/matchups | Football + as-of nflreadpy injuries |
| --- | ---: | ---: | ---: |
| Win classification accuracy | **50/79, 63.29%** | 46/79, 58.23% | 46/79, 58.23% |
| Brier | **0.230307** | 0.233711 | 0.233584 |

**2026 completed-week candidate audit** (base model chosen from 2023–25: no-vig market): Week 1 247 mathematical combinations / 0 qualifying; Week 2 8,178 / 0; Week 3 1,013 / 0; Week 4 502 / 0; Week 5 502 / 0; Week 6 Thursday 0 / 0. **10,442 mathematical parlay combinations, 0 qualifying**. These reuse games, are NOT 10,442 independent bettable tickets, and their compounded odds are unverified historic proxies. The report does **not** assess live Week 6 Sunday opportunities.

**nflreadpy injury source actually exists for 2026**, with **1,767** report records. However, for the 79 completed games in this replay, **zero** had pre-kickoff status-known QB injury reports for *both teams*, and **zero** had both-team OL coverage; **254** player report rows were filtered out because their `date_modified` was at/after kickoff. Lack of coverage does **not** mean healthy. QB injury reports are not confirmed starting-QB identities. `nflreadpy.load_depth_charts` can return depth chart projections but has not supplied reliable initial-publication timestamps to establish historical starter knowledge; those are NOT backfilled.

**ESPN prospectively captured:** On 2026-10-16T19:30:19Z, all **34/34** requested unofficial ESPN endpoints responded: league teams, league injury report, and depth charts for 32 teams. This includes projected QB rankings and injury data observed at capture, not verified starter confirmation and not historical reconstruction. The raw, timestamped `espn_asof_2026/snapshot.json` artifact is retained in the run. It may inform **games with kickoff after the capture time** once joined and validated; it cannot retroactively change Oct 15 or older bets. The current code stores source snapshots; it does not yet claim quantitative improvement from projected starters.

The 2024 football-only Brier 0.200528 was slightly worse than market 0.200162 and adding available 2023–24 QB/OL reports worsened 2024 Brier to 0.201660. In 2025, only zero both-team pre-kickoff QB/OL reports were available; 2025 football-only Brier 0.213435 and injury-augmented 0.213448 vs market 0.212139. Treat the personnel model as experimental, not automatically superior.

**Important:** 2026 results were reviewed after the 2016–2025 model designs were created; this is a same-day retrospective replay, not frozen, prospective bookmaker-executed bets. Reported historical moneylines may not represent prices available before kickoff. The official dollar-funded historical V3 checkpoint is retained as a separate research comparison, but the **current percentage-only 2026 selection interface ignores stakes entirely**.
