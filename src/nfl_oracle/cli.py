"""CLI. Download public nflverse schedules on demand or use cached CSV."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from .core import FEATURES, make_features, eligible, fit_model, backtest, rank_edges


def get_data(path: str|None, seasons: list[int]) -> pd.DataFrame:
    if path:
        df = pd.read_csv(path)
    else:
        import nflreadpy as nfl
        df = nfl.load_schedules(seasons).to_pandas()
    if 'game_type' in df.columns:
        df=df.loc[df['game_type'].isin(['REG'])].copy()
    return df


def main():
    parser=argparse.ArgumentParser(description='NFL Oracle: evaluate historical NFL moneyline probabilities.')
    parser.add_argument('command',choices=['backtest','shortlist'])
    parser.add_argument('--schedule-csv', default=None,help='Offline nflverse schedule CSV; otherwise use nflreadpy download')
    parser.add_argument('--test-year',type=int,default=2025)
    parser.add_argument('--seasons',nargs='+',type=int,default=list(range(2020,2027)))
    parser.add_argument('--as-of',help='For shortlist: America/New_York timestamp YYYY-MM-DDTHH:MM; excludes results from that date onward')
    parser.add_argument('--odds-csv',help='For shortlist: independent current bookmaker odds CSV with game_id, home_moneyline, away_moneyline, captured_at, bookmaker')
    parser.add_argument('--live-odds', action='store_true', help='Fetch upcoming market quotes using ODDS_API_KEY instead of a CSV')
    parser.add_argument('--parlay-bankroll',type=float,default=1000.,help='Research bankroll used to dynamically size the aggregate experimental parlay sleeve; it is not automatically wagered')
    parser.add_argument('--output',default='results',help='Output directory')
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    games=get_data(args.schedule_csv,args.seasons)
    if args.command=='backtest':
        report, rows = backtest(games,test_year=args.test_year)
        (output/'backtest_report.json').write_text(json.dumps(report,indent=2))
        rows.to_csv(output/'backtest_predictions.csv',index=False)
        print(json.dumps(report,indent=2))
        print('Results written to', output.resolve())
        return
    if not args.as_of or (bool(args.odds_csv)==bool(args.live_odds)):
        parser.error('shortlist requires --as-of and exactly one of --odds-csv or --live-odds.')
    now=pd.Timestamp(args.as_of)
    if now.tz is not None:
        now=now.tz_convert('America/New_York').tz_localize(None)
    day_start=now.normalize()
    historical = games.copy()
    historical['gameday']=pd.to_datetime(historical['gameday'])
    # Critically: even if source CSV contains future final scores, don't use them.
    historical.loc[historical['gameday']>=day_start,['home_score','away_score']]=float('nan')
    features=make_features(historical)
    train=eligible(features.loc[features['gameday']<day_start])
    model=fit_model(train)
    coming=features.loc[(features['gameday']>=day_start)&(features['gameday']<day_start+pd.Timedelta(days=7)) & ~features['completed']].copy()
    # Odds must be external and verified captured at/before as-of; book/lines not blindly taken from nflverse future data.
    if args.live_odds:
        from .odds import fetch_live_odds
        odds=fetch_live_odds(historical)
        odds.to_csv(output/'live_quotes_audit.csv',index=False)
    else:
        odds=pd.read_csv(args.odds_csv)
    needed={'game_id','home_moneyline','away_moneyline','bookmaker','captured_at'}
    if not needed.issubset(odds.columns):
        raise ValueError(f'Odds CSV requires columns {sorted(needed)}')
    odds['captured_at']=pd.to_datetime(odds['captured_at'],utc=True,errors='raise')
    quote_time=now.tz_localize('America/New_York').tz_convert('UTC')
    odds=odds.loc[(odds['captured_at']<=quote_time)&(odds['captured_at']>=quote_time-pd.Timedelta(hours=12))]
    if odds.empty:
        print('NO VERIFIED ODDS: no watchlist generated.'); return
    coming=coming.drop(columns=['home_moneyline','away_moneyline','market_home_prob','market_logit'])
    # Features except market_logit must come from form BEFORE matchup.
    coming=coming.merge(odds,on='game_id',how='inner')
    if coming.empty:
        print('NO MATCHING UPCOMING GAMES: no watchlist generated.');return
    from .core import no_vig_home
    import numpy as np
    coming['market_home_prob']=coming.apply(lambda r:no_vig_home(r['home_moneyline'],r['away_moneyline']),axis=1)
    p=coming['market_home_prob'].clip(1e-6,1-1e-6)
    coming['market_logit']=np.log(p/(1-p))
    coming['model_home_prob']=model.predict_proba(coming[FEATURES])[:,1]
    shortlist=rank_edges(coming)
    shortlist=shortlist.merge(coming[['game_id','bookmaker','captured_at']],on='game_id',how='left')
    shortlist.to_csv(output/'research_watchlist.csv',index=False)
    # Experimental two-leg cross-game ML parlay check. The same-book constraint
    # avoids combining best prices from bookmakers that cannot be parlayed.
    # The calculation uses only timestamp-filtered, pregame odds and estimates.
    from .parlays import multi_leg_parlay_watchlist
    candidate_parlays, audited_parlays = multi_leg_parlay_watchlist(
        coming, shortlist, quote_time, parlay_bankroll=args.parlay_bankroll,
    )
    candidate_parlays.to_csv(output/'experimental_parlay_watchlist.csv',index=False)
    audited_parlays.to_csv(output/'parlay_candidate_audit.csv',index=False)
    (output/'parlay_research_report.json').write_text(json.dumps({
        'candidates_reviewed':len(audited_parlays),
        'candidates_featured':len(candidate_parlays),
        'research_parlay_bankroll':args.parlay_bankroll,
        'method':'All 2..N different-game cross-game ML subsets at one bookmaker; overlapping straight bets and round-robin tickets permitted; decimal odds multiply; collective risk allocated dynamically from configured bankroll.',
        'limitations':[
            'Model probabilities are an experimental shrinkage blend, not prospectively calibrated for parlays.',
            'The odds provider reports quote update times; valid timestamps do not guarantee ticket acceptance.',
            'Parlays have high variance; no amount of compounding negative-value legs manufactures value.',
            'An experimental stake is a research output, never an automated wager.',
        ],
    },indent=2))
    print(f'{len(candidate_parlays)} experimental cross-game parlay opportunity reviewed; '
          f'{len(audited_parlays)} pairs audited (not proven betting recommendations).')
    print(f'{len(shortlist)} research candidates (NOT validated betting recommendations).')
    print(shortlist.to_string(index=False) if len(shortlist) else 'NO EDGE: No candidates cleared the configured filters.')

if __name__=='__main__':main()
