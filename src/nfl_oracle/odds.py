"""Optional live NFL moneyline quotes from The Odds API (never historical snapshots)."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from urllib.parse import urlencode
from urllib.request import urlopen
import pandas as pd

NFL_TEAMS = {
    'Arizona Cardinals':'ARI','Atlanta Falcons':'ATL','Baltimore Ravens':'BAL',
    'Buffalo Bills':'BUF','Carolina Panthers':'CAR','Chicago Bears':'CHI',
    'Cincinnati Bengals':'CIN','Cleveland Browns':'CLE','Dallas Cowboys':'DAL',
    'Denver Broncos':'DEN','Detroit Lions':'DET','Green Bay Packers':'GB',
    'Houston Texans':'HOU','Indianapolis Colts':'IND','Jacksonville Jaguars':'JAX',
    'Kansas City Chiefs':'KC','Las Vegas Raiders':'LV','Los Angeles Chargers':'LAC',
    'Los Angeles Rams':'LA','Miami Dolphins':'MIA','Minnesota Vikings':'MIN',
    'New England Patriots':'NE','New Orleans Saints':'NO','New York Giants':'NYG',
    'New York Jets':'NYJ','Philadelphia Eagles':'PHI','Pittsburgh Steelers':'PIT',
    'San Francisco 49ers':'SF','Seattle Seahawks':'SEA','Tampa Bay Buccaneers':'TB',
    'Tennessee Titans':'TEN','Washington Commanders':'WAS',
}
ALIASES = {'JAC':'JAX','STL':'LA','SD':'LAC','OAK':'LV'}


def extract_quotes(events: list, schedule: pd.DataFrame) -> pd.DataFrame:
    """Join only a same-day same-teams game; never guess another team's match."""
    schedule=schedule.copy()
    schedule['gameday']=pd.to_datetime(schedule['gameday']).dt.date
    rows=[]
    for event in events:
        home=NFL_TEAMS.get(event.get('home_team'))
        away=NFL_TEAMS.get(event.get('away_team'))
        when=pd.to_datetime(event.get('commence_time'),utc=True,errors='coerce')
        if not home or not away or pd.isna(when):continue
        date=when.tz_convert('America/New_York').date()
        candidates=schedule.loc[(schedule['gameday']==date)&
              (schedule['home_team'].replace(ALIASES)==home)&
              (schedule['away_team'].replace(ALIASES)==away)]
        if len(candidates)!=1:continue
        game_id=candidates.iloc[0]['game_id']
        for book in event.get('bookmakers',[]):
            for market in book.get('markets',[]):
                if market.get('key')!='h2h':continue
                outcomes={o.get('name'):o.get('price') for o in market.get('outcomes',[])}
                hml=outcomes.get(event.get('home_team'))
                aml=outcomes.get(event.get('away_team'))
                if hml is None or aml is None:continue
                stamp=market.get('last_update') or book.get('last_update')
                if not stamp:continue
                rows.append({'game_id':game_id,'home_moneyline':hml,
                    'away_moneyline':aml,'bookmaker':book.get('title') or book.get('key','unknown'),
                    'captured_at':stamp})
    return pd.DataFrame(rows,columns=['game_id','home_moneyline','away_moneyline','bookmaker','captured_at'])


def fetch_live_odds(schedule:pd.DataFrame, api_key:str|None=None) -> pd.DataFrame:
    """Only reads public upcoming h2h odds; may require a paid provider tier."""
    key=api_key or os.getenv('ODDS_API_KEY')
    if not key:raise ValueError('Set ODDS_API_KEY environment variable; do not commit the key.')
    params=urlencode({'apiKey':key,'regions':'us','markets':'h2h','oddsFormat':'american'})
    endpoint=f'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?{params}'
    with urlopen(endpoint,timeout=20) as response:
        events=json.load(response)
    if not isinstance(events,list):raise ValueError('Unexpected odds provider response.')
    return extract_quotes(events,schedule)
