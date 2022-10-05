import pandas as pd
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder

import arbitrage
from utils import utils as ut

schemas = ['asianodds']
markets = ['1x2', 'ou', 'ah']
ip = "147.32.83.171"

markets_cols = {'1x2': ['draw', 'home', 'away'], 'bts': ['NO', 'YES'], 'ou': ['Under', 'Over'], 'ah': ['away', 'home'],
                'dc': ['homedraw', 'homeaway', 'awaydraw']}
sql_x = """ SELECT \"1\",\"2\",\"x\",\"Timestamp\" FROM asianodds.\"Odds_1x2_FT\" WHERE \"Odds_1x2_FT\".\"MatchID\" = \'{matchid}\'
"""

sql_ou = """SELECT 
\"Over\",\"1\",\"2\" FROM asianodds.\"Odds_ou_FT\" WHERE \"Odds_ou_FT\".\"MatchID\" = \'{matchid}\' AND
\"Odds_ou_FT\".\"Timestamp\" = \'{timestamp}\'
"""
sql_ah = """ SELECT 
\"Handicap\","1",\"2\"FROM asianodds.\"Odds_ah_FT\" WHERE \"Odds_ah_FT\".\"MatchID\" = \'{matchid}\' AND
\"Odds_ah_FT\".\"Timestamp\" = \'{timestamp}\'
"""

ssh_tunnel = SSHTunnelForwarder(
    ip,
    ssh_username='syrovzde',
    ssh_private_key='C:\\Users\\zdesi\\.ssh\\syrovzde_rsa',
    remote_bind_address=('localhost', 5432)
)
ssh_tunnel.start()

engine = create_engine("postgresql://{user}@{host}:{port}/{db}".format(
    host='localhost',
    port=ssh_tunnel.local_bind_port,
    user='syrovzde',
    db='asianodds'
))
matchid = 1559766347
points = 11

possible_results = [str(j) + '-' + str(i) for i in range(points) for j in range(points)]

matrix_A = pd.DataFrame({'PosState': possible_results})

timestamp = None
arb = arbitrage.Arbitrage(engine.connect(), schemas=schemas, markets=markets, bookmakers=[], moving_odds=False)
for market in markets:
    odds = ut.load_asian_odds(engine=engine, MatchID=matchid, timestamp=timestamp, market=market)
    if market == '1x2':
        print(odds)
        odds = odds.iloc[-1]
        hlp = {'home': [odds['1']], 'away': [odds['2']], 'draw': [odds['x']], 'Timestamp': [odds['Timestamp']]}
        odds = pd.DataFrame(hlp)
        timestamp = odds['Timestamp'].values[0]
        print(timestamp)
    print(odds)
    columns, match_rows = arb.bookmaker_filter_asian_odds(odds=odds, market=market)
    for index, match_row in match_rows.iterrows():
        matrix_A = arb.append_columns(matrix_A=matrix_A, market=market, bookmaker="",
                                      markets_cols=markets_cols,
                                      index=index, match_row=match_row, columns=columns)
print(matrix_A)
