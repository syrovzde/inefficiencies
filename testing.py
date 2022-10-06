import pandas as pd
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder
from threshhold_calculator import indices_threshhold
import arbitrage
from utils import utils as ut
import gurobipy as gb
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

sql_all_matches = """
SELECT \"MatchID\",\"Time\",\"Home\",\"Away\" FROM asianodds.\"Matches\"
"""

def find_all_completed_matches(engine,timestamp = '2022-09-27 21:00:00.000000'):
    matches = pd.read_sql(sql=sql_all_matches,con=engine)
    matches=matches[matches['Time'] < timestamp]
    return matches


def test(p=0.95,matchids=[1559766347],timestamp='2022-09-27 21:00:00.000000'):
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
    if matchids[0] == 0:
        matchids = find_all_completed_matches(engine=engine,timestamp=timestamp)
    points = 11
    possible_results = [str(j) + '-' + str(i) for i in range(points) for j in range(points)]
    probabilities = None
    counter = 0
    succ_counter = 0
    for matchid in matchids['MatchID'].to_numpy():
        matrix_A = pd.DataFrame({'PosState': possible_results})
        timestamp = None
        arb = arbitrage.Arbitrage(engine, schemas=schemas, markets=markets, bookmakers=[], moving_odds=False)
        for market in markets:
            odds = ut.load_asian_odds(engine=engine, MatchID=matchid, timestamp=timestamp, market=market)
            if market == '1x2':
                if odds.empty:
                    break
                odds = odds.iloc[-1]
                hlp = {'home': [odds['1']], 'away': [odds['2']], 'draw': [odds['x']], 'Timestamp': [odds['Timestamp']]}
                odds = pd.DataFrame(hlp)
                timestamp = odds['Timestamp'].values[0]
            columns, match_rows = arb.bookmaker_filter_asian_odds(odds=odds, market=market)
            for index, match_row in match_rows.iterrows():
                matrix_A = arb.append_columns(matrix_A=matrix_A, market=market, bookmaker="",
                                              markets_cols=markets_cols,
                                              index=index, match_row=match_row, columns=columns)
        _,_,rows_to_drop,probabilities = indices_threshhold(p=p,probabilities=probabilities)
        matrix_A.drop(rows_to_drop,inplace=True)
        #print(matrix_A)
        succ,x=arb.solve_maxprofit_gurobi(matrix_A=matrix_A,MatchID=None,verbose=False)
        if succ:
            #print(matrix_A.keys())
            succ_counter+=1
        counter +=1
    print(succ_counter/counter)
#def update_matrix(indices,matrix_a):


test(matchids=[0],timestamp='2022-09-27 21:00:00',p=0.95)

