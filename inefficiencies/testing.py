import pandas as pd
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder

import arbitrage
from threshhold_calculator import indices_threshhold
from utils import utils as ut
import numpy as np
import maping

schemas = ['asianodds']
# markets = ['1x2', 'ou', 'ah']
markets = ['ou','ah']
ip = "147.32.83.171"

markets_cols = {'1x2': ['draw', 'home', 'away'], 'bts': ['NO', 'YES'], 'ou': ['Over', 'Under'], 'ah': ['home', 'away'],
                'dc': ['homedraw', 'homeaway', 'awaydraw']}

sql_all_matches = """
SELECT \"MatchID\",\"Time\",\"Home\",\"Away\",\"League\" FROM asianodds.\"Matches\"
"""
sql_one_match = """
SELECT \"MatchID\",\"Time\",\"Home\",\"Away\",\"League\" FROM asianodds.\"Matches\" WHERE \"MatchID\" = \'{id}\'
"""

sql_results = """
SELECT DISTINCT \"Result\" FROM football.\"Matches\" WHERE \"Home\" = \'{h}\' and \"Away\" = \'{a}\' and \"Time\" = \'{t}\'
"""


def find_all_completed_matches(engine, timestamp='2022-09-27 21:00:00.000000'):
    matches = pd.read_sql(sql=sql_all_matches, con=engine)
    matches = matches[matches['Time'] >= timestamp]
    matches = matches[matches['Time'] < '2022-11-30 11:00:00.000000']
    return matches

def find_one_match(engine,matchID):
    matches = pd.read_sql(sql=sql_one_match.format(id=matchID),con=engine)
    return matches

def find_result(home, away, engine, res_engine, time):
    translated_home, translated_away = maping.translate(ao_name_home=home, ao_name_away=away)
    translated_time = maping.change_date(AO_date=str(time))
    return pd.read_sql(sql_results.format(h=translated_home, a=translated_away, t=translated_time), con=res_engine)


def calculate_mapping(matchids=None,engine=None, res_engine=None):
    timestamp = '2022-11-13 00:00:00'
    succ = 1
    so_far = 0
    if matchids is None:
        matchids = find_all_completed_matches(engine=engine, timestamp=timestamp)
    print(len(matchids))
    csv_file_with_mapping = "bet_AsianOdds_AO2BE.csv"
    translate_table = pd.read_csv(csv_file_with_mapping)
    for matchid, home, away, time, league in matchids[['MatchID', 'Home', 'Away', 'Time', 'League']].to_numpy():
        print(league)
        so_far +=1
        if  "No. of" in home:
            continue
        count,translate_table = maping.find_match(ao_home=home,ao_away=away,engine=res_engine,date=time,league=league,translate_table=translate_table)
        succ += count
        print(succ/so_far)




def test(p=0.95, matchids=None, timestamp='2022-09-27 21:00:00.000000', max_bet=750, engine=None, res_engine=None,
         weighted=False, weights=None, points=11):
    profits = []
    betting_vectors = []
    weighted = False
    if weights is not None:
        weighted = True
    counter = 0
    succ_counter = 0
    res_available = 0
    total = 0
    if matchids is None:
        matchids = find_all_completed_matches(engine=engine, timestamp=timestamp)
    #else:
    #    matchids = find_one_match(engine=engine,matchID=matchids[0])
    possible_results = [str(j) + '-' + str(i) for i in range(points) for j in range(points)]
    rows_to_stay, rows_to_drop, probabilities = indices_threshhold(p=p, probabilities=None)
    if weighted:
        weights = weights[rows_to_stay]
    for matchid, home, away, time, league in matchids[['MatchID', 'Home', 'Away', 'Time', 'League']].to_numpy():
        if '\'' in home or '\'' in away:
            continue
        res = find_result(home=home, away=away, engine=engine, res_engine=res_engine, time=time)
        if res.empty:
            counter+=1
            continue
        timestamp = None
        arb = arbitrage.Arbitrage(engine, schemas=schemas, markets=markets, bookmakers=[], moving_odds=False,
                                  max_bet=max_bet)
        xodds = ut.load_asian_odds(engine=engine, MatchID=matchid, timestamp=timestamp, market='1x2')
        """Timestamps are synchronized over available 1x2 odds"""
        if xodds.empty:
            continue
        for index,odds in xodds.iterrows():
            matrix_A = pd.DataFrame({'PosState': possible_results})
            hlp = {'draw': [odds['x']], 'home': [odds['1']], 'away': [odds['2']], 'Timestamp': [odds['Timestamp']]}
            odds = pd.DataFrame(hlp)
            timestamp = odds['Timestamp'].values[0]
            columns, match_rows = arb.bookmaker_filter_asian_odds(odds=odds, market='1x2')
            for index, match_row in match_rows.iterrows():
                matrix_A = arb.append_columns(matrix_A=matrix_A, market='1x2', bookmaker="",
                                                  markets_cols=markets_cols,
                                                  index=index, match_row=match_row, columns=columns)
            for market in markets:
                dif_odds = ut.load_asian_odds(engine=engine, MatchID=matchid, timestamp=timestamp, market=market)
                """Timestamps are synchronized over available 1x2 odds"""
                columns, match_rows = arb.bookmaker_filter_asian_odds(odds=dif_odds, market=market)
                for index, match_row in match_rows.iterrows():
                    matrix_A = arb.append_columns(matrix_A=matrix_A, market=market, bookmaker="",
                                                      markets_cols=markets_cols,
                                                      index=index, match_row=match_row, columns=columns)
            matrix_B = matrix_A.copy()
            matrix_A.drop(rows_to_drop, inplace=True)
            if weighted:
                succ, x, z = arb.solve_maxprofit_gurobi(matrix_A=matrix_A, MatchID=None, verbose=False, weights=weights)
            else:
                succ, x, _ = arb.solve_maxprofit_gurobi(matrix_A=matrix_A, MatchID=None, verbose=False, weights=weights)
            if succ:
                succ_counter += 1
                try:
                    res = res.to_numpy()[0][0]
                except AttributeError:
                    continue
                print_arb_columns(matrix_B,x,weighted=weighted)
                res_vector = matrix_B.loc[matrix_B['PosState'] == res].to_numpy()[0][1:]
                profit = ret_profit(res_vector=res_vector, weighted=weighted, x=x)
                print(profit/np.sum(x[:-1]))
                profits.append(profit)
            #print(succ_counter/res_available)
                betting_vectors.append(x)
            counter += 1
            res_available +=1
    print(succ_counter)
    print(succ_counter / res_available)
    return profits, betting_vectors


def ret_profit(res_vector, weighted, x):
    if weighted:
        return np.dot(res_vector, np.array(x))
    return np.dot(res_vector, np.array(x[:-1]))


def print_arb_columns(matrix_B, x, weighted=False):
    indexing = [True]
    arr = x if weighted else x[:-1]
    for i in arr:
        if i != 0:
            indexing.append(True)
        else:
            indexing.append(False)
    print(matrix_B.loc[:, indexing])


if __name__ == '__main__':
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

    result_engine = create_engine("postgresql://{user}@{host}:{port}/{db}".format(
        host='localhost',
        port=ssh_tunnel.local_bind_port,
        user='syrovzde',
        db='betexplorer'
    ))
    weights = np.loadtxt('weights.txt')
    #test(timestamp='2022-11-13 21:00:00', p=1, engine=engine, res_engine=result_engine, weights=weights)
    #test(timestamp='2022-09-09 21:00:00', p=1, engine=engine, res_engine=result_engine,matchids=['1559692220'])
    #test(timestamp='2022-11-13 21:00:00', p=0.8, engine=engine, res_engine=result_engine)
    calculate_mapping(engine=engine,res_engine=result_engine,)
