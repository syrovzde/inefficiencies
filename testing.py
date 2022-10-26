import pandas as pd
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder

import arbitrage
from threshhold_calculator import indices_threshhold
from utils import utils as ut
import numpy as np
import maping

schemas = ['asianodds']
#markets = ['1x2', 'ou', 'ah']
markets = ['1x2','ah']
ip = "147.32.83.171"

markets_cols = {'1x2': ['draw', 'home', 'away'], 'bts': ['NO', 'YES'], 'ou': ['Over', 'Under'], 'ah': ['home', 'away'],
                'dc': ['homedraw', 'homeaway', 'awaydraw']}

sql_all_matches = """
SELECT \"MatchID\",\"Time\",\"Home\",\"Away\",\"League\" FROM asianodds.\"Matches\"
"""

sql_results = """
SELECT DISTINCT \"Result\" FROM football.\"Matches\" WHERE \"Home\" = \'{h}\' and \"Away\" = \'{a}\' and \"Time\" = \'{t}\'
"""

def find_all_completed_matches(engine, timestamp='2022-09-27 21:00:00.000000'):
    matches = pd.read_sql(sql=sql_all_matches, con=engine)
    matches = matches[matches['Time'] <= timestamp]
    return matches


def test(p=0.95, matchids=[1559766347], timestamp='2022-09-27 21:00:00.000000',max_bet=750,engine=None,res_engine=None):
    profits = []
    if matchids[0] == 0:
        matchids = find_all_completed_matches(engine=engine, timestamp=timestamp)
    points = 11
    possible_results = [str(j) + '-' + str(i) for i in range(points) for j in range(points)]
    probabilities = None
    counter = 0
    succ_counter = 0
    for matchid,home,away,time,league in matchids[['MatchID','Home','Away','Time','League']].to_numpy():
        matrix_A = pd.DataFrame({'PosState': possible_results})
        timestamp = None
        arb = arbitrage.Arbitrage(engine, schemas=schemas, markets=markets, bookmakers=[], moving_odds=False,max_bet=max_bet)
        for market in markets:
            odds = ut.load_asian_odds(engine=engine, MatchID=matchid, timestamp=timestamp, market=market)
            if market == '1x2':
                if odds.empty:
                    break
                odds = odds.iloc[-1]
                #order dependent
                hlp = {'draw': [odds['x']],'home': [odds['1']], 'away': [odds['2']],  'Timestamp': [odds['Timestamp']]}
                odds = pd.DataFrame(hlp)
                timestamp = odds['Timestamp'].values[0]
                #print(timestamp)
            columns, match_rows = arb.bookmaker_filter_asian_odds(odds=odds, market=market)
            if market != '1x2':
                for index, match_row in match_rows.iterrows():
                    matrix_A = arb.append_columns(matrix_A=matrix_A, market=market, bookmaker="",
                                              markets_cols=markets_cols,
                                              index=index, match_row=match_row, columns=columns)
        _, _, rows_to_drop, probabilities = indices_threshhold(p=p, probabilities=probabilities)
        for i in matrix_A.keys():
            if i == 'PosState':
                continue
            if not np.any(matrix_A[i] > 0):
                print(i)
                matrix_A.drop(i,axis=1,inplace=True)
        print(matrix_A)
        #print(matrix_A)
        matrix_B = matrix_A.copy()
        matrix_A.drop(rows_to_drop, inplace=True)
        succ, x = arb.solve_maxprofit_gurobi(matrix_A=matrix_A, MatchID=None, verbose=False)
        if succ:
            #print("{h} - {a}".format(h=home,a=away))
            # print(matrix_A.keys())
            succ_counter += 1
            #print(matrix_A.iloc[:, 1:].to_numpy()@x[:-1])
            translated_home, translated_away=maping.translate(ao_name_home=home,ao_name_away=away,engine=engine)
            translated_time = maping.change_date(AO_date=str(time))
            res=pd.read_sql(sql_results.format(h=translated_home,a=translated_away,t=translated_time),con=res_engine)
            if not res.empty:
                res = res.to_numpy()[0][0]
                #print(x,matchid)
                #print(timestamp)
                res_vector = matrix_B.loc[matrix_B['PosState'] == res].to_numpy()[0][1:]
                profit = np.dot(res_vector,np.array(x[:-1]))
                profits.append(profit)
        counter += 1
    print(succ_counter / counter)
    return sum(profits)
# def update_matrix(indices,matrix_a):


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
    print(test(matchids=[0], timestamp='2022-10-11 21:00:00', p=0.95,engine=engine,res_engine=result_engine))
