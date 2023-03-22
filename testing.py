import pandas as pd
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder
import sql_queries
import arbitrage
from threshhold_calculator import indices_threshhold
from utils import utils as ut
import numpy as np
import maping
from poisson import poisson


schemas = ['asianodds']
# markets = ['1x2', 'ou', 'ah']
markets = ['ou', 'ah']
ip = "147.32.83.171"

markets_cols = {'1x2': ['draw', 'home', 'away'], 'bts': ['NO', 'YES'], 'ou': ['Over', 'Under'], 'ah': ['home', 'away'],
                'dc': ['homedraw', 'homeaway', 'awaydraw']}


def find_all_completed_matches(results_engine, timestamp='2022-09-27 21:00:00.000000'):
    matches = pd.read_sql(sql=sql_queries.sql_all_matches, con=results_engine)
    # matches = matches[matches['Time'] <= timestamp]
    matches = matches[matches['Time'] > timestamp]
    return matches


def find_one_match(asian_odds_engine, match_id):
    matches = pd.read_sql(sql=sql_queries.sql_one_match.format(id=match_id), con=asian_odds_engine)
    return matches


def find_result(home, away, res_engine, time):
    translated_home, translated_away = maping.translate(ao_name_home=home, ao_name_away=away)
    translated_time = maping.change_date(AO_date=str(time))
    return pd.read_sql(sql_queries.sql_results.format(h=translated_home, a=translated_away, t=translated_time),
                       con=res_engine)


def calculate_mapping(match_ids=None, results_engine=None, res_engine=None):
    timestamp = '2022-11-13 00:00:00'
    succ = 1
    so_far = 0
    if match_ids is None:
        match_ids = find_all_completed_matches(results_engine=results_engine, timestamp=timestamp)
    print(len(match_ids))
    csv_file_with_mapping = "csv_files/bet_AsianOdds_AO2BE.csv"
    translate_table = pd.read_csv(csv_file_with_mapping)
    for matchid, home, away, time, league in match_ids[['MatchID', 'Home', 'Away', 'Time', 'League']].to_numpy():
        print(league)
        so_far += 1
        if "No. of" in home:
            continue
        count, translate_table = maping.find_match(ao_home=home, ao_away=away, engine=res_engine, date=time,
                                                   league=league, translate_table=translate_table)
        succ += count
        print(succ / so_far)


def res_test(res):
    if res.empty:
        return False
    try:
        res.to_numpy()[0][0]
    except AttributeError:
        return False
    return True


def get_lambdas(home_team,away_team,league,time):
    return None,None

def test(p=0.95, match_ids=None, timestamp='2022-09-27 21:00:00.000000', max_bet=750, engine=None, res_engine=None,
         weighted=False, weights=None, points=11):
    profits = []
    betting_vectors = []
    matchids_profit = []
    if weights is not None:
        weighted = True
    if match_ids is None:
        match_ids = find_all_completed_matches(results_engine=engine, timestamp=timestamp)
    else:
        matches = find_all_completed_matches(results_engine=engine, timestamp='2022-01-01 21:00:00')
        match_ids = matches[matches['MatchID'].isin(match_ids)]

    possible_results = [str(j) + '-' + str(i) for i in range(points) for j in range(points)]
    rows_to_stay, rows_to_drop, probabilities = indices_threshhold(p=p, probabilities=None)
    print(len(match_ids))
    if weighted:
        weights = probabilities[rows_to_stay]
        # weights[:] = 1/121
    for match_id, home, away, time, league in match_ids[['MatchID', 'Home', 'Away', 'Time', 'League']].to_numpy():
        if '\'' in home or '\'' in away:
            continue
        res = find_result(home=home, away=away, res_engine=res_engine, time=time)
        if not res_test(res):
            continue
        res = res.to_numpy()[0][0]
        timestamp = None
        arb = arbitrage.Arbitrage(engine, schemas=schemas, markets=markets, bookmakers=[], moving_odds=False,
                                  max_bet=max_bet)
        xodds = ut.load_asian_odds(engine=engine, MatchID=match_id, timestamp=timestamp, market='1x2')
        if xodds.empty:
            continue
        for index, odds in xodds.iterrows():
            matrix_A = pd.DataFrame({'PosState': possible_results})
            hlp = {'draw': [odds['x']], 'home': [odds['1']], 'away': [odds['2']], 'Timestamp': [odds['Timestamp']]}
            odds = pd.DataFrame(hlp)
            timestamp = odds['Timestamp'].values[0]
            columns, match_rows = arb.bookmaker_filter_asian_odds(odds=odds, market='1x2')
            for inner_index, match_row in match_rows.iterrows():
                matrix_A = arb.append_columns(matrix_A=matrix_A, market='1x2', bookmaker="",
                                              markets_cols=markets_cols,
                                              index=inner_index, match_row=match_row, columns=columns)
            for market in markets:
                dif_odds = ut.load_asian_odds(engine=engine, MatchID=match_id, timestamp=timestamp, market=market)
                columns, match_rows = arb.bookmaker_filter_asian_odds(odds=dif_odds, market=market)
                for inner_index, match_row in match_rows.iterrows():
                    matrix_A = arb.append_columns(matrix_A=matrix_A, market=market, bookmaker="",
                                                  markets_cols=markets_cols,
                                                  index=inner_index, match_row=match_row, columns=columns)
            if weighted:
                lambda_h,lambda_a = get_lambdas(home_team=home,away_team=away,time=time,league=league)
                weights = poisson(lambda_h=lambda_h,lambda_a=lambda_a,points=points)
            matrix_B = matrix_A.copy()
            matrix_A.drop(rows_to_drop, inplace=True)
            if weighted:
                succ, x, z = arb.solve_maxprofit_gurobi(matrix_A=matrix_A, MatchID=None, verbose=False, weights=weights)
            else:
                succ, x, _ = arb.solve_maxprofit_gurobi(matrix_A=matrix_A, MatchID=None, verbose=False, weights=weights)
            if succ:
                res_vector = matrix_B.loc[matrix_B['PosState'] == res].to_numpy()[0][1:]
                profit = ret_profit(res_vector=res_vector, weighted=weighted, x=x)
                profits.append(profit)
                matchids_profit.append(match_id)
                betting_vectors.append(x)
    return profits, betting_vectors, matchids_profit


def ret_profit(res_vector, weighted, x):
    if weighted:
        return np.dot(res_vector, np.array(x))
    return np.dot(res_vector, np.array(x[:-1]))


def preprocess(results_engine):
    lambdas = pd.read_csv('cleaned_lambdas.csv')
    all_matches = pd.read_sql(sql_queries.sql_all_results, results_engine)
    ids = lambdas["MatchID"].values
    print(all_matches[all_matches['MatchID'].isin(ids)])


def print_arb_columns(matrix, x, weighted=False):
    indexing = [True]
    arr = x if weighted else x[:-1]
    for i in arr:
        if i != 0:
            indexing.append(True)
        else:
            indexing.append(False)
    print(matrix.loc[:, indexing])


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

    # test(timestamp='2022-11-13 21:00:00', p=1, engine=engine, res_engine=result_engine, weights=weights)
    # test(timestamp='2022-11-13 21:00:00', p=1, engine=engine, res_engine=result_engine)
    # test(timestamp='2022-01-01 21:00:00', p=1, engine=engine, res_engine=result_engine)
    test(p=1, engine=engine, res_engine=result_engine, timestamp='2022-01-15 21:00:00')
