import pandas as pd
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder
import sql_queries
import time
import arbitrage
from threshhold_calculator import indices_threshhold
from utils import utils as ut
import numpy as np
import maping
import poisson as ps
from poisson import poisson
import scipy.stats as stats
from utils import helper_functions as hlp

csv_file_with_mapping = 'csv_files/bet_AsianOdds_AO2BE.csv'
schemas = ['asianodds']
# markets = ['1x2', 'ou', 'ah']
markets = ['ou', 'ah']
ip = "147.32.83.171"

markets_cols = {'1x2': ['draw', 'home', 'away'], 'bts': ['NO', 'YES'], 'ou': ['Over', 'Under'], 'ah': ['home', 'away'],
                'dc': ['homedraw', 'homeaway', 'awaydraw']}


def clean_results(profits, ids, threhsold=40000):
    new_profits = []
    last_id = None
    for p, id in zip(profits, ids):
        if id != last_id:
            last_id = id
            if abs(p) > threhsold:
                continue
            new_profits.append(p)
    return np.array(new_profits)


def find_all_completed_matches(results_engine, timestamp='2022-09-27 21:00:00.000000', limited=False):
    if not limited:
        matches = pd.read_sql(sql=sql_queries.sql_all_matches_limited, con=results_engine)
    else:
        matches = pd.read_sql(sql=sql_queries.sql_all_matches_limited, con=results_engine)
    matches = matches[matches['Time'] > timestamp]
    return matches


def find_one_match(asian_odds_engine, match_id):
    matches = pd.read_sql(sql=sql_queries.sql_one_match.format(id=match_id), con=asian_odds_engine)
    return matches


def find_result(home, away, res_engine, time):
    translated_home, translated_away = maping.translate(ao_name_home=home, ao_name_away=away)
    translated_time = maping.change_date(AO_date=str(time))
    return translated_home, translated_away, pd.read_sql(
        sql_queries.sql_results.format(h=translated_home, a=translated_away, t=translated_time),
        con=res_engine), translated_time


def calculate_mapping(match_ids=None, results_engine=None, res_engine=None):
    timestamp = '2022-11-13 00:00:00'
    succ = 1
    so_far = 0
    if match_ids is None:
        match_ids = find_all_completed_matches(results_engine=results_engine, timestamp=timestamp)
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


def time_parse(home_time, away_time):
    home_time = home_time['Season'].split("/")
    away_time = away_time['Season'].split("/")
    ln_h = len(home_time)
    ln_a = len(away_time)
    # f.e 2019/2020 -> 2020
    home_time = home_time[ln_h - 1]
    away_time = away_time[ln_a - 1]
    return home_time, away_time


def res_test(res):
    if res.empty:
        return False
    try:
        res.to_numpy()[0][0]
    except AttributeError:
        return False
    return True


def get_lambdas(home_team: str, away_team: str, league: str, time, lambdas_h: pd.DataFrame, lambdas_a: pd.DataFrame):
    try:
        away_lambdas = lambdas_a.loc[lambdas_a['Away'] == away_team]
        away_lambda = away_lambdas.iloc[-1]['lambda_a']
        away_time = away_lambdas.iloc[-1][['Season']]
        home_lambdas = lambdas_h.loc[lambdas_h['Home'] == home_team]
        home_lambda = home_lambdas.iloc[-1]['lambda_h']
        home_time = home_lambdas.iloc[-1][['Season']]
    except:
        return None, None, None, None
    return float(home_lambda), float(away_lambda), home_time, away_time


def find_ids(engine, result_engine):
    home_lambdas = pd.read_csv('csv_files/home.csv')
    away_lambdas = pd.read_csv('csv_files/away.csv')
    match_ids = find_all_completed_matches(results_engine=engine)
    ids = []
    for match_id, home, away, time, league in match_ids[['MatchID', 'Home', 'Away', 'Time', 'League']].to_numpy():
        be_home, be_away, res, _ = find_result(home=home, away=away, res_engine=result_engine, time=time)
        if not res_test(res):
            continue
        lambda_h, lambda_a, home_time, away_time = get_lambdas(home_team=be_home, away_team=be_away, time=time,
                                                               league=league,
                                                               lambdas_a=away_lambdas, lambdas_h=home_lambdas)
        if home_time is None or away_time is None:
            continue
        ids.append(match_id)
    return ids


def arbitrage_profit(p=1, match_ids=None, timestamp='2022-09-27 21:00:00.000000', engine=None, res_engine=None,
                     weighted=False, weights=None, points=11, lambda_weighted=False, uniform_weights=False, min_bet=0,
                     max_bet=1000, threshold=0):
    """

    :param p: probability for p-arbitrage program
    :param match_ids: if only some ids should be considered
    :param timestamp: earliest date considered
    :param engine: database engine
    :param res_engine: second database engine
    :param weighted: bool whether program should use weighted variants
    :param weights: weights
    :param points: maximum number of points
    :param lambda_weighted: whether program should be lambda weighted
    :param uniform_weights: whether program should use uniform weights
    :param min_bet: min bet parameter of  linear program
    :param max_bet: max bet parameter of linear program
    :param threshold: threshold for negative outcome program
    :return: profits, matchids, betting strategy
    """
    profits = []
    betting_vectors = []
    """Load lambdas from local file"""
    home_lambdas = pd.read_csv('csv_files/home.csv')
    away_lambdas = pd.read_csv('csv_files/away.csv')
    matchids_profit = []

    """ When match ids are not specified load all available"""
    if match_ids is None:
        match_ids = find_all_completed_matches(results_engine=engine, timestamp=timestamp)
    else:
        matches = find_all_completed_matches(results_engine=engine, timestamp='2022-01-01 21:00:00')
        match_ids = matches[matches['MatchID'].isin(match_ids)]
    possible_results = [str(j) + '-' + str(i) for i in range(points) for j in range(points)]
    "remove rows for p-arbitrage"
    rows_to_stay, rows_to_drop, probabilities = indices_threshhold(p=p, probabilities=None)
    "create weights if weights are global"
    if weighted and not lambda_weighted:
        weights = probabilities[rows_to_stay]
        if uniform_weights:
            weights[:] = 1 / (points + 1) ** 2
    "solve each match one by one"
    for match_id, home, away, time, league in match_ids[['MatchID', 'Home', 'Away', 'Time', 'League']].to_numpy():
        "Several checks whether data are correct"
        if '\'' in home or '\'' in away:
            continue
        "find result and mapping of teams"
        be_home, be_away, res, _ = find_result(home=home, away=away, res_engine=res_engine, time=time)
        "check if result is in correct format"
        if not res_test(res):
            continue
        res = res.to_numpy()[0][0]
        """Load lambdas if available"""
        if weighted and lambda_weighted:
            lambda_h, lambda_a, _, _ = get_lambdas(home_team=be_home, away_team=be_away, time=time, league=league,
                                                   lambdas_a=away_lambdas, lambdas_h=home_lambdas)
            if lambda_h is None:
                continue
            """Create weights for lambda"""
            weights = poisson(lambda_h=lambda_h, lambda_a=lambda_a, points=points)[rows_to_stay]
        timestamp = None

        arb = arbitrage.Arbitrage(engine, schemas=schemas, markets=markets, bookmakers=[], moving_odds=False,
                                  max_bet=max_bet, min_bet=min_bet)
        """Get 1x2 market odds"""
        xodds = ut.load_asian_odds(engine=engine, MatchID=match_id, timestamp=timestamp, market='1x2')
        if xodds.empty:
            continue
        """parse 1x2 odds and transform them in to correct format"""
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
            """Same but for different markets"""
            for market in markets:
                dif_odds = ut.load_asian_odds(engine=engine, MatchID=match_id, timestamp=timestamp, market=market)
                columns, match_rows = arb.bookmaker_filter_asian_odds(odds=dif_odds, market=market)
                for inner_index, match_row in match_rows.iterrows():
                    matrix_A = arb.append_columns(matrix_A=matrix_A, market=market, bookmaker="",
                                                  markets_cols=markets_cols,
                                                  index=inner_index, match_row=match_row, columns=columns)
            matrix_B = matrix_A.copy()
            matrix_A.drop(rows_to_drop, inplace=True)
            """Select correct program to run"""
            if weighted:
                succ, x, z = arb.solve_maxprofit_gurobi(matrix_A=matrix_A, MatchID=None, verbose=False, weights=weights,
                                                        threshold=threshold)
            else:
                succ, x, _ = arb.solve_maxprofit_gurobi(matrix_A=matrix_A, MatchID=None, verbose=False, weights=weights)
            if succ:
                """If solution was found save it"""
                res_vector = matrix_B.loc[matrix_B['PosState'] == res].to_numpy()[0][1:]
                profit = ret_profit(res_vector=res_vector, weighted=weighted, x=x)
                profits.append(profit)
                matchids_profit.append(match_id)
                betting_vectors.append(x)
    return profits, betting_vectors, matchids_profit


def find_outcome(res: str):
    print(res)
    h, a = res.split('-')
    h = int(h)
    a = int(a)
    return h - a


def ret_profit(res_vector, weighted, x):
    if weighted:
        return np.dot(res_vector, np.array(x))
    return np.dot(res_vector, np.array(x[:-1]))


def preprocess(results_engine):
    lambdas = pd.read_csv('cleaned_lambdas.csv')
    all_matches = pd.read_sql(sql_queries.sql_all_results, results_engine)
    ids = lambdas["MatchID"].values
    print(all_matches[all_matches['MatchID'].isin(ids)])


def create_matrix_a_all_markets(home, draw, away, overs, unders, home_handicaps, away_handicaps, points):
    points += 1
    possible_results = [str(j) + '-' + str(i) for i in range(points) for j in range(points)]
    d, h, a = hlp.x_market_table(points=points)
    d = draw * d - 1
    h = home * h - 1
    a = away * a - 1
    result_tables = {'PosState': possible_results, 'draw': d[:, 0], 'home': h[:, 0], 'away': a[:, 0]}
    for threshold in overs.keys():
        over, under = hlp.ou_result_table(totals=[float(threshold)], points=points)
        result_tables['over' + ' ' + threshold] = (over * overs[threshold] - 1)[:, 0]
        result_tables['under' + ' ' + threshold] = (under * unders[threshold] - 1)[:, 0]
    for handicaps in home_handicaps.keys():
        handicap_home, handicap_away = hlp.ah_result_table([float(handicaps)], points=points)
        result_tables['ah_home' + ' ' + handicaps] = (home_handicaps[handicaps] * handicap_home - 1)[:, 0]
        result_tables['ah_away' + '' + '-' + handicaps] = (away_handicaps['-' + handicaps] * handicap_away - 1)[:, 0]
    matrix_A = pd.DataFrame(result_tables)
    print(matrix_A)


def create_matrix_a(home, draw, away, points):
    points += 1
    possible_results = [str(j) + '-' + str(i) for i in range(points) for j in range(points)]
    d, h, a = hlp.x_market_table(points=points)
    d = draw * d - 1
    h = home * h - 1
    a = away * a - 1
    matrix_A = pd.DataFrame({'PosState': possible_results, 'draw': d[:, 0], 'home': h[:, 0], 'away': a[:, 0]})
    # print(matrix_A)
    return matrix_A


def calculate_profit_easy_example(bets: np.ndarray, odds: np.ndarray, points: int):
    d, h, a = hlp.x_market_table(points=points + 1)
    wins = bets * odds
    sm = bets.sum()
    d = wins[0] * d
    h = wins[1] * h
    a = wins[2] * a
    overall = np.array((d + h + a - sm).reshape((points + 1, points + 1)))
    return overall


def simple_test_real_probs(lambda_h_book, lambda_a_book, lambda_a_model, lambda_h_model, real_lambda_h, real_lambda_a,
                           arb, points=3, threshold=0):
    """

    :param lambda_h_book: bookmaker lambda of home team
    :param lambda_a_book:  bookmaker lambda of away team
    :param lambda_a_model:  our model lambda of away team
    :param lambda_h_model: our model lambda of home team
    :param real_lambda_h: real lambda
    :param real_lambda_a: home lambda
    :param arb: Arbitrage object
    :param points: maximum points allowed
    :param threshold: threhsold of negative outcome program
    :return: profit
    """
    real_probs = ps.lambda_to_prob(lambda_h=real_lambda_h, lambda_a=real_lambda_a)
    props = np.array(ps.lambda_to_prob(lambda_h=lambda_h_book, lambda_a=lambda_a_book))
    odds = 1 / props - 0.03
    margin = sum(1 / odds) - 1
    matrix_A = create_matrix_a(odds[0], odds[1], odds[2], points)
    true_props = ps.poisson(lambda_h=real_lambda_h, lambda_a=real_lambda_a, points=points + 1)
    model = ps.poisson(lambda_h=lambda_h_model, lambda_a=lambda_a_model, points=points + 1)
    w = model
    _, x, _ = arb.solve_maxprofit_gurobi(matrix_A=matrix_A, weights=None, threshold=threshold)
    _, x_w, z_w = arb.solve_maxprofit_gurobi(matrix_A=matrix_A, weights=w, threshold=threshold)
    result = sum(true_props * calculate_profit_easy_example(x_w, odds, points=points).reshape(-1))
    result_basic = sum(true_props * calculate_profit_easy_example(x[:-1], odds, points=points).reshape(-1))
    return result, result_basic, x[:-1], x_w


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
    for threshold in np.linspace(0.01, 0.1, num=10):
        splits = ps.load_txt(k=10)
        profits, betting_vectors, matchids_profit = arbitrage_profit(engine=engine, res_engine=result_engine,
                                                                     match_ids=splits, weighted=True,
                                                                     lambda_weighted=True, threshold=threshold)
        with open("profits_threshold_{t}.txt".format(t=threshold), 'w') as f:
            f.write(str(profits))
