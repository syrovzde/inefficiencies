import sql_queries
from arbitrage_main import find_all_completed_matches, get_lambdas
from arbitrage_main import find_result
from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine
from arbitrage_main import indices_threshhold
from arbitrage_main import res_test
from arbitrage_main import ret_profit
import utils.utils as ut
import arbitrage
import pandas as pd
from poisson import poisson
csv_file_with_mapping = 'csv_files/bet_AsianOdds_AO2BE.csv'
schemas = ['asianodds']
# markets = ['1x2', 'ou', 'ah']
markets = ['ou', 'ah']
ip = "147.32.83.171"
markets_cols = {'1x2': ['draw', 'home', 'away'], 'bts': ['NO', 'YES'], 'ou': ['Over', 'Under'], 'ah': ['home', 'away'],
                'dc': ['homedraw', 'homeaway', 'awaydraw']}

"""ids chosen as test data """
match_ids = ['1562868529', '1563026767', '1563108486', '1563032872', '1124895', '1125355', '1125355', '1125355',
             '1125355', '1125355', '1125355', '1069731784', '-818990875', '-818990875', '-818990875', '-818990875',
             '1563196446', '1563173599', '1562944376', '1563026806', '1563272518', '1563272518', '1563076444',
             '1563076444', '1563076041', '1563075993', '1563076275', '1563076246', '1563137534', '1563137534',
             '1563137534', '1563137534', '1563137534', '1563137534', '1563137534', '1563137534', '1563137534',
             '1563278866', '1563278866', '1563290671', '1563344125', '1563327032', '1563327032', '1563330733',
             '1563278880', '1563278880', '1563278909', '1563316090', '1563316090', '1563347359', '1563347359',
             '1563309882', '1563309882', '1563309882', '1563278997', '1563278997', '1563415916', '1563212686',
             '1563279346', '1563447847', '1563170318', '1563459710', '1563479791', '1563287438', '1563287438',
             '1563298350', '1563484566', '1563330633', '1563330225', '1563330402', '1563330204', '1563330204',
             '1563330435', '1563330392', '1563330392', '1563330303', '1563330303', '1563330303', '1563567614',
             '1563567614', '1563330411', '1563330411', '1563582455', '1563548691', '1563548349', '1563548696',
             '1563567088', '1563567088', '1563567088', '1563567088', '1563567088', '1563567088', '1563567088',
             '1563567088', '1563576622', '1563611391', '1563611391', '1563610244', '1563631135', '1563610241',
             '1563631133', '1563610237', '1563436869', '1563673298', '1563673298', '1563673296', '1563700816',
             '1563714589', '1563781645', '1564030349', '1563789328', '1563972354', '1563946046', '1563941912',
             '1563861593', '1563960777', '1563861358', '1563821891', '1564011560', '1564012012', '1564012012',
             '1564012012', '1564012012', '1563861184', '1563861055', '1563861055', '1563994527', '1564034875',
             '1564083301', '1563934490', '1564039918', '1564133487', '1564133487']



def exctract_results(res_engine):
    all_results = {}
    all_results['matchid'] = []
    all_results['res'] = []
    match_ids = pd.read_csv('match_ids_sample.csv')
    i=0
    for match_id, home, away, time in match_ids[['MatchID', 'Home', 'Away', 'Time']].to_numpy():
        be_home, be_away, res, _ = find_result(home=home, away=away, res_engine=res_engine, time=time)
        if not res_test(res):
            continue
        res = res.to_numpy()[0][0]
        all_results['matchid'].append(match_id)
        all_results['res'].append(res)
    df = pd.DataFrame(all_results)
    df = df.set_index('matchid')
    df.to_csv('results_sample.csv')
def extract_csv(engine, match_ids):
    baseline = "{k}_sample.csv"
    matches = find_all_completed_matches(results_engine=engine, timestamp='2022-01-01 21:00:00')
    ids = matches[matches['MatchID'].isin(match_ids)]
    ids.to_csv(baseline.format(k='match_ids'), index=False)
    x2 = pd.read_sql(sql_queries.sql_all_1x2, con=engine)
    x2 = x2[x2['MatchID'].isin(match_ids)]
    x2.to_csv(baseline.format(k='1x2'), index=False)
    ah = pd.read_sql(sql_queries.sql_all_ah, con=engine)
    ah = ah[ah['MatchID'].isin(match_ids)]
    ah.to_csv(baseline.format(k='ah'), index=False)
    ou = pd.read_sql(sql_queries.sql_all_ou, con=engine)
    ou = ou[ou['MatchID'].isin(match_ids)]
    ou.to_csv(baseline.format(k='ou'), index=False)

def arbitrage_profit_tutorial(p=1,
                     weighted=False, weights=None, points=5, lambda_weighted=False, uniform_weights=False, min_bet=0,
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
    results = pd.read_csv('results_sample.csv')
    matchids_profit = []

    """ When match ids are not specified load all available"""
    match_ids = pd.read_csv('match_ids_sample.csv')
    possible_results = [str(j) + '-' + str(i) for i in range(points) for j in range(points)]
    "remove rows for p-arbitrage"
    rows_to_stay, rows_to_drop, probabilities = indices_threshhold(p=p, probabilities=None,points=points+1)
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
        be_home, be_away = find_result(home=home, away=away, res_engine=None, time=time)
        "check if result is in correct format"
        res = results[results['matchid']==match_id]['res'].to_numpy()[0]
        """Load lambdas if available"""
        if weighted and lambda_weighted:
            lambda_h, lambda_a, _, _ = get_lambdas(home_team=be_home, away_team=be_away, time=time, league=league,
                                                   lambdas_a=away_lambdas, lambdas_h=home_lambdas)
            if lambda_h is None:
                continue
            """Create weights for lambda"""
            weights = poisson(lambda_h=lambda_h, lambda_a=lambda_a, points=points)[rows_to_stay]
        timestamp = None
        arb = arbitrage.Arbitrage(schemas=schemas, markets=markets, bookmakers=[],
                                  max_bet=max_bet, min_bet=min_bet,db=None,points=points)
        """Get 1x2 market odds"""
        xodds = ut.load_asian_odds(engine=None, MatchID=match_id, timestamp=timestamp, market='1x2',from_csv=True)
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
                dif_odds = ut.load_asian_odds(engine=None, MatchID=match_id, timestamp=timestamp, market=market,from_csv=True)
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



if __name__ == '__main__':
    arbitrage_profit_tutorial(points=11)
