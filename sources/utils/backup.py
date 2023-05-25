from sqlalchemy import create_engine
import argparse
import pandas as pd
from utils import utils
from utils import helper_functions as hlp
from scipy import optimize
import numpy as np
import gurobipy as gp
from gurobipy import GRB

all_markets = ['1x2', 'bts', 'ou', 'ah', 'dc']
all_bookmakers = ['10Bet', '1xBet', '888sport', 'Betfair', 'Betway', 'bwin', 'ComeOn', 'Unibet', 'William Hill',
                  'youwin', 'Betfair Exchange', 'Pinnacle', 'bet365', 'Paddy Power', '188BET', 'Betclic', 'bet-at-home',
                  'Betsafe', 'Betsson', 'SBOBET', 'Interwetten', 'BetVictor']
markets_cols = {'1x2': ['draw', 'home', 'away'], 'bts': ['NO', 'YES'], 'ou': ['Under', 'Over'], 'ah': ['away', 'home'],
                'dc': ['homedraw', 'homeaway', 'awaydraw']}  # order is important
odds_type = {'closing': 0, 'opening': 1}

# 10Bet, 188BET, Betsafe

exit_status = {0: 'Optimization terminated successfully', 1: 'Iteration limit reached.',
               2: 'Problem appears to be infeasible.',
               3: 'Problem appears to be unbounded.', 4: 'Numerical difficulties encountered.'}
unique_cols = '","'.join(['MatchID'] + all_markets + all_bookmakers)




def bookmaker_filter(odds, bookmaker, market, points):
    if market == 'ou':
        totals = odds[odds['Bookmaker'] == bookmaker]['Total'].astype(float).reset_index(drop=True)
        columns = hlp.market_tables(market=market, points=points, totals=totals)
        match_rows = odds[odds['Bookmaker'] == bookmaker].reset_index(drop=True)
    elif market == 'ah':
        handicap = odds[odds['Bookmaker'] == bookmaker]['Handicap'].astype(float).reset_index(drop=True)
        columns = hlp.market_tables(market=market, points=points, handicap=handicap)
        match_rows = odds[odds['Bookmaker'] == bookmaker].reset_index(drop=True)
    else:
        columns = hlp.market_tables(market=market, points=points)
        match_rows = odds[odds['Bookmaker'] == bookmaker].reset_index(drop=True)
    return columns, match_rows


def create_column(column, match_row, market_column):
    column[column == 1] = float(match_row[market_column]) - 1
    column[column == 0] = - 1
    return column


def append_columns(matrix_A, market, bookmaker, markets_cols, index, match_row, columns, points=11):
    for i in range(len(columns)):
        if market == 'ou':
            column_name = '{}-{}-{}-{}'.format(bookmaker, market, markets_cols[market][i], match_row['Total'])
            column = create_column(columns[i][:, index], match_row, str(i))
        elif market == 'ah':
            column_name = '{}-{}-{}-{}'.format(bookmaker, market, markets_cols[market][i],
                                               match_row['Handicap'])
            column = create_column(columns[i][:, index], match_row, str(i))
        else:
            column_name = '{}-{}-{}'.format(bookmaker, market, markets_cols[market][i])
            column = create_column(columns[i][:, index], match_row, str(i))
        matrix_A[column_name] = column
    return matrix_A


def create_matrix(db, schema, MatchID, markets, bookmakers, odds_time):
    points = 11
    possible_results = [str(j) + '-' + str(i) for i in range(points) for j in range(points)]
    matrix_A = pd.DataFrame({'PosState': possible_results})
    # function to create matrix_A
    for market in markets:
        odds = utils.load_data(db, schema=schema, market=market, singleID=True, matchID=MatchID, limit=False)[
            odds_type[odds_time]]
        if odds is None:
            continue
        for bookmaker in bookmakers:
            columns, match_rows = bookmaker_filter(odds, bookmaker, market, points)
            if match_rows.shape[0] > 0:
                for index, match_row in match_rows.iterrows():
                    matrix_A = append_columns(matrix_A=matrix_A, market=market, bookmaker=bookmaker,
                                              markets_cols=markets_cols,
                                              index=index, match_row=match_row, columns=columns, points=points)
    return matrix_A


def solve_feasibility(matrix_A, prop_profit, one_bet, MatchID, markets, bookmakers, db, odds_time):
    matrix = matrix_A.iloc[:, 1:].to_numpy()
    c = np.ones(matrix.shape[1], dtype=np.uint)
    b_ub = np.zeros(matrix.shape[0], dtype=np.uint)
    A = -(matrix - prop_profit)
    res = optimize.linprog(c=c, A_ub=A, b_ub=b_ub, bounds=(1, one_bet), method='interior-point')

    # print('Matrix shape:', matrix.shape)
    # print('Success: ', res['success'], '\nStatus:', exit_status[res['status']])
    if res['success']:
        print('Success: ', res['success'], '\nStatus:', exit_status[res['status']])
        dct = {'MatchID': [MatchID], 'ArbPos': [True], 'Odds_type': [odds_time]}
        for m in all_markets:
            dct[m] = m in markets
        for b in all_bookmakers:
            dct[b] = b in bookmakers
        df = pd.DataFrame(dct)
        utils.upsert_table(df, 'Arbitrage', db, 'football', update_on_conflict=True, update_cols=['ArbPos'],
                           unique_cols=None)
        # m = pd.read_sql_table('Arbitrage', db, schema='football')['Markets'][0][1:-2].split(',')
        # print('Second Multiplication result:', matrix @ (res['x']*100))


def get_optparams(matrix, model, min_bet, max_bet):
    c = np.zeros(matrix.shape[1] + 1, dtype=np.int)
    c[-1] = -1  # -1 because we are maximizing
    b_ub = np.zeros(matrix.shape[0] + 1, dtype=np.int)
    b_ub[-1] = max_bet
    A = np.hstack((-matrix, np.ones((matrix.shape[0], 1))))
    budget_vector = np.ones((1, matrix.shape[1] + 1))
    budget_vector[-1] = 0
    A = np.vstack((A, budget_vector))

    y_vec = model.addMVar(len(c) - 1, vtype=gp.GRB.BINARY, name="y")
    lb = np.ones(len(c)) * min_bet
    lb[-1] = 1
    ub = np.ones(len(c)) * max_bet
    ub[-1] = GRB.INFINITY
    for i in range(len(c) - 1):
        print(y_vec[i])
        lb[i] = min_bet * y_vec[i]
        ub[i] = max_bet * y_vec[i]
    return A, c, lb, ub, b_ub


def solve_maxprofit_gurobi(matrix_A, min_bet, max_bet, MatchID, markets, bookmakers, db, odds_time, verbose=False,
                           save2db=True):
    matrix = matrix_A.iloc[:, 1:].to_numpy()

    model = gp.Model('LP')

    A, c, lb, ub, b_ub = get_optparams(matrix, model, min_bet, max_bet)
    if not verbose: model.setParam('LogToConsole', 0)

    x = model.addMVar(len(c), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="x")

    # model.addMConstr(np.eye(len(c)),x,GRB.LESS_EQUAL, y*max_bet)
    model.addMConstr(A, x, GRB.LESS_EQUAL, b_ub)
    model.setObjective(c @ x)
    model.update()
    model.optimize()

    if model.status == GRB.INF_OR_UNBD:
        # Turn presolve off to determine whether model is infeasible
        # or unbounded
        model.setParam(GRB.Param.Presolve, 0)
        model.optimize()
    if verbose:
        if model.status == GRB.OPTIMAL:
            print('Success', ' Status:', model.status)
            # print('result sum(x): ', sum(res['x'][:-1]))
            print('Profit: ', x.x[-1])
            model.printAttr('x')
            print('Profit (%): ', x.x[-1] / sum(x.x[:-1]))
            print('Multiplication result min: ', min(matrix @ x.x[:-1]))
            # print('Optimal objective: %g' % model.objVal)
        elif model.status != GRB.INFEASIBLE:
            print('Optimization was stopped with status %d' % model.status)
        else:  # Model is infeasible
            print('Model is infeasible')
        print()

    # saving to db
    if model.status == GRB.OPTIMAL and save2db:
        dct = {'MatchID': [MatchID], 'ArbPos': [True], 'Odds_type': [odds_time],
               'Budget': budget, 'Betsum': sum(x.x[:-1]), 'Profit_prop': x.x[-1] / sum(x.x[:-1])}
        for m in all_markets:
            dct[m] = m in markets
        for b in all_bookmakers:
            dct[b] = b in bookmakers
        df = pd.DataFrame(dct)
        if len(bookmakers) == 1:
            table_name = 'Arbitrage_1BM'
        else:
            table_name = 'Arbitrage_moreBM'
        utils.upsert_table(df, table_name, db, 'football', update_on_conflict=True, update_cols=df.columns,
                           unique_cols=unique_cols)


def solve_maxprofit(matrix_A, budget, MatchID, markets, bookmakers, db, odds_time):
    matrix = matrix_A.iloc[:, 1:].to_numpy()
    c = np.zeros(matrix.shape[1] + 1, dtype=np.int)
    c[-1] = -1  # -1 because we are maximizing
    b_ub = np.zeros(matrix.shape[0] + 1, dtype=np.int)
    b_ub[-1] = budget
    A = np.hstack((-matrix, np.ones((matrix.shape[0], 1))))
    budget_vector = np.ones((1, matrix.shape[1] + 1))
    budget_vector[-1] = 0
    A = np.vstack((A, budget_vector))
    bounds = [(0, budget) for i in range(A.shape[1] - 1)]
    bounds.append((1, budget))
    try:
        res = optimize.linprog(c=c, A_ub=A, b_ub=b_ub, bounds=bounds, method='revised simplex')
    except Exception as e:
        print(e)
        return
    print('Scipy ', MatchID)

    if res['success']:
        print('Success: ', res['success'], ' Status:', exit_status[res['status']])
        # print('result sum(x): ', sum(res['x'][:-1]))
        print('Profit: ', res['x'][-1])
        print('x: ', res['x'])
        print('Profit (%): ', res['x'][-1] / sum(res['x'][:-1]))
        print('Multiplication result min: ', min(matrix @ res['x'][:-1]))
        print()
        dct = {'MatchID': [MatchID], 'ArbPos': [True], 'Odds_type': [odds_time],
               'Budget': budget, 'Betsum': sum(res['x'][:-1]), 'Profit_prop': res['x'][-1] / sum(res['x'][:-1]), }
        for m in all_markets:
            dct[m] = m in markets
        for b in all_bookmakers:
            dct[b] = b in bookmakers
        df = pd.DataFrame(dct)
        if len(bookmakers) == 1:
            table_name = 'Arbitrage_1BM'
        else:
            table_name = 'Arbitrage_moreBM'
        utils.upsert_table(df, table_name, db, 'football', update_on_conflict=True, update_cols=df.columns,
                           unique_cols=unique_cols)
    else:
        print('Not success: ', exit_status[res['status']])
        print()


def find_arbitrages(db, schemas, markets, bookmakers, odds_time, min_bet, max_bet):
    for schema in schemas:
        MatchIDs = pd.read_sql_table('Odds_done', db, schema=schema)['MatchID']
        # k = 20900
        # analyzed_data = 0
        for MatchID in MatchIDs:
            # MatchID = 'rokpY26c'
            # if k%100==0: #pro all bookies je to finished do 285000
            #    print('k: ', k, ' analyzed: ', analyzed_data)
            # k = k + 1
            matrix_A = create_matrix(db, schema, MatchID, markets, bookmakers, odds_time)
            if matrix_A.shape[1] > 1:  # there are some data in Matrix_A
                solve_maxprofit_gurobi(matrix_A, min_bet, max_bet, MatchID, markets, bookmakers, db, odds_time,
                                       verbose=True, save2db=True)
                # solve_maxprofit(matrix_A, budget, MatchID, markets, bookmakers, db, odds_time)

                # analyzed_data += 1

    return True
