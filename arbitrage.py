import copy

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from utils import helper_functions as hlp
from utils import utils

all_markets = ['1x2', 'bts', 'ou', 'ah', 'dc']
all_bookmakers = ['10Bet', '1xBet', '888sport', 'Betfair', 'Betway', 'bwin', 'ComeOn', 'Unibet', 'William Hill',
                  'youwin', 'Betfair Exchange', 'Pinnacle', 'bet365', 'Paddy Power', '188BET', 'Betclic', 'bet-at-home',
                  'Betsafe', 'Betsson', 'SBOBET', 'Interwetten', 'BetVictor']
markets_cols = {'1x2': ['draw', 'home', 'away'], 'bts': ['NO', 'YES'], 'ou': ['Under', 'Over'], 'ah': ['home', 'away'],
                'dc': ['homedraw', 'homeaway', 'awaydraw']}  # order is important
odds_type = {'closing': 0, 'opening': 1}

# 10Bet, 188BET, Betsafe
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

exit_status = {0: 'Optimization terminated successfully', 1: 'Iteration limit reached.',
               2: 'Problem appears to be infeasible.',
               3: 'Problem appears to be unbounded.', 4: 'Numerical difficulties encountered.'}
unique_cols = '","'.join(['MatchID', 'Timestamp'] + all_markets + all_bookmakers)


class Arbitrage:
    def __init__(self, db, schemas, markets, bookmakers, min_bet=5, max_bet=1000, moving_odds=True):
        self.db = db
        self.schemas = schemas
        self.markets = markets
        self.bookmakers = bookmakers
        # self.timelimit = timelimit
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.points = 11
        self.moving_odds = moving_odds
        if len(self.bookmakers) == 1:
            self.table_name = 'Arbitrage_1BM'
        else:
            self.table_name = 'Arbitrage_moreBM'

    def find_arbitrages(self):
        for schema in self.schemas:
            if self.moving_odds:
                odds_done_table = 'Matches_movingOdds'
            else:
                odds_done_table = 'Matches'
            MatchIDs = pd.read_sql_table(odds_done_table, self.db, schema=schema)  # ['MatchID']

            # try:
            #     arbitrages = pd.read_sql_table(self.table_name, self.db,
            #                                schema=schema)  # check if data are already analyzed
            #     arbitrages = arbitrages[
            #         (arbitrages['Min_bet'] == self.min_bet) & (arbitrages['Timelimit'] == self.timelimit)
            #         & (arbitrages['Max_bet'] == self.max_bet) & (arbitrages[self.bookmakers].all(axis='columns'))
            #         & (arbitrages[self.markets].all(axis='columns'))]
            #     MatchIDs = pd.merge(MatchIDs, arbitrages, on=['MatchID'], how="outer", indicator=True)
            #     MatchIDs = MatchIDs[MatchIDs['_merge'] == 'left_only']
            # except ValueError:
            #     pass

            for MatchID in set(MatchIDs['MatchID']):
                print('Looking for arbitrage for ', MatchID)
                matrixes_dict = self.create_matrix(schema, MatchID)
                # if bool(matrixes_dict):  # matrix_A.shape[1] > 1:  # there are some data in Matrix_A
                self.solve_maxprofit_gurobi(matrixes_dict, MatchID, verbose=False, save2db=True)

        return True

    def asian_odds_create_matrix(self, MatchID):
        xmarkets = pd.read_sql(sql_x.format(matchid=MatchID), self.db)
        latest = xmarkets.iloc[-1]
        timestamp = latest['Timestamp']
        ahs = pd.read_sql(sql_ah.format(matchid=MatchID, timestamp=timestamp), self.db)
        ous = pd.read_sql(sql_ou.format(matchid=MatchID, timestamp=timestamp), self.db)
        hlp.x_market_table(self.points)
        # remove over/under in form 2-2.5
        ous = ous.loc[ous['Over'].str.contains('^\d\.\d$')]
        under, over = (hlp.ou_result_table(totals=ous['Over'].astype(float).to_numpy(), points=self.points))
        # remove all non half asian handicaps
        ahs = hlp.parse_handicap(ahs)
        table_one, table_two = hlp.ah_result_table(ahs['Handicap'].astype(float).to_numpy())

    def create_matrix(self, schema, MatchID, timestamp=False):
        """function to create matrix_A"""
        possible_results = [str(j) + '-' + str(i) for i in range(self.points) for j in range(self.points)]
        timestamps = dict()
        odds = dict()
        first_timestamp = pd.Timestamp('2030-01-01T12')
        last_timestamp = pd.Timestamp('2010-01-01T12')
        for market in self.markets:
            market_odds = utils.load_data(self.db, schema=schema, market=market, singleID=True, matchID=MatchID,
                                          limit=False)  # [odds_type[self.odds_type]]
            if market_odds is None:
                continue
            # if timestamp:
            #    market_odds = market_odds[market_odds['Timestamp'] == pd.Timestamp(timestamp)]
            odds[market] = market_odds
            if market_odds['Timestamp'].min() < first_timestamp:
                first_timestamp = market_odds['Timestamp'].min()
            if market_odds['Timestamp'].max() > last_timestamp:
                last_timestamp = market_odds['Timestamp'].max()
        if first_timestamp == pd.Timestamp('2010-01-01T12') or last_timestamp == pd.Timestamp('2030-01-01T12'):
            print('No odds for ', MatchID)
            return None
        # print(first_timestamp, last_timestamp)
        counter_timestamp = first_timestamp

        while counter_timestamp <= last_timestamp:
            timestamps[counter_timestamp] = [pd.DataFrame({'PosState': possible_results}),
                                             pd.Timestamp('2030-01-01T12')]
            counter_timestamp = counter_timestamp + pd.Timedelta(minutes=1)
        print('initialized list')
        # if timestamp:
        #    timestamps = {pd.Timestamp(timestamp): timestamps[pd.Timestamp(timestamp)]}
        for market in odds:
            for bookmaker in self.bookmakers:
                columns, match_rows = self.bookmaker_filter(odds[market], bookmaker, market)
                if match_rows.shape[0] > 0:
                    for index, match_row in match_rows.iterrows():
                        not_found = True
                        # if not timestamp:
                        odds_times = []
                        counter_timestamp = match_row.Timestamp
                        while counter_timestamp < match_row.NextTimestamp:
                            odds_times.append(counter_timestamp)
                            counter_timestamp = counter_timestamp + pd.Timedelta(minutes=1)
                        # else:
                        #    odds_times = [pd.Timestamp(timestamp)]
                        for times in odds_times:

                            # nexttimestamp = timestamps[times][0]
                            # #print((times - match_row.Timestamp).total_seconds()/60)
                            # # tady upravit to porovnavani - asi to nejdulezitejsi
                            # #if abs((times - match_row.Timestamp).total_seconds()/60) <= self.timelimit:
                            # if times <= match_row.Timestamp & match_row.Timestamp <= nexttimestamp:
                            #  print(times)
                            # if times == pd.Timestamp('2021-04-08 01:36:00'):
                            #     print(timestamps[times])
                            #     pass
                            nextTimestamp = timestamps[times][1]
                            if match_row.NextTimestamp < nextTimestamp:
                                nextTimestamp = match_row.NextTimestamp
                            timestamps[times] = [self.append_columns(
                                matrix_A=timestamps[times][0], market=market, bookmaker=bookmaker,
                                markets_cols=markets_cols,
                                index=index, match_row=match_row, columns=copy.deepcopy(columns)), nextTimestamp]
                            not_found = False
                            # if times == pd.Timestamp('2021-04-08 01:36:00'):
                            #     print(timestamps[times])
                            #     pass

                        if not_found:
                            nextTimestamp = timestamps[match_row.Timestamp][1]
                            if match_row.NextTimestamp < nextTimestamp:
                                nextTimestamp = match_row.NextTimestamp
                            timestamps[match_row.Timestamp] = [pd.DataFrame({'PosState': possible_results}),
                                                               match_row.NextTimestamp]
                            timestamps[match_row.Timestamp] = [self.append_columns(
                                matrix_A=timestamps[match_row.Timestamp][0], market=market, bookmaker=bookmaker,
                                markets_cols=markets_cols,
                                index=index, match_row=match_row, columns=copy.deepcopy(columns)), nextTimestamp]
        final_timestamps = dict()
        sorted_timestamps = sorted(timestamps.keys())
        # tst = timestamps[pd.Timestamp('2021-04-08 01:36:00.000000')]
        if len(sorted_timestamps) >= 1:
            final_timestamps[sorted_timestamps[0]] = timestamps[sorted_timestamps[0]]
        if len(sorted_timestamps) >= 2:
            for times in sorted_timestamps[1:]:
                if timestamps[times - pd.Timedelta(minutes=1)][0].equals(timestamps[times][0]):
                    continue
                final_timestamps[times] = timestamps[times]
                # del timestamps[times + pd.Timedelta(minutes=1)]
        if timestamp:
            final_timestamps = {pd.Timestamp(timestamp): final_timestamps[pd.Timestamp(timestamp)]}
        return final_timestamps

    def bookmaker_filter_asian_odds(self, odds, market):
        if market == 'ou':
            columns = hlp.market_tables(market=market, points=self.points, totals=odds['Over'].astype(float).to_numpy())
            match_rows = odds
        elif market == 'ah':
            columns = hlp.market_tables(market=market, points=self.points, handicap=odds['Handicap'])
            match_rows = odds
        else:
            columns = hlp.market_tables(market=market, points=self.points)
            match_rows = odds
        return columns, match_rows

    def bookmaker_filter(self, odds, bookmaker, market):
        if market == 'ou':
            totals = odds[odds['Bookmaker'] == bookmaker]['Total'].astype(float).reset_index(drop=True)
            columns = hlp.market_tables(market=market, points=self.points, totals=totals)
            match_rows = odds[odds['Bookmaker'] == bookmaker].reset_index(drop=True)
        elif market == 'ah':
            handicap = odds[odds['Bookmaker'] == bookmaker]['Handicap'].astype(float).reset_index(drop=True)
            columns = hlp.market_tables(market=market, points=self.points, handicap=handicap)
            match_rows = odds[odds['Bookmaker'] == bookmaker].reset_index(drop=True)
        else:
            columns = hlp.market_tables(market=market, points=self.points)
            match_rows = odds[odds['Bookmaker'] == bookmaker].reset_index(drop=True)
        return columns, match_rows

    def append_columns(self, matrix_A, market, bookmaker, markets_cols, index, match_row, columns):
        for i in range(len(columns)):
            if market == 'ou':
                column_name = '{}_{}_{}_{}'.format(bookmaker, market, markets_cols[market][i], match_row['Over'])
                column = self.create_column(columns[i][:, index], match_row, str(i))
            elif market == 'ah':
                column_name = '{}_{}_{}_{}'.format(bookmaker, market, markets_cols[market][i],
                                                   match_row['Handicap'])
                column = self.create_column(columns[i][:, index], match_row, str(i))
            else:
                column_name = '{}_{}_{}'.format(bookmaker, market, markets_cols[market][i])
                column = self.create_column(columns[i][:, 0], match_row, str(i))
            matrix_A[column_name] = column
        return matrix_A

    def create_column(self, column, match_row, market_column):
        try:
            a = float(match_row[market_column])
            pass
        except:
            pass
        column[column == 0] = - 1
        column[column == 1] = float(match_row[int(market_column)]) - 1
        column[column == -0.1] = 0
        return column

    def solve_maxprofit_gurobi(self, matrix_A, MatchID, verbose=False, save2db=False):
        # for timestamp in matrixes_dict:
        # matrix_A = matrixes_dict[timestamp][0]
        model, A, x, y = self.create_model(matrix_A, verbose)
        model.update()
        model.optimize()
        if model.status == GRB.INF_OR_UNBD:
            # Turn presolve off to determine whether model is infeasible
            # or unbounded
            model.setParam(GRB.Param.Presolve, 0)
            model.optimize()
        if verbose: self.print_stats(model, A, x, y, matrix_A, None)
        if model.status == GRB.OPTIMAL:
            return True,x.x
        return False,None
       #if save2db:
       #     self.save_modelinfo(model, matrix_A, MatchID, x, y, timestamp, matrixes_dict[timestamp][1])

    def create_model(self, matrix_A, verbose):
        try:
            A = matrix_A.iloc[:, 1:].to_numpy()
        except AttributeError:
            A = matrix_A
        N = A.shape[1]
        A = np.hstack((A, -np.ones((A.shape[0], 1))))
        model = gp.Model('LP')
        x = model.addMVar(shape=N + 1, vtype=GRB.CONTINUOUS, name="x")
        y = model.addMVar(shape=N, vtype=GRB.BINARY, name="y")
        model.addConstr(x[:-1] >= self.min_bet * y)
        model.addConstr(x[:-1] <= self.max_bet * y)
        model.addConstr(x[-1] >= 1)
        model.addConstr(A @ x >= np.zeros(A.shape[0]), name="c")
        if not verbose: model.setParam('LogToConsole', 0)
        model.setObjective(x[-1], GRB.MAXIMIZE)
        return model, A, x, y

    def print_stats(self, model, A, x, y, matrix_A, timestamp):
        if model.status == GRB.OPTIMAL:
            #print('Success', ' Status:', model.status)
            #print('Timestamp: ', timestamp)
            #print('Profit (%): ', x.x[-1] / sum(x.x[:-1]))
            print(x.x)
            #print(matrix_A)
            #print('Multiplication result min: ', min(A[:, :-1] @ x.x[:-1]))
            #print('Optimal objective: %g' % model.objVal)
            #active_cols = ['PosState']
            #for i in range(y.shape[0]):
            #    if y[i].x == 1:
            #        active_cols.append(matrix_A.columns[i + 1])
            #print(matrix_A.loc[:, active_cols])

        elif model.status != GRB.INFEASIBLE:
            print('Optimization was stopped with status %d' % model.status)
        else:  # Model is infeasible
            print('Model is infeasible')

    def save_modelinfo(self, model, matrix_A, MatchID, x, y, timestamp, nextTimestamp):
        if model.status == GRB.OPTIMAL:
            dct = {'MatchID': [MatchID], 'ArbPos': [True], 'Timestamp': [timestamp], 'NextTimestamp': [nextTimestamp],
                   'OpDuration[m]': [pd.Timedelta(nextTimestamp - timestamp).seconds / 60.0],
                   'Min_bet': self.min_bet, 'Max_bet': self.max_bet, 'Betsum': sum(x.x[:-1]),
                   'Profit_prop': x.x[-1] / sum(x.x[:-1])}  # pridat profit mean
            for i in range(y.shape[0]):  # or N
                if y[i].x == 1:
                    # print(matrix_A.columns[i + 1])
                    # dct[matrix_A.columns[i + 1].split('_', 1)[1]] = True
                    dct[matrix_A.columns[i + 1]] = True
        else:
            dct = {'MatchID': [MatchID], 'ArbPos': [False], 'Timestamp': [timestamp], 'NextTimestamp': [nextTimestamp],
                   'OpDuration[m]': [pd.Timedelta(nextTimestamp - timestamp).seconds / 60.0],
                   'Min_bet': self.min_bet, 'Max_bet': self.max_bet, 'Betsum': 0.0,
                   'Profit_prop': 0.0}
        for m in all_markets:
            dct[m] = m in self.markets
        for b in all_bookmakers:
            dct[b] = b in self.bookmakers
        try:
            db_columns = next(
                pd.read_sql_table(table_name=self.table_name, con=self.db, schema='football', chunksize=1))
            for col in db_columns:
                if col not in dct:
                    dct[col] = False
        except ValueError:
            all_columns = self.get_all_columns()
            for col in all_columns:
                if col not in dct:
                    dct[col] = False
            pass

        df = pd.DataFrame(dct)
        utils.upsert_table(df, self.table_name, self.db, 'football', update_on_conflict=True, update_cols=df.columns,
                           unique_cols=unique_cols)

    def get_all_columns(self):
        columns = []
        for b in self.bookmakers:
            for m in self.markets:
                # for m in all_markets:
                for mc in markets_cols[m]:
                    if m == 'ou':
                        for i in range(1, 11):
                            column_name = '{}_{}_{}_{}'.format(b, m, mc, str(i - 0.5))
                            columns.append(column_name)
                    elif m == 'ah':
                        for i in range(-9, 11):
                            column_name = '{}_{}_{}_{}'.format(b, m, mc, str(i - 0.5))
                            columns.append(column_name)
                    else:
                        column_name = '{}_{}_{}'.format(b, m, mc)
                        columns.append(column_name)
        return columns

#
# def solve_maxprofit(matrix_A, budget, MatchID, markets, bookmakers, db, odds_time):
#     matrix = matrix_A.iloc[:, 1:].to_numpy()
#     c = np.zeros(matrix.shape[1] + 1, dtype=np.int)
#     c[-1] = -1  # -1 because we are maximizing
#     b_ub = np.zeros(matrix.shape[0] + 1, dtype=np.int)
#     b_ub[-1] = budget
#     A = np.hstack((-matrix, np.ones((matrix.shape[0], 1))))
#     budget_vector = np.ones((1, matrix.shape[1] + 1))
#     budget_vector[-1] = 0
#     A = np.vstack((A, budget_vector))
#     bounds = [(0, budget) for i in range(A.shape[1] - 1)]
#     bounds.append((1, budget))
#     try:
#         res = optimize.linprog(c=c, A_ub=A, b_ub=b_ub, bounds=bounds, method='revised simplex')
#     except Exception as e:
#         print(e)
#         return
#     print('Scipy ', MatchID)
#
#     if res['success']:
#         print('Success: ', res['success'], ' Status:', exit_status[res['status']])
#         #print('result sum(x): ', sum(res['x'][:-1]))
#         print('Profit: ', res['x'][-1])
#         print('x: ', res['x'])
#         print('Profit (%): ', res['x'][-1] / sum(res['x'][:-1]))
#         print('Multiplication result min: ', min(matrix @ res['x'][:-1]))
#         print()
#         dct = {'MatchID': [MatchID], 'ArbPos': [True], 'Odds_type': [odds_time],
#                'Budget': budget, 'Betsum': sum(res['x'][:-1]), 'Profit_prop': res['x'][-1] / sum(res['x'][:-1]), }
#         for m in all_markets:
#             dct[m] = m in markets
#         for b in all_bookmakers:
#             dct[b] = b in bookmakers
#         df = pd.DataFrame(dct)
#         if len(bookmakers) == 1:
#             table_name = 'Arbitrage_1BM'
#         else:
#             table_name = 'Arbitrage_moreBM'
#         utils.upsert_table(df, table_name, db, 'football', update_on_conflict=True, update_cols=df.columns,
#                            unique_cols=unique_cols)
#     else:
#         print('Not success: ',exit_status[res['status']])
#         print()
#
