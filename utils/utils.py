import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert
import utils.helper_functions as hlp

# import analysis.market_efficiency.Tests
import argparse
import sqlalchemy
import time
import sys

# import analysis.market_efficiency.utils.odds2probs as marg
# from scipy.optimize import linprog


market_choices = {'1x2': 3, 'ah': 2, 'ou': 2, 'bts': 2, 'dc': 3, 'ha': 2}

sql_x = """ SELECT \"x\",\"1\",\"2\",\"Timestamp\" FROM asianodds.\"Odds_1x2_FT\" WHERE \"Odds_1x2_FT\".\"MatchID\" = \'{matchid}\'
"""

sql_ou = """SELECT 
\"Over\",\"1\" as \"0\",\"2\" as \"1\" FROM asianodds.\"Odds_ou_FT\" WHERE \"Odds_ou_FT\".\"MatchID\" = \'{matchid}\' AND
\"Odds_ou_FT\".\"Timestamp\" = \'{timestamp}\'
"""
sql_ah = """ SELECT 
\"Handicap\",\"1\" as \"0\" ,\"2\" as \"1\" FROM asianodds.\"Odds_ah_FT\" WHERE \"Odds_ah_FT\".\"MatchID\" = \'{matchid}\' AND
\"Odds_ah_FT\".\"Timestamp\" = \'{timestamp}\'
"""

def get_select_script(market, schema, singleID, matchID, limit=True, moving_odds=True):
    """
    Loads data from database
    :param market:
    :param schema:
    :return:
    """
    if limit:
        lim = 'LIMIT 100000'
    else:
        lim = ''
    if moving_odds:
        matches_suffix = '_movingOdds'
        x_col = 'x'

    else:
        matches_suffix = ''
        x_col = 'X'
    if not singleID:
        if market == '1x2':  # pridat ty gap before a udelat to robustnejsi a hezci
            return 'SELECT "Matches{x}"."MatchID","1","1_gap_before", "{m}" as "0", "{m}_gap_before" "2", "2_gap_before","Bookmaker","Timestamp" FROM {y}.' '"Matches{x}" ' \
                   'INNER JOIN {y}."Odds_{z}{x}" on "Odds_{z}{x}"."MatchID" = "Matches{x}"."MatchID" '.format(
                x=matches_suffix, y=schema, z=market, m=x_col) + lim
        if market == 'ah':
            return 'SELECT "Matches{x}"."MatchID","1" as "0" , "2" as "1","Bookmaker","Handicap","Timestamp" FROM {y}.' '"Matches{x}" ' \
                   'INNER JOIN {y}."Odds_{z}{x}" on "Odds_{z}{x}"."MatchID" = "Matches{x}"."MatchID" '.format(
                x=matches_suffix, y=schema, z=market) + lim
        if market == 'bts':
            return 'SELECT "Matches{x}"."MatchID","YES" as "1", "NO" as "0","Bookmaker","Timestamp" FROM {y}.' '"Matches{x}" ' \
                   'INNER JOIN {y}."Odds_{z}{x}" on "Odds_{z}{x}"."MatchID" = "Matches{x}"."MatchID" '.format(
                x=matches_suffix, y=schema, z=market) + lim
        if market == 'ou':
            return 'SELECT "Matches{x}"."MatchID","Under" as "0", "Over" as "1","Total","Bookmaker","Details","Timestamp" FROM {y}.' '"Matches{x}" ' \
                   'INNER JOIN {y}."Odds_{z}{x}" on "Odds_{z}{x}"."MatchID" = "Matches{x}"."MatchID" '.format(
                x=matches_suffix, y=schema, z=market) + lim
        if market == 'dc':
            return 'SELECT "Matches{x}"."MatchID","1X" as "0","12" as "1","X2" as "2","Bookmaker","Timestamp" FROM {y}.' '"Matches{x}" ' \
                   'INNER JOIN {y}."Odds_{z}{x}" on "Odds_{z}{x}"."MatchID" = "Matches{x}"."MatchID" '.format(
                x=matches_suffix, y=schema, z=market) + lim
        if market == 'ha':
            return 'SELECT "Matches{x}"."MatchID","1" , "2" as "0", "Bookmaker","Timestamp" FROM {y}.' '"Matches{x}" ' \
                   'INNER JOIN {y}."Odds_{z}{x}" on "Odds_{z}{x}"."MatchID" = "Matches{x}"."MatchID" '.format(
                x=matches_suffix, y=schema, z=market) + lim
        # if market == '1x2':
        #     return 'SELECT "Matches"."MatchID","1", "X" as "0", "2", "Result","Bookmaker","Timestamp" FROM {y}.' '"Matches" ' \
        #            'INNER JOIN {y}."Odds_1x2" on "Odds_1x2"."MatchID" = "Matches"."MatchID" '.format(y=schema) + lim
        # if market == 'ah':
        #     return 'SELECT "Matches"."MatchID","1" as "0" , "2" as "1", "Result","Bookmaker","Handicap","Timestamp" FROM {y}.' '"Matches" ' \
        #            'INNER JOIN {y}."Odds_ah" on "Odds_ah"."MatchID" = "Matches"."MatchID" '.format(y=schema) + lim
        # if market == 'bts':
        #     return 'SELECT "Matches"."MatchID","YES" as "1", "NO" as "0", "Result","Bookmaker","Timestamp" FROM {y}.' '"Matches" ' \
        #            'INNER JOIN {y}."Odds_bts" on "Odds_bts"."MatchID" = "Matches"."MatchID" '.format(y=schema) + lim
        # if market == 'ou':
        #     return 'SELECT "Matches"."MatchID","Under" as "0", "Over" as "1","Total", "Result","Bookmaker","Details","Timestamp" FROM {y}.' '"Matches" ' \
        #            'INNER JOIN {y}."Odds_ou" on "Odds_ou"."MatchID" = "Matches"."MatchID" '.format(y=schema) + lim
        # if market == 'dc':
        #     return 'SELECT "Matches"."MatchID","1X" as "0","12" as "1","X2" as "2", "Result","Bookmaker","Timestamp" FROM {y}.' '"Matches" ' \
        #            'INNER JOIN {y}."Odds_dc" on "Odds_dc"."MatchID" = "Matches"."MatchID" '.format(y=schema) + lim
        # if market == 'ha':
        #     return 'SELECT "Matches"."MatchID","1" , "2" as "0", "Result","Bookmaker","Timestamp" FROM {y}.' '"Matches" ' \
        #            'INNER JOIN {y}."Odds_ha" on "Odds_ha"."MatchID" = "Matches"."MatchID" '.format(y=schema) + lim
    else:
        matchID = "'" + str(matchID) + "'"
        if market == '1x2':
            return 'SELECT "Matches{x}"."MatchID","1","1_gap_before", "{n}" as "0","X_gap_before" as "0_gap_before", "2", "2_gap_before", "Bookmaker","Timestamp" FROM {y}.' '"Matches{x}" ' \
                   'INNER JOIN {y}."Odds_{z}{x}" on "Odds_{z}{x}"."MatchID" = "Matches{x}"."MatchID" WHERE "Matches{x}"."MatchID" = {m}'.format(
                x=matches_suffix, y=schema, z=market, m=matchID, n=x_col)
        if market == 'ah':
            return 'SELECT "Matches{x}"."MatchID","1" as "0" , "1_gap_before" as "0_gap_before", "2" as "1", "2_gap_before" as "1_gap_before", "Bookmaker","Handicap","Timestamp" FROM {y}.' '"Matches{x}" ' \
                   'INNER JOIN {y}."Odds_{z}{x}" on "Odds_{z}{x}"."MatchID" = "Matches{x}"."MatchID" WHERE "Matches{x}"."MatchID" = {m}'.format(
                x=matches_suffix, y=schema, z=market, m=matchID)
        if market == 'bts':
            return 'SELECT "Matches{x}"."MatchID","YES" as "1", "YES_gap_before" as "1_gap_before", "NO" as "0", "NO_gap_before" as "0_gap_before", "Bookmaker","Timestamp" FROM {y}.' '"Matches{x}" ' \
                   'INNER JOIN {y}."Odds_{z}{x}" on "Odds_{z}{x}"."MatchID" = "Matches{x}"."MatchID" WHERE "Matches{x}"."MatchID" = {m}'.format(
                x=matches_suffix, y=schema, z=market, m=matchID)
        if market == 'ou':
            return 'SELECT "Matches{x}"."MatchID","Under" as "0", "Under_gap_before" as "0_gap_before", "Over" as "1", "Over_gap_before" as "1_gap_before", "Total", "Bookmaker","Details","Timestamp" FROM {y}.' '"Matches{x}" ' \
                   'INNER JOIN {y}."Odds_{z}{x}" on "Odds_{z}{x}"."MatchID" = "Matches{x}"."MatchID" WHERE "Matches{x}"."MatchID" = {m}'.format(
                x=matches_suffix, y=schema, z=market, m=matchID)
        if market == 'dc':
            return 'SELECT "Matches{x}"."MatchID","1X" as "0","1X_gap_before" as "0_gap_before","12" as "1","12_gap_before" as "1_gap_before","X2" as "2", "X2_gap_before" as "2_gap_before", "Bookmaker","Timestamp" FROM {y}.' '"Matches{x}" ' \
                   'INNER JOIN {y}."Odds_{z}{x}" on "Odds_{z}{x}"."MatchID" = "Matches{x}"."MatchID" WHERE "Matches{x}"."MatchID" = {m}'.format(
                x=matches_suffix, y=schema, z=market, m=matchID)
        if market == 'ha':
            return 'SELECT "Matches{x}"."MatchID","1","1_gap_before" , "2" as "0","2_gap_before" as "0_gap_before", "Bookmaker","Timestamp" FROM {y}.' '"Matches{x}" ' \
                   'INNER JOIN {y}."Odds_{z}{x}" on "Odds_{z}{x}"."MatchID" = "Matches{x}"."MatchID" WHERE "Matches{x}"."MatchID" = {m}'.format(
                x=matches_suffix, y=schema, z=market, m=matchID)
        # if market == '1x2':
        #     return 'SELECT "Matches"."MatchID","1", "X" as "0", "2", "Result","Bookmaker","Timestamp" FROM {y}.' '"Matches" ' \
        #            'INNER JOIN {y}."Odds_1x2" on "Odds_1x2"."MatchID" = "Matches"."MatchID" WHERE "Matches"."MatchID" = {z}'.format(
        #         y=schema, z=matchID)
        # if market == 'ah':
        #     return 'SELECT "Matches"."MatchID","1" as "0" , "2" as "1", "Result","Bookmaker","Handicap","Timestamp" FROM {y}.' '"Matches" ' \
        #            'INNER JOIN {y}."Odds_ah" on "Odds_ah"."MatchID" = "Matches"."MatchID" WHERE "Matches"."MatchID" = {z}'.format(
        #         y=schema, z=matchID)
        # if market == 'bts':
        #     return 'SELECT "Matches"."MatchID","YES" as "1", "NO" as "0", "Result","Bookmaker","Timestamp" FROM {y}.' '"Matches" ' \
        #            'INNER JOIN {y}."Odds_bts" on "Odds_bts"."MatchID" = "Matches"."MatchID" WHERE "Matches"."MatchID" = {z}'.format(
        #         y=schema, z=matchID)
        # if market == 'ou':
        #     return 'SELECT "Matches"."MatchID","Under" as "0", "Over" as "1","Total", "Result","Bookmaker","Details","Timestamp" FROM {y}.' '"Matches" ' \
        #            'INNER JOIN {y}."Odds_ou" on "Odds_ou"."MatchID" = "Matches"."MatchID" WHERE "Matches"."MatchID" = {z}'.format(
        #         y=schema, z=matchID)
        # if market == 'dc':
        #     return 'SELECT "Matches"."MatchID","1X" as "0","12" as "1","X2" as "2", "Result","Bookmaker","Timestamp" FROM {y}.' '"Matches" ' \
        #            'INNER JOIN {y}."Odds_dc" on "Odds_dc"."MatchID" = "Matches"."MatchID" WHERE "Matches"."MatchID" = {z}'.format(
        #         y=schema, z=matchID)
        # if market == 'ha':
        #     return 'SELECT "Matches"."MatchID","1" , "2" as "0", "Result","Bookmaker","Timestamp" FROM {y}.' '"Matches" ' \
        #            'INNER JOIN {y}."Odds_ha" on "Odds_ha"."MatchID" = "Matches"."MatchID" WHERE "Matches"."MatchID" = {z}'.format(
        #         y=schema, z=matchID)


def load_asian_odds(engine,MatchID,market,timestamp):
    if market == 'ou':
        odds = pd.read_sql(sql_ou.format(matchid=MatchID, timestamp=timestamp), engine)
        odds = odds.loc[odds['Over'].str.contains('^\d\.5$')].reset_index(drop=True)
        return odds
    if market == 'ah':
        odds = pd.read_sql(sql_ah.format(matchid=MatchID, timestamp=timestamp), engine)
        odds = hlp.parse_handicap(odds).reset_index(drop=True)
        return odds
    if market == '1x2':
        odds=pd.read_sql(sql_x.format(matchid=MatchID), engine)
        return odds

def load_data(conn, singleID=False, matchID=None, schema=None, market=None, to_numpy=False, csv=False, csv_file="",
              limit=True):
    """
    Loads and parses data
    :param schema:
    :param market:
    :param dtb_url:
    :param to_numpy:
    :param csv:
    :param csv_file:
    :return:
    """
    if not csv:
        select_script = get_select_script(market=market, schema=schema, singleID=singleID, matchID=matchID, limit=limit)
        try:
            df = pd.read_sql_query(select_script, con=conn)
        except sqlalchemy.exc.ProgrammingError as e:
            return None
    else:
        df = pd.read_csv(csv_file)
    if df.empty:
        return None
    # df = df.dropna()
    # df = df[df['Result'] != '-']
    # df = df[df['Result'] != "---"]
    # if market_choices[market] == 2:
    #     df = df[np.logical_and(df['1'].astype(float) >= 1, df['0'].astype(float) >= 1)]
    # else:
    #     df = df[
    #         np.logical_and.reduce([df['1'].astype(float) >= 1, df['0'].astype(float) >= 1, df['2'].astype(float) >= 1])]
    df = df.reset_index(drop=True)
    if df.empty:
        return None
    if market == 'ou':
        # select only whole number totals
        df['Total'] = pd.to_numeric(df["Total"], downcast="float")
        df = df[(df['Total'] % 1 != 0.0) & (df['Total'] % 0.5 == 0.0) & (df['Total'] < 10)]
        df = df.reset_index()
        if df.empty:
            return None
        # df['results'] = hlp.getResults(df['Result'], market, df['Total'], sport=schema)
        odds = divide_closed_opening(df, ['MatchID', 'Bookmaker', 'Total'], 'Total')
    elif market == 'ah':
        df = hlp.parse_handicap(df)
        if df.empty:
            return None
        df = df.reset_index(drop=True)
        # df['results'] = hlp.getResults(results=df['Result'], market=market, handicap=df['Handicap'], sport=schema,
        #                               from_string=not csv)
        odds = divide_closed_opening(df, ['MatchID', 'Bookmaker', 'Handicap'], 'Handicap')
    else:
        # df['results'] = hlp.getResults(df['Result'], market)
        odds = divide_closed_opening(df)
    # if df['results'] is None:
    #    return None, None, None, None

    # if to_numpy:
    #     if market_choices[market] == 3:
    #         closed = closed[['0', '1', '2']].to_numpy(dtype=np.float)
    #         opening = opening[['0', '1', '2']].to_numpy(dtype=np.float)
    #     if market_choices[market] == 2:
    #         closed = closed[['0', '1']].to_numpy(dtype=np.float)
    #         opening = opening[['0', '1']].to_numpy(dtype=np.float)

    # tohle predelat:
    # closed = closed.drop_duplicates(subset=closed.columns.difference(['index', '1', '0', '2']))
    # opening = opening.drop_duplicates(subset=opening.columns.difference(['index', '1', '0', '2']))
    # closed = closed.drop_duplicates(subset=['MatchID','Bookmaker'],keep='first')
    # opening = opening.drop_duplicates(subset=['MatchID','Bookmaker'])

    return odds


def divide_closed_opening(df, unique_cols=['MatchID', 'Bookmaker'], special_col=''):
    """
    Selects opening and closing odds from dataframe - legacy..will do it differently
    :param df:
    :return:
    """
    unique_cols = unique_cols + ['Timestamp']
    # kdyz je none u prvniho tak ten smazat
    input_df = df.sort_values(by=unique_cols, ascending=True)
    final_df = pd.DataFrame()
    sp_values = ['']
    if special_col != '':
        sp_values = df[special_col].drop_duplicates()
    for sp_val in sp_values:
        for bookie in list(df['Bookmaker'].drop_duplicates()):
            if special_col != '':
                df = input_df[(input_df['Bookmaker'] == bookie) & (input_df[special_col] == sp_val)]
            else:
                df = input_df[input_df['Bookmaker'] == bookie]
            # df = df[~(df['Bookmaker'].isin( #tohle mi ted vsechno smaze?? - asi dat pryc nejak - uz neni potreba
            #     list(df[(df['Bookmaker'] != df['Bookmaker'].shift(1)) & (df.isnull().any(axis=1))]['Bookmaker'])))]
            # df = df.fillna(method='ffill') #nemuzu pouzit ffill, nevim jake odds byly misto None
            # df = df.fillna(0)
            df = df.drop_duplicates()
            # df = df.dropna().reset_index()
            gap_cols = [col for col in df.columns if 'gap_before' in col]
            not_gap_cols = [col for col in df.columns if 'gap_before' not in col]
            # ffill pro asian odds atd musi byt pres markety
            # tyhle dva radky pak dat pryc, je to pro closing odds - nebo mozna spis ne
            df.loc[:, not_gap_cols] = df.loc[:, not_gap_cols].ffill()  # do ffill
            # df = df.fillna(method='ffill') # do ffill
            df.loc[:, 'NextTimestamp'] = df['Timestamp'].shift(-1)
            # df[]
            # tr =
            df.loc[df[gap_cols].any(axis=1).shift(-1, fill_value=True), 'NextTimestamp'] = df['Timestamp']
            df = df[~df[gap_cols].any(axis=1)]  # remove entries with gaps
            # df = df.groupby(unique_cols).last().reset_index() #closing
            df = df.drop(columns=gap_cols)
            df = df.dropna()
            final_df = final_df.append(df)
    return final_df


def upsert_table(df, table_name, db, schema, update_on_conflict=False, update_cols=None, unique_cols=None):
    if df.empty:
        return

    try:
        df.to_sql(table_name, db, index=False, schema=schema)  # try to create nonexistent table and set it up first
        if not unique_cols:
            unique_cols = '","'.join(list(df.columns))

        db.execute(
            'ALTER TABLE "' + schema + '"."' + table_name + '" ADD CONSTRAINT '
            + table_name.lower() + '_unique_rows UNIQUE("' + unique_cols + '");')
        return
    except Exception as e:
        # print(e)
        pass  # table already exists, just "upsert" it

    database = pd.io.sql.pandasSQL_builder(db, schema=schema)

    sql_table = pd.io.sql.SQLTable(table_name, database, frame=df, index=False,
                                   if_exists='fail', prefix='pandas', index_label=None,
                                   schema=None, keys=None, dtype=None)

    keys, data_list = sql_table.insert_data()
    data = [{k: v for k, v in zip(keys, row)} for row in zip(*data_list)]

    stmt = insert(sql_table.table, values=data)
    if update_on_conflict:
        excluded = dict(stmt.excluded)
        to_be_updated = {col: excluded[col] for col in update_cols}
        stmt = stmt.on_conflict_do_update(constraint=f'{table_name.lower()}_unique_rows', set_=to_be_updated)
    else:
        stmt = stmt.on_conflict_do_nothing()

    try:
        db.execute(stmt)
    except sqlalchemy.exc.ProgrammingError as e:
        print("SQL upsert failed: " + str(e), file=sys.stderr)
        pass
    except Exception as e:
        print("SQL upsert failed: " + str(e), file=sys.stderr)
        pass
