import numpy as np
import pandas
import pandas as pd


# import analysis.market_efficiency.Bettor as Bettor
# import analysis.market_efficiency.method as method
# from analysis.market_efficiency.KL import KL
# from analysis.market_efficiency.generating_method import GeneratingMethod
# from analysis.market_efficiency.regression import Regression
# import analysis.market_efficiency.utils.odds2probs


def digit_sum(n):
    sum = 0
    for i in n:
        sum += int(i)
    return sum


def market_tables(market, points=11, totals=None, handicap=None):
    if market == '1x2':
        return x_market_table(points)
    elif market == 'bts':
        return bts_market_table(points)
    elif market == 'ou':
        return ou_result_table(totals=totals, points=11)
    elif market == 'ah':
        return ah_result_table(handicap, points)
    elif market == 'dc':
        return dc_result_table(points)
    # elif market == 'ha': TO DO: maybe add this, might not be possible
    # ah_result_table()
    else:
        raise Exception("Invalid market selected")


def x_market_table(points=11):
    table = np.ones((points, points), dtype=np.float)
    draw = np.eye(points, dtype=np.float).reshape(points ** 2)[np.newaxis, :].T
    home = np.triu(table, k=1).reshape(points ** 2)[np.newaxis, :].T
    away = np.tril(table, k=-1).reshape(points ** 2)[np.newaxis, :].T
    return [draw, home, away]


def bts_market_table(points=11):
    scored = np.ones((points, points), dtype=np.float)
    pom = np.arange(points)
    scored[pom, 0] = 0
    scored[0, pom] = 0
    scored = scored.reshape((points ** 2))[np.newaxis, :].T
    not_scored = np.array(scored == 0, dtype=np.float)
    return [not_scored, scored]


def ou_result_table(totals, points=11):
    count = len(totals)
    cums_array = np.ones((points, points))
    table = np.ones((count, points, points), dtype=np.bool)
    equal_table = np.zeros((count, points, points), dtype=np.bool)
    over = np.zeros((count, points ** 2), dtype=np.float)
    under = np.zeros((count, points ** 2), dtype=np.float)

    for i in range(0, points):
        cums_array[i, :] = np.arange(i, points + i)
    for i in range(count):
        table[i] = cums_array > totals[i]
        equal_table[i] = cums_array == totals[i]
    for i in range(count):
        all_over = np.zeros((points, points), dtype=np.float)
        all_under = np.ones((points, points), dtype=np.float)
        all_over[table[i]] = 1
        all_over[equal_table[i]] = -0.1
        all_under[table[i]] = 0
        all_under[equal_table[i]] = -0.1
        over[i] = all_over.flatten()
        under[i] = all_under.flatten()
    return over.T, under.T




def ah_result_table(handicap, points=11):
    count = np.size(handicap)
    cums_array = np.ones((points, points))
    table_one = np.zeros((count, points * points))
    table_two = np.zeros((count, points * points))
    table = np.zeros((count, points, points))
    for i in range(0, points):
        cums_array[i, :] = np.arange(-i, points - i)
    for i in range(count):
        table[i, :, :] = cums_array + handicap[i]
    for i in range(count):
        all_over = table[i, :] > 0
        all_under = table[i, :] < 0
        table_one[i] = all_over.flatten()  #home win
        table_two[i] = all_under.flatten() #away win
    return table_one.T, table_two.T


def dc_result_table(points):
    table = np.ones((points, points), dtype=np.float)
    homeaway = (table - np.eye(points, dtype=np.float)).reshape(points ** 2)[np.newaxis, :].T
    homedraw = np.triu(table, k=0).reshape(points ** 2)[np.newaxis, :].T
    awaydraw = np.tril(table, k=0).reshape(points ** 2)[np.newaxis, :].T
    return [homedraw, homeaway, awaydraw]


def parse_handicap(df: pandas.DataFrame):
    df['Handicap'] = pd.to_numeric(df['Handicap'], errors='coerce')
    df = df.dropna(subset=['Handicap'])
    df = df[(df['Handicap'] % 1 == 0.5) & (df['Handicap'] < 10) & (df['Handicap'] > - 10)]
    #print(df)
    df['Handicap'] = df['Handicap'].apply(lambda x: x * -1)
    """duplicate_list = []
    checked = []
    #swapping handicap
    for i in np.abs(df['Handicap']):
        if i in checked:
            duplicate_list.append(i)
        else:
            checked.append(i)
    for i in duplicate_list:
        swp = df[df['Handicap'] == i]['0'].values[0]
        df.loc[df['Handicap'] == i,'0'] =  df[df['Handicap'] == -i]['0'].values
        df.loc[df['Handicap'] == -i,'0'] = swp
    #case where handicaps are solo f.e -0.5 is in data but 0.5 not
    for i in df['Handicap']:
        if not i in duplicate_list and not abs(i) in duplicate_list:
            save = df[df['Handicap'] == i]['0'].values[0]
            df.loc[df['Handicap'] == i,'0'] = 1
            nw = pd.Series({'Handicap':-i,'0':save,'1':1})
            df = pd.concat([df,nw.to_frame().T],ignore_index=True)"""
    return df.copy()


def asian_handicap_results(results, handicap, from_string=True):
    pom = results.str.split("-", expand=True)
    ASC = np.array(pom[1], dtype=float)
    HSC = np.array(pom[0], dtype=float)
    difference = HSC - ASC
    shifted_difference = difference + handicap
    shifted_difference[shifted_difference >= 0.5] = 1
    shifted_difference[shifted_difference <= -0.5] = -1  #zmenit to tady
    return pandas.Series(shifted_difference)


def bts_results(results):
    pom = results.str.split("-", expand=True)
    ASC = np.array(pom[1], dtype=float)
    HSC = np.array(pom[0], dtype=float)
    bet_result = np.logical_and(ASC != 0, HSC != 0)
    return bet_result.astype(int)


def tennis_ou_results(results, total):
    sets = results.str.replace("\D+", "", regex=True)
    totals = []
    pom = results.str.findall("[^^].\d")
    for index, value in sets.items():
        pom = digit_sum(value)
        totals.append(pom)
    return pandas.Series(totals)


def result_sum(results):
    pom = results.str.split("-", expand=True)
    ASC = pom[1]
    HSC = pom[0]
    match_total = ASC.astype(int) + HSC.astype(int)
    return match_total


def ou_results(results, total, sport):
    try:
        total = total.str.findall("\d{1,3}\.*\d*").str[0]
    except AttributeError:
        pass
    if sport == 'tennis':
        return tennis_ou_results(results, total)
    pom = results.str.split("-", expand=True)
    ASC = pom[1]
    HSC = pom[0]
    match_total = ASC.astype(int) + HSC.astype(int)
    total = total.astype(str).astype(float)
    return (match_total > total).astype(int)


def getResults(results, market, total=None, sport='football', handicap=None, from_string=True, details=None):
    if market == "1x2":
        return x12market_results(results)
    if market == "ou":
        if sport == 'volleyball':
            return None
        return ou_results(results, total, sport)
    if market == "bts":
        return bts_results(results)
    if market == 'ah':
        if sport == 'volleyball':
            return None
        return asian_handicap_results(results, handicap)
    if market == 'ha':
        return asian_handicap_results(results, handicap=None)


def x12market_results(results, details=None, sport=None):
    pom = results.str.split("-", expand=True)
    ASC = np.array(pom[1], dtype=float)
    HSC = np.array(pom[0], dtype=float)
    winner = HSC != ASC
    home = HSC < ASC
    results = home.astype(int) + winner.astype(int)
    return results



def find_correct_odds(fair, margin, generating):
    if fair:
        odds = generating.find_fair_odds(method=margin)
    else:
        odds = generating.find_odds()
    return odds


