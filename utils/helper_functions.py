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
    count = totals.size
    cums_array = np.ones((points, points))
    table = np.ones((count, points, points), dtype=np.float)
    over = np.zeros((count, points ** 2), dtype=np.float)
    under = np.zeros((count, points ** 2), dtype=np.float)
    for i in range(0, points):
        cums_array[i, :] = np.arange(i, points + i)
    for i in range(count):
        table[i, :, :] = cums_array > totals[i]
    for i in range(count):
        all_over = table[i, :] == 1
        all_under = table[i, :] == 0
        over[i] = all_over.flatten()
        under[i] = all_under.flatten()
    return under.T, over.T



def ah_result_table(handicap, points=11):
    count = np.size(handicap)
    cums_array = np.ones((points, points))
    table_one = np.zeros((count, points * points), dtype=np.float)
    table_two = np.zeros((count, points * points), dtype=np.float)
    table = np.zeros((count, points, points))
    for i in range(0, points):
        cums_array[i, :] = np.arange(-i, points - i)
    for i in range(count):
        table[i, :, :] = cums_array + handicap[i]
    for i in range(count):
        all_over = table[i, :] > 0
        all_under = table[i, :] < 0
        table_one[i] = all_over.flatten()  # home win?
        table_two[i] = all_under.flatten()
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
    # df = df.copy()
    # df['Handicap_2'] = df['Handicap'].str.split(', ').str[-1]
    # df['Handicap'] = df['Handicap'].str.split(', ').str[0]
    # df['Handicap_2'] = pd.to_numeric(df["Handicap_2"], downcast="float")
    # df['Handicap'] = pd.to_numeric(df["Handicap"], downcast="float")
    # df = df[(df['Handicap'] % 1 == 0.5) | (df['Handicap_2'] % 1 == 0.5)].copy()
    # df.loc[df['Handicap_2'] % 1 == 0.5, 'Handicap'] = df.loc[df['Handicap_2'] % 1 == 0.5, 'Handicap_2']
    # df = df.drop(['Handicap_2'], axis=1)
    # df = df[(df['Handicap'] < 10 ) & (df['Handicap'] > - 10)] #limit handicap
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


#
# def random_dict(dict):
#     return random(dict['data'], dict['probabilities'], dict['fair'], dict['margin'], dict['odds'], dict['bet_choices'],
#                   dict['n_random'], dict['asian'],dict['results'])
#
#
# def random(data, probabilities=None, fair=True, margin='basic', odds=None, bet_choices=3, n_random=100, asian=False,results=None):
#     generating = GeneratingMethod(data=data, bet_choices=bet_choices)
#     if results is not None:
#         generating.results = results
#     if odds is None:
#         odds = find_correct_odds(fair, margin, generating)
#     if probabilities is None:
#         probabilities = np.ones(odds.shape)/bet_choices
#     generating.n_random = n_random
#     generating.length = results.size
#     bets = generating.run(probabilities)
#     return generating.evaluate(bets=bets, odds=odds, asian=asian)
#
#
# def bet_home_dict(dict):
#     return randomSingle(dict['data'], method.HOMEWIN, dict['fair'], dict['margin'], dict['odds'], dict['bet_choices'],
#                         dict['asian'],dict['results'])
#
#
# def bet_away_dict(dict):
#     if dict['bet_choices'] == 2:
#         return None
#     return randomSingle(dict['data'], method.AWAYWIN, dict['fair'], dict['margin'], dict['odds'], dict['bet_choices'],
#                         dict['asian'],dict['results'])
#
#
# def bet_draw_dict(dict):
#     return randomSingle(dict['data'], method.DRAW, dict['fair'], dict['margin'], dict['odds'], dict['bet_choices'],
#                         dict['asian'],dict['results'])

#
# def random_single_dict(dict):
#     return randomSingle(dict['data'], dict['number'], dict['fair'], dict['margin'], dict['odds'], dict['bet_choices'],
#                         dict['asian'],dict['results'])
#
#
# def randomSingle(data, number, fair=True, margin='basic', odds=None, bet_choices=3, asian=False,results = None):
#     generating = GeneratingMethod(data=data, bet_choices=bet_choices)
#     if results is not None:
#         generating.results = results
#     if odds is None:
#         odds = find_correct_odds(fair, margin, generating)
#     bets = number * np.ones(odds.shape[0], dtype=np.int)
#     return generating.evaluate(bets, odds=odds, asian=asian)


def find_correct_odds(fair, margin, generating):
    if fair:
        odds = generating.find_fair_odds(method=margin)
    else:
        odds = generating.find_odds()
    return odds


#
# def bet_favourite_dict(dict):
#     return betFavourite(dict['data'], dict['fair'], dict['margin'], dict['odds'], dict['bet_choices'], dict['asian'],dict['results'])
#
#
# def betFavourite(data, fair=True, margin="basic", odds=None, bet_choices=3, asian=False,results = None):
#     generating = method.Method(data, bet_choices=bet_choices)
#     if results is not None:
#         generating.results = results
#     if odds is None:
#         odds = find_correct_odds(fair, margin, generating)
#     favourite = np.argmin(odds, axis=1)
#     return generating.evaluate(bets=favourite, odds=odds, asian=asian)
#
#
# def bet_Underdog_dict(dict):
#     return betUnderdog(dict['data'], dict['fair'], dict['margin'], dict['odds'], dict['bet_choices'], dict['asian'],dict['results'])
#
#
# def betUnderdog(data, fair=True, margin="basic", odds=None, bet_choices=3, asian=False,results = None):
#     generating = method.Method(data, bet_choices=bet_choices)
#     if results is not None:
#         generating.results = results
#     if odds is None:
#         odds = find_correct_odds(fair, margin, generating)
#     underdog = np.argmax(odds, axis=1)
#     return generating.evaluate(bets=underdog, odds=odds, asian=asian)

#
# def bettorWithBankroll(data, fraction, bankroll, iterations, threshold, odds):
#     bettor = Bettor.Bettor(data=data, fraction=fraction, bankroll=bankroll, iterations=iterations, threshold=threshold)
#     iteration = bettor.bet(odds)
#     return bettor.bankroll, iteration

#
# def bettors_with_bankroll_dict(dict):
#     return bettorsWithBankroll(dict['data'], dict['count'], dict['fraction'], dict['bankroll'], dict['iterations'],
#                               dict['threshold'], dict['odds'])
#
# def bettorsWithBankroll(data, count, fraction, bankroll, iterations, threshold, odds):
#     bankrolls = np.zeros(count)
#     iteration = np.zeros(count)
#     for i in range(count):
#         bankrolls[i], iteration[i] = bettorWithBankroll(data, fraction=fraction, bankroll=bankroll,
#                                                         iterations=iterations, threshold=threshold, odds=odds)
#     return bankrolls, iteration

#
# def devide_bet_dict(dict):
#     return devide_bet(dict['data'], dict['margin'], dict['prob'], dict['bet_choices'])

#
# def devide_bet(data=None, margin="basic", prob=None, bet_choices=3,results = None,odds = None):
#     results = pandas.Series.to_numpy(results, dtype=np.int)
#     generating = method.Method(data, bet_choices=bet_choices)
#     if odds is None:
#         odds = generating.find_fair_odds(margin)
#     if prob is None:
#         prob = 1 / bet_choices * np.ones(
#             odds.shape)  # prob = 1 / 3 * np.ones(odds.shape)  # prob = generating.get_probabilities()  # prob = 1/odds
#     if results is None:
#         n_bet = data['results'].count()
#     else:
#         n_bet = np.size(results)
#     eval = np.zeros(prob.shape)
#     for i in range(n_bet):
#         eval[i, results[i]] = odds[i, results[i]] * prob[i, results[i]]
#     return eval


# def kl_divergence_dict(dict):
#     return kl_divergence(dict['data'], dict['margin'])


# def kl_divergence(data, margin='basic', bet_choices=3,odds = None,results = None):
#     kl = KL(data=data, bet_choices=bet_choices)
#     return kl.run(margin=margin,odds=odds,results=results)
#

# def regression_dict(dict):
#     return regression(dict['data'], dict['odds'])


# def regression(data, bet_choices=3, fair=True, margin='basic'):
#     linear = Regression(data=data, bet_choices=bet_choices)
#     odds = find_correct_odds(fair=fair, margin=margin, generating=linear)
#     lin = linear.run(model="Linear", odds=odds)
#     log = linear.run(model="Logistic", odds=odds)
#     return lin, log


if __name__ == '__main__':
    # results = ['1-1', '1-3', '2-2', '3-2']
    results = ['0-0', '1-0', '0-1', '1-1', '2-1', '1-2', '2-2']
    totals = ['0', '0, 0.5', '-1', '0.5', '1']
    # over = [2,3,5,1.5,1.2]
    # under = [1.8,1.5,1.2,2.3,8]
    # print(len(over))
    # over = pd.Series(over)
    # under = pd.Series(under)
    # results = pd.Series(results)
    # print(getResults(results=results,market='1x2'))
    # totals = pd.Series(totals)
    # ah_result_table(over,under,totals)
    # a =
    # totals = np.array(['0.5','1.5'])
    # totals = pd.Series(totals).astype(float)
    # over, under = ou_result_table(totals=totals, points=10)
    # print(over)
    # print(under.shape)
    results = pd.Series(results)
    # print(asian_handicap_results(results, None, from_string=True))
    print(ah_result_table(pd.Series([1.5]), 10)[0])
