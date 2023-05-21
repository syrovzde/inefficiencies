from scipy import stats
import numpy as np

def poisson(lambda_h,lambda_a,points=10):
    away_prop=stats.poisson.pmf([i for i in range(points)],lambda_a)
    home_prop= stats.poisson.pmf([i for i in range(points)],lambda_h)
    possible_results = np.array([away_prop[away]*home_prop[home] for away in range(points) for home in range(points)], dtype=float)
    return possible_results


def ou_poisson(lambda_h,lambda_a,over_threshold,points):
    total_scored = np.array([j+i for i in range(points) for j in range(points)])
    possible_results = np.array(poisson(lambda_h=lambda_h,lambda_a=lambda_a,points=points))
    prob_over = np.sum(possible_results[total_scored > over_threshold])
    prob_under = 1 - prob_over
    return prob_over,prob_under

def ah_poisson(lambda_h,lambda_a,handicap_threshold,points):
    difference = np.array([home-away for away in range(points) for home in range(points)])
    probs = np.array(poisson(lambda_h=lambda_h,lambda_a=lambda_a,points=points))
    handicap_home = np.sum(probs[difference+handicap_threshold > 0])
    handicap_away = 1 - handicap_home
    return handicap_home,handicap_away
def lambda_to_prob(lambda_h, lambda_a):
    p_h = stats.skellam.cdf(-1, lambda_a, lambda_h, loc=0)
    p_d = stats.skellam.pmf(0, lambda_a, lambda_h)
    p_a = 1 - p_h - p_d
    return p_h,p_d,p_a

def sample(lambda_h,lambda_a,count):
    #ph,pd,pa= lambda_to_prob(lambda_h=lambda_h,lambda_a=lambda_a)
    #r=np.random.random(size=count)
    #draw_threshold = pd + ph
    #results = np.zeros(shape=count,dtype=int)
    #results[r<=ph] = 1
    #results[r>draw_threshold] = 2
    home=np.random.poisson(lam=lambda_h,size=count)
    away=np.random.poisson(lam=lambda_a,size=count)
    overall = [(home[i],away[i]) for i in range(count)]
    return overall

def load_txt(k=10, file="ids.txt"):
    with open(file,'r') as f:
        profits = f.readline()
        match_ids=f.readline()
    match_ids=match_ids.split(',')
    match_ids=[match.replace("\'","").replace(']','').replace('[','').strip() for match in match_ids]
    profits = profits.split(',')
    profits = [float(profit.replace("\'", "").replace(']', '').replace('[', '').strip()) for profit in profits]
    #match_ids=np.array(match_ids)
    #match_ids = np.array_split(match_ids,k)
    return profits,match_ids

def remove_margin(odds):
    inv_odds = [1 / float(odd) for odd in odds]  # Calculate the inverse of each odd
    implied_probs = [inv_odd / sum(inv_odds) for inv_odd in
                     inv_odds]  # Calculate the implied probability for each outcome
    return implied_probs

#print(remove_margin([1.8,3.3,6]))
if __name__ == '__main__':
    load_txt()