from utils import helper_functions as hlp
import numpy as np
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder
import arbitrage
import pandas as pd
from utils import utils as ut
import textdistance

sql_results = """
SELECT DISTINCT \"Result\" FROM football.\"Matches\" WHERE \"Home\" = \'{h}\' and \"Away\" = \'{a}\' and \"Time\" = \'{t}\'
"""

csv_file_with_mapping = "bet_AsianOdds_AO2BE.csv"
ip = "147.32.83.171"
sql_matches = """
SELECT DISTINCT \"Home\",\"Away\" FROM asianodds.\"Matches\" where \"Home\"=\'{h}\' and \"Away\"=\'{a}\'
"""


sql_matches_by_date  = """
SELECT DISTINCT \"Home\",\"Away\" FROM football.\"Matches\" where \"Time\" = \'{t}\'
"""

def change_date(AO_date:str):
    parsed = AO_date.split('-')
    year = int(parsed[0])
    month = int(parsed[1])
    day = int(parsed[2].split(' ')[0])
    return '{d}.{m}. {y}'.format(d=day,m=month,y=year)

def find_best(ao_name, names):
    min = 100
    best_match = ""
    distances = []
    for n in names:
        cur = textdistance.levenshtein(ao_name,n)
        distances.append(cur)
    distances = np.array(distances)
    sor = np.argsort(distances)
    return names[sor[:3]].to_numpy()

def translate(ao_name_home='CA Banfield',ao_name_away='CA Banfield',translate_table=None):
    if translate_table is None:
        translate_table =pd.read_csv(csv_file_with_mapping)
    translate_home = translate_table['BE'].loc[translate_table['AO'] == ao_name_home].to_numpy()
    if translate_home.size > 0:
        translate_home = translate_home[-1]
    translate_away = translate_table['BE'].loc[translate_table['AO'] == ao_name_away].to_numpy()
    if translate_away.size > 0:
        translate_away = translate_away[-1]
    return translate_home,translate_away

def find_match(ao_home,ao_away,engine=None,date=None,translate_table=None,league=None):
    if translate_table is None:
        translate_table = pd.read_csv(csv_file_with_mapping)
    translated_home = translate_table['BE'].loc[translate_table['AO'] == ao_home].to_numpy()
    if translated_home.size > 0:
        translated_home = translated_home[-1]
    translated_away = translate_table['BE'].loc[translate_table['AO'] == ao_away].to_numpy()
    if translated_away.size > 0:
        translated_away = translated_away[-1]
    translated_time = change_date(AO_date=str(date))
    res =pd.read_sql(sql_results.format(h=translated_home, a=translated_away, t=translated_time), con=engine)
    if res.empty:
            kl=pd.read_sql(sql_matches_by_date.format(t=translated_time),con=engine)
            home = kl['Home']
            away = kl['Away']
            print(ao_home)
            pos=find_best(ao_home,home)
            print(pos)
            print("choose possibilities")
            print("one is {k}".format(k=pos[0]))
            print("two is {k}".format(k=pos[1]))
            print("three is {k}".format(k=pos[2]))
            i = input()
            choice = None
            if i == "1":
                choice = {'AO':ao_home,'BE':pos[0]}
            elif i == "2":
                print("two was chosen")
                choice = {'AO':ao_home,'BE':pos[1]}
            elif i == "3":
                print("three was chosen")
                choice = {'AO':ao_home,'BE':pos[2]}
            else:
                print("none")
            if choice is not None:
                translate_table=translate_table.append(pd.Series(choice),ignore_index=True)
                print(translate_table)
                translate_table.to_csv("bet_AsianOdds_AO2BE.csv")
            return 0,translate_table
    return 1,translate_table
            #print(home)

if __name__ == '__main__':
    df = pd.read_csv("bet_AsianOdds_AO2BE.csv",index_col=0)
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    print(df)
    df.to_csv("bet_AsianOdds_AO2BE.csv")