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

def change_date(AO_date:str):
    parsed = AO_date.split('-')
    year = int(parsed[0])
    month = int(parsed[1])
    day = int(parsed[2].split(' ')[0])
    return '{d}.{m}. {y}'.format(d=day,m=month,y=year)

def find_best(ao_name, names):
    min = 100
    best_match = ""
    for n in names:
        cur = textdistance.lcsseq(ao_name,n)
        cur = len(ao_name) -len(cur)
        if cur < min:
            min = cur
            best_match = n
    print(best_match,min)

def translate(ao_name_home='CA Banfield',ao_name_away='CA Banfield',translate_table=None):
    if translate_table is None:
        translate_table =pd.read_csv(csv_file_with_mapping)
    #l=pd.read_sql(sql=sql_matches.format(h=ao_name_home,a=ao_name_away), con=engine)
    translate_home = translate_table['BE'].loc[translate_table['AO'] == ao_name_home].to_numpy()
    if translate_home.size > 0:
        translate_home = translate_home[-1]
    translate_away = translate_table['BE'].loc[translate_table['AO'] == ao_name_away].to_numpy()
    if translate_away.size > 0:
        translate_away = translate_away[-1]
    return translate_home,translate_away

def find_match(ao_home,ao_away,engine=None,date=None,translate_table=None,league=None):
    if translate_table is None:
        translate_table =pd.read_csv(csv_file_with_mapping)
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
            find_best(ao_home,home)
            print('\n\n\n')
            #print(home)
