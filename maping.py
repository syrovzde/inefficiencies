from utils import helper_functions as hlp
import numpy as np
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder
import arbitrage
import pandas as pd
from utils import utils as ut

csv_file_with_mapping = "bet_AsianOdds_AO2BE.csv"
ip = "147.32.83.171"
sql_matches = """
SELECT DISTINCT \"Home\",\"Away\" FROM asianodds.\"Matches\" where \"Home\"=\'{h}\' and \"Away\"=\'{a}\'
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


def translate(ao_name_home='CA Banfield',ao_name_away='CA Banfield',engine=None,translate_table=None):
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
