from utils import helper_functions as hlp
import numpy as np
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder
import arbitrage
import pandas as pd
from utils import utils as ut

csv_file_with_mapping = "bet_AsianOdds_AO2BE.csv"
ip = "147.32.83.171"
sql_ou = """
SELECT DISTINCT \"Home\",\"Away\" FROM asianodds.\"Matches\" where \"Home\"=\'{y}\'
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


def translate(ao_name='CA Banfield',engine=None,translate_table=None):
    if translate_table is None:
        translate_table =pd.read_csv(csv_file_with_mapping)
    l=pd.read_sql(sql=sql_ou.format(y=ao_name),con=engine)
    ar=l['Home'].to_numpy()
    print(translate_table['BE'].loc[translate_table['AO'] == ar[0]].values)

translate(engine=engine)