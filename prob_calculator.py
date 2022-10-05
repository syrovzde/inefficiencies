import psycopg2 as pg
from sshtunnel import SSHTunnelForwarder
import numpy as np

test_sql = "SELECT \"Matches\".\"Result\" FROM football.\"Matches\""


def calculate(file_name='probability.txt', ip="147.32.83.171",conn =None):
    if conn is None:
        ssh_tunnel = SSHTunnelForwarder(
            ip,
            ssh_username='syrovzde',
            ssh_private_key='C:\\Users\\zdesi\\.ssh\\syrovzde_rsa',
            remote_bind_address=('localhost', 5432)
        )
        ssh_tunnel.start()

        conn = pg.connect(
            host='localhost',
            port=ssh_tunnel.local_bind_port,
            user='syrovzde',
            database='betexplorer')

    prior_prob = np.zeros((11, 11), dtype=np.uint)
    with conn.cursor() as curs:
        curs.execute(test_sql)
        for c in curs:
            if c[0] != '':
                try:
                    home, away = c[0].split('-')
                except:
                    continue
                if home != '' and away != '':
                    home = int(home)
                    away = int(away)
                    home = 10 if home > 10 else home
                    away = 10 if away > 10 else away
                    prior_prob[home, away] += 1
    np.savetxt(file_name, prior_prob, fmt='%d')
