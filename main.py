import psycopg2 as pg
from sshtunnel import SSHTunnelForwarder

ip = "147.32.83.171"
sql_ids = """ SELECT "MatchID" FROM asianodds.\"Matches\" LIMIT 1
"""

sql_x = """ SELECT \"1\",\"2\",\"x\",\"Timestamp\" FROM asianodds.\"Odds_1x2_FT\" WHERE \"Odds_1x2_FT\".\"MatchID\" = \'{l}\'
"""

sql_ou = """SELECT 
\"Over\",\"1\",\"2\" FROM asianodds.\"Odds_ou_FT\" WHERE \"Odds_ou_FT\".\"MatchID\" = \'{matchid}\' AND
\"Odds_ou_FT\".\"Timestamp\" = \'{timestamp}\'
"""
sql_ah = """ SELECT 
\"Handicap\","1",\"2\"FROM asianodds.\"Odds_ah_FT\" WHERE \"Odds_ah_FT\".\"MatchID\" = \'{matchid}\' AND
\"Odds_ah_FT\".\"Timestamp\" = \'{timestamp}\'
"""
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
    database='asianodds')
cur_id = conn.cursor()
cur_x = conn.cursor()
cur_ou = conn.cursor()
cur_ah = conn.cursor()
cur_id.execute(sql_ids)
for c in cur_id:
    id = c[0]
    print(id)
    cur_x.execute(sql_x.format(l=id))
    for d in cur_x:
        print(d)
        timestamp = d[3]
        cur_ah.execute(sql_ah.format(matchid=id, timestamp=timestamp))
        cur_ou.execute(sql_ou.format(matchid=id, timestamp=timestamp))
        for e in cur_ah:
            print(e)
        print("over-under")
        for e in cur_ou:
            print(e)
