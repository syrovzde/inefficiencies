
sql_all_matches = """
SELECT \"MatchID\",\"Time\",\"Home\",\"Away\",\"League\" FROM asianodds.\"Matches\"
"""

sql_all_matches_limited = """
SELECT \"MatchID\",\"Time\",\"Home\",\"Away\",\"League\" FROM asianodds.\"Matches\"
"""

sql_one_match = """
SELECT \"MatchID\",\"Time\",\"Home\",\"Away\",\"League\"FROM asianodds.\"Matches\" WHERE \"MatchID\" = \'{id}\'
"""

sql_results = """
SELECT DISTINCT \"Result\" FROM football.\"Matches\" WHERE \"Home\" = \'{h}\' and \"Away\" = \'{a}\' and \"Time\" = \'{t}\'
"""

sql_results_id = """
SELECT \"Result\", \"Home\", \"Away\",  \"Time\", \"League\", \"Season\"  FROM football.\"Matches\" WHERE \"MatchID\" =  \'{matchid}\'
"""

sql_all_results = """
SELECT \"MatchID\", \"Result\", \"Home\", \"Away\",  \"Time\", \"League\", \"Season\"  FROM football.\"Matches\" WHERE \"Season\" = \'2022\' or \"Season\" = \'2023\'
"""
