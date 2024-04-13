import pandas as pd
from bambooai import *

df = pd.read_csv('test_activity_data.csv')
bamboo = BambooAI(df, debug=False, vector_db=True, search_tool=True)
bamboo.pd_agent_converse()