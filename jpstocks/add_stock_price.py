# -*- coding: utf-8 -*-
#%%
# -*- coding: utf-8 -*-
import csv
import glob
import datetime
import os
import sqlite3
import pandas as pd
import datetime
from all_stocks import csv_to_db

def make_daily_csv(code_range, save_dir):
	for code in code_range:
		try:
			dfs = pd.read_html('https://m.finance.yahoo.co.jp/stock/historicaldata?code={}.T'.format(code))
			df = dfs[0]
			if '日付' in df.columns:
				df.to_csv('{}/{}.T.csv'.format(save_dir,code), index=False, encoding='cp932', line_terminator='\n')
		except Exception as e:
			pass

#%%
if __name__ == '__main__':
	import os
	os.chdir(os.path.dirname(os.path.abspath(__file__)))
	# scraping_daily_price((range(7203,7204)), os.getcwd())
	# dfs = pd.read_html(f'https://m.finance.yahoo.co.jp/stock/historicaldata?code=3558.T')
	today = format(datetime.date.today(), '%Y%m%d')
	daily_csv_dir = './daily/{}'.format(today)
	os.makedirs(daily_csv_dir, exist_ok=True)
	make_daily_csv(range(1300,10000),daily_csv_dir)
	csv_to_db.all_csv_file_to_db('sqllite/jpstocks.db',daily_csv_dir)
