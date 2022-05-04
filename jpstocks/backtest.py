#%%
import numpy as np
import pandas as pd
import datetime
import os
import sqlite3
import MetaTrader5 as mt5
import talib as ta


#%%
url = 'https://site1.sbisec.co.jp/ETGate/?OutSide=on&_ControlID=WPLETmgR001Control&_PageID=WPLETmgR001Mdtl20&_DataStoreID=DSWPLETmgR001Control&_ActionID=DefaultAID&getFlg=on&burl=search_market&cat1=market&cat2=none&dir=info&file=market_meigara_225.html'
# テーブルリストを取得
table_list = pd.read_html(url)
# 2番目が該当するテーブルなのでデータフレーム化
df_nk225_sbi = table_list[1]
# 最初の2カラムのみ残す
df_nk225_sbi = df_nk225_sbi.iloc[:, [0, 1]]
# カラム名を指定
df_nk225_sbi.columns = ['code', 'name']
df_nk225_sbi

# %%
code_result = [str(row['code']) for i,row in df_nk225_sbi.iterrows()]
print(code_result)
code_ids_string = ','.join(code_result)

#%%
db_file_name = './sqllite/jpstocks.db'
conn = sqlite3.connect(db_file_name)
sql = 'SELECT * from raw_prices where code = {} order by date'
df_list = []

with conn:
	for i,row in df_nk225_sbi.iterrows():
		#rows = conn.executemany(sql, str(row['code']))
		df=pd.read_sql_query(sql.format(row['code']), conn)
		df['SMA10'] = ta.SMA(df['close'], timeperiod=10)
		df_list.append(df)

n225_df = pd.concat(df_list)
print(n225_df)



#%%
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import MetaTrader5 as mt5
 
# MetaTrader 5に接続する
if not mt5.initialize():
   print("initialize() failed")
   mt5.shutdown()
 
# 接続状態とパラメータをリクエストする
print(mt5.terminal_info())
# MetaTrader 5バージョンについてのデータを取得する
print(mt5.version())
 
# 数々の方法で異なる銘柄からバーを取得する
ratesm = mt5.copy_rates_from("JP225Cash", mt5.TIMEFRAME_H1, datetime(2022,5,4,21), 300000)
rates_framem = pd.DataFrame(ratesm)
# 秒での時間をdatetime形式に変換する
rates_framem['time']=pd.to_datetime(rates_framem['time'], unit='s')
print(rates_framem)

# MetaTrader 5への接続をシャットダウンする
mt5.shutdown()

#%%
df = rates_framem.set_index('time')
df.index = df.index.tz_localize('Asia/Famagusta')
df.index = df.index.tz_convert('Asia/Tokyo')
df = df.at_time('15:00')
df

#%%
#backtest_start_date = '2016-05-26'
backtest_start_date = '2022-01-01'
df['proportion'] = None
for date in pd.unique(n225_df['date']):
	if date > backtest_start_date:
		date_jpy = '{} 15:00:00+09:00'.format(date)
		print(date_jpy)
		date_df = n225_df[n225_df['date'] == date].copy()
		print(len(date_df),len(date_df[date_df['close']>=date_df['SMA10']]),len(date_df[date_df['close']>=date_df['SMA10']])/len(date_df))
		
		dt_tz = datetime.strptime(date_jpy, '%Y-%m-%d %H:%M:%S%z')
		df.loc[df.index==dt_tz, ['proportion']]=len(date_df[date_df['close']>=date_df['SMA10']])/len(date_df)

df

#%%
buy_exit_rate = 0.25
sell_exit_rate = 0.75
def calc_exit_price(close_price=None, proportion=None, exit_rate=None):
    y = close_price.copy()
    y[:] = np.nan
    force_entry_time = close_price.copy()
    force_entry_time[:] = np.nan
    for i in range(close_price.size):
        for j in range(i + 1, close_price.size):
            if proportion[j] <= 0.25:
                y[i] = close_price[j]
                break
    return y

df = df.dropna()
df['buy_executed'] = (df['proportion']>=sell_exit_rate).astype('float64')
df['buy_exit_price'] = calc_exit_price(df['close'].astype(np.float32),df['proportion'].astype(np.float32),buy_exit_rate)
df['y_buy'] = np.where(
    df['buy_executed'],
    df['buy_exit_price']-df['close'],
    0
)

df['sell_executed'] = (df['proportion']<=buy_exit_rate).astype('float64')
df['sell_exit_price'] = calc_exit_price(df['close'].astype(np.float32),-df['proportion'].astype(np.float32),-1*sell_exit_rate)
df['y_sell'] = np.where(
    df['sell_executed'],
    df['close']-df['sell_exit_price'],
    0
)
df

#%%
print('累積リターン')
df['y_buy'].cumsum().plot(label='buy')
df['y_sell'].cumsum().plot(label='sell')
plt.title('return')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.show()