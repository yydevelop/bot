#%%
import ccxt
import time
import pandas as pd
import numpy as np
import random
import requests
import settings


api_url = "https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD"

exchange = ccxt.bybit()
exchange.apiKey = settings.apiKey
exchange.secret = settings.secret
#exchange.headers = {'FTX-SUBACCOUNT': 'FTXのサブアカウント名を入力して下さい'}

TICKER = 'XAUT-PERP'
LOT = 5.5
MIN_LOT = 0.1
TICK = 0.1
MARGIN = 0.0025

#%%
try:
    while True:
        # フェアプライスを取得
        response = requests.get(api_url)
        res = response.json()
        fair_mid = (res[0]['spreadProfilePrices'][0]['bid']+res[0]['spreadProfilePrices'][0]['ask'])/2
        
        # 古いオーダーをキャンセル
        try:
            exchange.cancel_all_orders(TICKER)
        except Exception as e: print(e); pass

        try: 
            #OrderbookとPositionの取得
            ob = exchange.fetch_order_book(TICKER, limit=1)
            best_bid = ob['bids'][0][0]
            best_ask = ob['asks'][0][0]
            posi_df = pd.DataFrame(exchange.private_get_positions()['result'])
            posi_df = posi_df.apply(lambda x: pd.to_numeric(x,errors='ignore'))
            if len(posi_df)!=0:
                posi = posi_df[posi_df['future']==TICKER]['netSize'].values[0]
            else: 
                posi = 0
        except Exception as e: print(e); pass
        

        # positionがある場合にはfair_midでのアンワインド、残りのNew、反対サイドのNewの3つ
        # 新規買い
        try: 
            exchange.create_limit_order(TICKER, 'buy', LOT - posi if posi>0 else LOT, min(fair_mid * (1-MARGIN), best_ask-TICK*5))
        except Exception as e: print(e); pass    
        # 新規売り
        try: 
            exchange.create_limit_order(TICKER, 'sell', LOT - posi if posi>0 else LOT, max(fair_mid * (1+MARGIN), best_bid+TICK*5)) 
        except Exception as e: print(e); pass   
        # アンワインド
        if abs(posi)>MIN_LOT:
            try:
                exchange.create_limit_order(TICKER, 'sell' if posi>0 else 'buy', abs(posi), max(fair_mid, best_bid+TICK) if posi>0 else min(fair_mid, best_ask-TICK))     
            except Exception as e: print(e); pass    
        
        # ランダムな間隔スリープ
        time.sleep(random.randint(60, 120))
except KeyboardInterrupt: pass
except Exception as e: print(e); pass
# %%
