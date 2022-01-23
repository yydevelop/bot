import pickle
import joblib
from datetime import datetime, timedelta
import time
import requests
import pandas as pd
import numpy as np
import os
import ccxt
import traceback
import settings
from features import features,calc_features

#BybitのOHLCV情報を取得
def get_bybit_ohlcv(from_time,interval,limit):
    ohlcv_list = ccxt.bybit().publicLinearGetKline({
        'symbol': "BTCUSDT",
        'from': from_time,
        'interval': interval,
        'limit': limit
    })["result"]

    df = pd.DataFrame(ohlcv_list,columns=['open_time', 'open', 'high','low','close','volume']) 
    df.rename(columns={'open_time':'timestamp','open':'op','high':'hi','low':'lo','close':'cl'},inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'] , unit='s')
    df.set_index("timestamp",inplace=True)
    df["op"]=df["op"].astype(float)
    df["hi"]=df["hi"].astype(float)
    df["lo"]=df["lo"].astype(float)
    df["cl"]=df["cl"].astype(float)
    df["volume"]=df["volume"].astype(float)
    df.sort_index(inplace=True)
    return df

#ポジション情報を取得
def get_bybit_position(bybit):
    poss = bybit.privateLinearGetPositionList({"symbol": 'BTCUSDT'})['result']
    size= pnl = 0.0
    print(poss)
    for p in poss:
        if p['side'] == 'Buy':
            size += float(p['size'])
            pnl += float(p['unrealised_pnl'])
        if p['side'] == 'Sell':
            size -= float(p['size'])
            pnl -= float(p['unrealised_pnl'])
    if size == 0: 
        side = 'NONE'
    elif size > 0:
        side = 'BUY'
    else:
        side = 'SELL'
    return {'side':side, 'size':size, 'pnl':pnl}

#Bybitへ注文
def order_bybit(exchange,order_side,order_price,order_size):
    order = exchange.private_linear_post_order_create(
        {
            "side": order_side,
            "symbol": "BTCUSDT",
            "order_type": "Limit",
            "qty": order_size,
            "price":order_price,
            "time_in_force": "PostOnly",
            "reduce_only": False,
            "close_on_trigger": False,
            "position_idx":0
        }
    )
    print(order)

#ボット起動
def start(exchange,max_lot,lot,interval):

    print("paibot for bybit is started!\nmax_lot:{0}btc lot:{1}btc interval:{2}min".format(max_lot,lot,interval))

    first_flg = True
    while True:

        dt_now = datetime.now()
        
        #指定した時間間隔ごとに実行
        if dt_now.minute % interval == 0 or first_flg:
            first_flg = False
            try:

                #全注文をキャンセル
                exchange.cancelAllOrders("BTC/USDT")

                #OHLCV情報を取得
                time_now = datetime.now()
                from_time = int((time_now + timedelta(minutes= - 200 * interval)).timestamp())
                limit = 200
                df = get_bybit_ohlcv(from_time,interval,limit)

                #特徴量計算
                df_features = calc_features(df)

                #モデル読み込み
                model_y_buy = joblib.load('./model/model_y_buy_bybit.xz')
                model_y_sell = joblib.load('./model/model_y_sell_bybit.xz')

                #推論
                df_features["y_predict_buy"] = model_y_buy.predict(df_features[features])
                df_features["y_predict_sell"] = model_y_sell.predict(df_features[features])

                pips = 0.5
                limit_price_dist = df_features['ATR'] * 0.5
                limit_price_dist = np.maximum(1, (limit_price_dist / pips).round().fillna(1)) * pips
                df_features["buy_price"] = df_features["cl"] - limit_price_dist 
                df_features["sell_price"] = df_features["cl"] + limit_price_dist 

                #売買判定のための情報取得
                predict_buy = df_features["y_predict_buy"].iloc[-1]
                predict_sell = df_features["y_predict_sell"].iloc[-1]

                buy_price =  df_features["buy_price"].iloc[-1]
                sell_price = df_features["sell_price"].iloc[-1]

                position = get_bybit_position(exchange)

                print("predict_buy:{0} predict_sell:{1}".format(str(predict_buy),str(predict_sell)))
                print("buy price:{0} sell price:{1}".format(str(buy_price),str(sell_price)))
                print("position side:{0} size:{1}".format(str(position["side"]),str(position["size"])))

                #注文処理

                order_side = "NONE"

                #エグジット
                if position["side"] == "BUY":
                    order_side = "Sell"
                    order_price = sell_price
                    order_size = abs(position["size"])
                    order_bybit(exchange,order_side,order_price,order_size)
                    print("買い建玉のExit注文")
                if position["side"] == "SELL":
                    order_side = "Buy"
                    order_price = buy_price
                    order_size = abs(position["size"])
                    order_bybit(exchange,order_side,order_price,order_size)
                    print("売り建玉のExit注文")
                #エントリー
                if predict_buy > 0 and position["size"] < max_lot:
                    order_side = "Buy"
                    order_price = buy_price
                    order_size = lot
                    order_bybit(exchange,order_side,order_price,order_size)
                    print("買い注文")
                if predict_sell > 0 and position["size"] > -max_lot:
                    order_side = "Sell"
                    order_price = sell_price
                    order_size = lot                    
                    order_bybit(exchange,order_side,order_price,order_size)
                    print("売り注文")

            except Exception as e:
                print(traceback.format_exc())
                pass

        time.sleep(60)

if __name__  == '__main__':
    apiKey    = settings.bybitApiKey
    secretKey = settings.bybitSecret
    lot = float(settings.lot)
    max_lot = float(settings.max_lot)
    interval = int(settings.interval)

    # apiKey    = os.getenv("API_KEY")
    # secretKey = os.getenv("SECRET_KEY")
    # lot = float(os.getenv("LOT"))
    # max_lot = float(os.getenv("MAX_LOT"))
    # interval = int(os.getenv("INTERVAL"))

    exchange = ccxt.bybit({"apiKey":apiKey, "secret":secretKey})
    start(exchange, max_lot, lot, interval)


