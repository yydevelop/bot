import pickle
import joblib
from datetime import datetime, timedelta
import time
import requests
import pandas as pd
import os
import settings
import traceback

from exchange.gmocoin import GmoCoinExchange
from features import features,calc_features

#GMOからOHLCVデータを取得
def get_gmo_ohlcv(day,interval):
    endPoint = 'https://api.coin.z.com/public'
    path     = '/v1/klines?symbol=BTC_JPY&interval='+interval+'min&date='+day

    response = requests.get(endPoint + path)
    df = pd.DataFrame(response.json()["data"])
    df.rename(columns={'openTime':'timestamp','open':'op','high':'hi','low':'lo','close':'cl'},inplace=True)
    df["op"]=df["op"].astype(float)
    df["hi"]=df["hi"].astype(float)
    df["lo"]=df["lo"].astype(float)
    df["cl"]=df["cl"].astype(float)
    df["volume"]=df["volume"].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'] , unit='ms')
    df.set_index("timestamp",inplace=True)
    return df

#ボット起動
def start(exchange,max_lot,lot,interval):

    print("paibot started! max_lot:{0}btc lot:{1}btc interval:{2}min".format(max_lot,lot,interval))
    firstFlg = True
    last_minute = 99

    while True:

        dt_now = datetime.now()
        
        #15分ごとに実行
        if (dt_now.minute % interval == 0 or firstFlg) and last_minute != dt_now.minute:
            firstFlg = False
            last_minute = dt_now.minute

            try:

                #全注文をキャンセル
                exchange.create_cancel_all_order("BTC_JPY")

                #有効注文がなくなるまで待機
                while True:
                    orders = exchange.get_active_orders("BTC_JPY")
                    if len(orders["data"]) == 0:
                        break
                    time.sleep(1)

                #OHLCV情報を取得
                #今日と昨日のklineを取得してマージ
                today = datetime.today()
                yesterday = today - timedelta(days=1)

                today = today.strftime('%Y%m%d')
                yesterday = yesterday.strftime('%Y%m%d')

                df_today = get_gmo_ohlcv(today,str(interval))
                df_yesterday = get_gmo_ohlcv(yesterday,str(interval))
                df = pd.concat([df_today, df_yesterday[~df_yesterday.index.isin(df_today.index)]]).sort_index()
                print(df.iloc[-1].name + timedelta(hours=9),df.iloc[-1]['cl'])

                #特徴量計算
                df_features = calc_features(df)

                #モデル読み込み
                model_y_buy = joblib.load('./model/model_y_buy.xz')
                model_y_sell = joblib.load('./model/model_y_sell.xz')

                #推論
                df_features["y_predict_buy"] = model_y_buy.predict(df_features[features])
                df_features["y_predict_sell"] = model_y_sell.predict(df_features[features])

                df_features["buy_price"] = df_features["cl"] - df_features["ATR"] * 0.5
                df_features["sell_price"] = df_features["cl"] + df_features["ATR"] * 0.5

                #売買判定のための情報取得
                predict_buy = df_features["y_predict_buy"].iloc[-1]
                predict_sell = df_features["y_predict_sell"].iloc[-1]

                buy_price =  int(df_features["buy_price"].iloc[-1])
                sell_price = int(df_features["sell_price"].iloc[-1])

                position = exchange.get_position("BTC_JPY")
                total_quantity = position["buy"] + position["sell"]
                position_quantity = position["buy"] - position["sell"]

                print("predict_buy:{0} predict_sell:{1}".format(str(predict_buy),str(predict_sell)))
                print("buy price:{0} sell price:{1}".format(str(buy_price),str(sell_price)))
                print("position buy:{0} position sell:{1}".format(str(position["buy"]),str(position["sell"])))

                #注文処理

                #エグジット
                if position["buy"] > 0:
                    exchange.create_limit_close_bulk_order("BTC_JPY","SELL",abs(position_quantity),sell_price)
                if position["sell"] > 0:
                    exchange.create_limit_close_bulk_order("BTC_JPY","BUY",abs(position_quantity),buy_price)                           

                #エントリー
                if predict_buy > 0 and position_quantity < max_lot:
                    if position["sell"] >= lot and predict_sell > 0 :
                        exchange.create_limit_close_bulk_order("BTC_JPY","BUY",lot,buy_price)   
                    else:
                        exchange.create_limit_order("BTC_JPY","BUY",lot,buy_price)
                if predict_sell > 0 and position_quantity > -max_lot:
                    if position["buy"] >= lot and  predict_buy > 0 :
                        exchange.create_limit_close_bulk_order("BTC_JPY","SELL",lot,sell_price)
                    else:
                        exchange.create_limit_order("BTC_JPY","SELL",lot,sell_price)

            except Exception as e:
                print(traceback.format_exc())
                pass

        time.sleep(10)

if __name__  == '__main__':
    apiKey    = str(settings.gmoApiKey)
    secretKey = str(settings.gmoSecret)
    lot = float(settings.lot)
    max_lot = float(settings.max_lot)
    interval = int(settings.interval)

    exchange = GmoCoinExchange(apiKey,secretKey)
    start(exchange, max_lot, lot, interval)
