import pickle
import joblib
from datetime import datetime, timedelta
import time
import requests
import pandas as pd
import os
import pybitflyer
import traceback
import settings
from features import features,calc_features
import numpy as np

#BitFlyerからOHLCVデータを取得
def get_bitflyer_ohlcv(target_coin,time_scale):
    #OHLCV取得
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}
    unixtime = datetime.now().timestamp() * 1000
    ohlc_list=[]
    #1000本以上の1分足を取得
    while len(ohlc_list) < 1000:
        response = requests.get( f"https://lightchart.bitflyer.com/api/ohlc?symbol={target_coin}&period=m&before={unixtime}", headers= headers).json()
        ohlc_list.extend(response)
        current_after = datetime.fromtimestamp(response[-1][0]/1000)
        next_before = current_after - timedelta(minutes=1)
        unixtime = int(next_before.timestamp() * 1000)

    df_1m = pd.DataFrame(ohlc_list,columns=['timestamp', 'op', 'hi', 'lo', 'cl', 'volume','volume_buy_sum','volume_sell_sum','volume_buy','volume_sell'])
    df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"]/1000,unit='s', utc=True)
    df_1m.set_index("timestamp",inplace=True)
    df_1m.sort_index(inplace=True)

    #指定したタイムスケールに分足を変換
    df = pd.DataFrame()
    rule = time_scale
    df["op"] = df_1m["op"].resample(rule).first()
    df["hi"] = df_1m["hi"].resample(rule).max()
    df["lo"] = df_1m["lo"].resample(rule).min()
    df["cl"] = df_1m["cl"].resample(rule).last()
    df["volume"] = df_1m["volume"].resample(rule).sum()
    df["volume_buy"] = df_1m["volume_buy"].resample(rule).sum()
    df["volume_sell"] = df_1m["volume_sell"].resample(rule).sum()
    df = df.dropna()
    return df

def get_bitflyer_position(bitflyer): 
    poss = bitflyer.getpositions(product_code="FX_BTC_JPY")
    size= pnl = 0
    for p in poss:
        if p['side'] == 'BUY':
            size += p['size']
            pnl += p['pnl']
        if p['side'] == 'SELL':
            size -= p['size']
            pnl -= p['pnl']
    if size == 0: 
        side = 'NONE'
    elif size > 0:
        side = 'BUY'
    else:
        side = 'SELL'
    return {'side':side, 'size':size, 'pnl':pnl}

def order_bitflyer(exchange,order_side,order_price,order_size):
    order = exchange.sendchildorder(
        product_code = "FX_BTC_JPY",
        child_order_type = 'LIMIT',
        side = order_side,
        price = order_price,
        size = order_size
    )
    print(order)


#ボット起動
def start(exchange,max_lot,lot,interval):

    print("paibot started! max_lot:{0}btc lot:{1}btc interval:{2}min".format(max_lot,lot,interval))

    firstFlg = True
    last_minute = 99

    while True:

        dt_now = datetime.now()
        
        #指定した時間間隔ごとに実行
        if (dt_now.minute % interval == 0 or firstFlg) and last_minute != dt_now.minute:
            firstFlg = False
            last_minute = dt_now.minute

            try:

                #全注文をキャンセル
                exchange.cancelallchildorders(product_code="FX_BTC_JPY")

                #OHLCV情報を取得
                df_bf_fx = get_bitflyer_ohlcv("FX_BTC_JPY","15T")
                df = df_bf_fx.dropna()
                df.to_csv('./df.csv')
                print(df.iloc[-1].name + timedelta(hours=9),df.iloc[-1]['cl'])
            

                #特徴量計算
                df_features = calc_features(df)

                #モデル読み込み
                model_y_buy = joblib.load('./model/model_y_buy_bffx.xz')
                model_y_sell = joblib.load('./model/model_y_sell_bffx.xz')

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

                position = get_bitflyer_position(exchange)

                print("predict_buy:{0} predict_sell:{1}".format(str(predict_buy),str(predict_sell)))
                print("buy price:{0} sell price:{1}".format(str(buy_price),str(sell_price)))
                print("position side:{0} size:{1}".format(str(position["side"]),str(position["size"])))

                #注文処理

                order_side = "NONE"

                #エグジット
                if predict_buy < 0 and position["side"] == "BUY":
                    order_side = "SELL"
                    order_price = sell_price
                    order_size = round(abs(position["size"]),8)
                    order_bitflyer(exchange,order_side,order_price,order_size)
                if predict_sell < 0 and position["side"] == "SELL":
                    order_side = "BUY"
                    order_price = buy_price
                    order_size = round(abs(position["size"]),8)
                    order_bitflyer(exchange,order_side,order_price,order_size)
                #エントリー
                if predict_buy > 0 and position["size"] < max_lot:
                    order_side = "BUY"
                    order_price = buy_price
                    order_size = lot
                    order_bitflyer(exchange,order_side,order_price,order_size)
                if predict_sell > 0 and position["size"] > -max_lot:
                    order_side = "SELL"
                    order_price = sell_price
                    order_size = lot                    
                    order_bitflyer(exchange,order_side,order_price,order_size) 

            except Exception as e:
                print(traceback.format_exc())
                pass

        time.sleep(10)

if __name__  == '__main__':
    apiKey    = settings.apiKey
    secretKey = settings.secret
    lot = float(settings.lot)
    max_lot = float(settings.max_lot)
    interval = int(settings.interval)

    exchange = pybitflyer.API(api_key=apiKey, api_secret=secretKey)
    start(exchange, max_lot, lot, interval)


