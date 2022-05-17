#%%
import pickle
import joblib
from datetime import datetime, timedelta
import time
import requests
import pandas as pd
import os
import traceback
import settings
from pyfeatures import features,calc_features
import numpy as np
import MetaTrader5 as mt5
import warnings
warnings.filterwarnings('ignore')

SYMBOL = 'EURGBP'
MAGIC = 12345
LOT=0.01

def iniMT5():
	if not mt5.initialize():
		print("initialize() failed, error code =",mt5.last_error())
		return False

	authorized=mt5.login(int(settings.number), password=settings.password,server=settings.server)
	if authorized:
		print("connected to account #{}".format(settings.number))
		return True
	else:
		print("failed to connect at account #{}, error code: {}".format(settings.number, mt5.last_error()))
		return False

def cancel_all_order():
    # ポジションを確認する
    orders = mt5.orders_get()
    if len(orders) == 0:
        print("未決オーダーなし")
        return

    for order in orders:
        if order.magic == MAGIC:
            print(f"Open: {order.price_open}")
            print(f"Current: {order.price_current}")
            print(f"Ticket: {order.ticket}")
            request={
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order.ticket,
            }
            result=mt5.order_send(request)
            print(result)

def check_position():
    buy_return_list = []
    sell_return_list = []

    # ポジションを確認する
    position = mt5.positions_get(symbol=SYMBOL)
    if position is None:
        print("ポジションが存在しない")
        return buy_return_list, sell_return_list

    for pos in position:
        if pos.magic != MAGIC:
            continue
        print(f"Open: {pos.price_open}")
        print(f"Current: {pos.price_current}")
        print(f"Swap: {pos.swap}")
        print(f"Profit: {pos.profit}")
        print(f"Ticket: {pos.ticket}")
        if pos.type == mt5.POSITION_TYPE_BUY:
            buy_return_list.append(pos)
        if pos.type == mt5.POSITION_TYPE_SELL:
            sell_return_list.append(pos)
        
    return buy_return_list, sell_return_list

def send_order(tp,price,sl,buysell):
    deviation = 10
    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": SYMBOL,
        "volume": LOT,
        "type": buysell,
        "price": price,
        "sl": sl,
        "tp": tp,
        # "deviation": deviation,
        "magic": MAGIC,
        "comment": "python script",
        # "expiration":expiration_datetime,
        # "type_time": mt5.ORDER_TIME_SPECIFIED,
        "type_time": mt5.ORDER_TIME_DAY,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # 取引リクエストを送信する
    result = mt5.order_send(request)

    # 実行結果を確認する
    code = result.retcode
    if code == 10009:
        print("注文完了")
    elif code == 10013:
        print("無効なリクエスト")
    elif code == 10018:
        print("マーケットが休止中")
    else:
        print("その他エラー")

    print("1. order_send(): by {} {} lots at {} with deviation={} points".format(SYMBOL,lot,price,deviation));
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("2. order_send failed, retcode={}".format(result.retcode))
        # 結果をディクショナリとしてリクエストし、要素ごとに表示する
        result_dict=result._asdict()
        for field in result_dict.keys():
            print("   {}={}".format(field,result_dict[field]))
            # これが取引リクエスト構造体の場合は要素ごとに表示する
            if field=="request":
                traderequest_dict=result_dict[field]._asdict()
                for tradereq_filed in traderequest_dict:
                    print("       traderequest: {}={}".format(tradereq_filed,traderequest_dict[tradereq_filed]))


def update_tpsl(ticket,tp,sl):
	request={
	"action": mt5.TRADE_ACTION_SLTP,
	"symbol": SYMBOL,
	"position": ticket,
	"tp": tp,
    "sl": sl,
    "magic":MAGIC,
	}
	# 取引リクエストを送信する
	result=mt5.order_send(request)
	# 実行結果を確認する
	print("update_tpsl:{}:{}".format(ticket,SYMBOL));
	if result.retcode != mt5.TRADE_RETCODE_DONE:
		print("order_send failed, retcode={}".format(result.retcode))
		print("   result",result)
	else:
		print("position #{} update, {}".format(ticket,result))



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

def get_mt5_ohlcv():
    # 数々の方法で異なる銘柄からバーを取得する
    # ratesm = mt5.copy_rates_from("EURGBP", mt5.TIMEFRAME_M15, datetime(2022,5,4,21), 400000)

    ratesm = mt5.copy_rates_from_pos(
        SYMBOL, # 銘柄 
        mt5.TIMEFRAME_M15, # 時間軸
        0, # 開始バーの位置。0は現在を表す
        100, # 取得するバーの数
    )

    rates_framem = pd.DataFrame(ratesm)
    # 秒での時間をdatetime形式に変換する
    rates_framem['time']=pd.to_datetime(rates_framem['time'], unit='s')

    df = rates_framem.rename(columns={'time': 'timestamp','open': 'op','close': 'cl','high': 'hi','low': 'lo','tick_volume': 'volume'})
    df = df.set_index('timestamp')
    df.index = df.index.tz_localize('Asia/Famagusta')
    df.index = df.index.tz_convert('Asia/Tokyo')
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
        if (dt_now.minute % interval == 0 and last_minute != dt_now.minute) or firstFlg:
            time.sleep(1)
            firstFlg = False
            last_minute = dt_now.minute


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
            model_y_buy = joblib.load('./model/model_y_buy_mlbot.xz')
            model_y_sell = joblib.load('./model/model_y_sell_mlbot.xz')

            #推論
            df_features["y_predict_buy"] = model_y_buy.predict(df_features[features])
            df_features["y_predict_sell"] = model_y_sell.predict(df_features[features])

            limit_price_dist = df_features["ATR"] * 0.5
            df_features["buy_price"] = df_features["cl"] - limit_price_dist
            df_features["sell_price"] = df_features["cl"] + limit_price_dist

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
            if position["side"] == "BUY" and abs(position["size"]) >= 0.01:
                order_side = "SELL"
                order_price = sell_price
                order_size = round(abs(position["size"]),8)
                print("買い建玉のExit注文")
                order_bitflyer(exchange,order_side,order_price,order_size)
            if position["side"] == "SELL" and abs(position["size"]) >= 0.01:
                order_side = "BUY"
                order_price = buy_price
                order_size = round(abs(position["size"]),8)
                print("売り建玉のExit注文")
                order_bitflyer(exchange,order_side,order_price,order_size)
            #エントリー
            if predict_buy > 0 and position["size"] < max_lot:
                order_side = "BUY"
                order_price = buy_price
                order_size = lot
                print("買い注文")
                order_bitflyer(exchange,order_side,order_price,order_size)
            if predict_sell > 0 and position["size"] > -max_lot:
                order_side = "SELL"
                order_price = sell_price
                order_size = lot                    
                print("売り注文")
                order_bitflyer(exchange,order_side,order_price,order_size) 


        time.sleep(10)

if __name__  == '__main__':
    lot = float(settings.lot)
    max_lot = float(settings.max_lot)
    interval = int(settings.interval)

    # MetaTrader 5に接続する
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
    firstFlg = True
    last_minute = 99

    while True:

        dt_now = datetime.now()
        
        #指定した時間間隔ごとに実行
        if (dt_now.minute % interval == 0 and last_minute != dt_now.minute) or firstFlg:
            time.sleep(1)
            firstFlg = False
            last_minute = dt_now.minute


            

            cancel_all_order()

            df = get_mt5_ohlcv()
            df_features = calc_features(df)

            #モデル読み込み
            model_y_buy = joblib.load('./model/model_y_buy_mlbot.xz')
            model_y_sell = joblib.load('./model/model_y_sell_mlbot.xz')

            #推論
            df_features["y_predict_buy"] = model_y_buy.predict(df_features[features])
            df_features["y_predict_sell"] = model_y_sell.predict(df_features[features])

            limit_price_dist = df_features["ATR"] * 0.5
            df_features["buy_price"] = df_features["cl"] - limit_price_dist
            df_features["buy_sl"] = df_features["cl"] - (limit_price_dist * 4)
            df_features["sell_price"] = df_features["cl"] + limit_price_dist
            df_features["sell_sl"] = df_features["cl"] + (limit_price_dist * 4)

            #売買判定のための情報取得
            predict_buy = df_features["y_predict_buy"].iloc[-1]
            predict_sell = df_features["y_predict_sell"].iloc[-1]

            buy_price =  df_features["buy_price"].iloc[-1]
            buy_sl =  df_features["buy_sl"].iloc[-1]
            sell_price = df_features["sell_price"].iloc[-1]
            sell_sl = df_features["sell_sl"].iloc[-1]

            print("predict_buy:{0} predict_sell:{1}".format(str(predict_buy),str(predict_sell)))
            print("buy price:{0} sell price:{1}".format(str(buy_price),str(sell_price)))
            
            buy_position_list,sell_position_list = check_position()

            if len(buy_position_list) >= 1:
                for pos in buy_position_list:
                    update_tpsl(pos.ticket,sell_price,buy_sl)
            else:
                if predict_buy > 0:
                    send_order(sell_price,buy_price,buy_sl,mt5.ORDER_TYPE_BUY_LIMIT )
            if len(sell_position_list) >= 1:
                for pos in sell_position_list:
                    update_tpsl(pos.ticket,buy_price,sell_sl)
            else:
                if predict_sell > 0:
                    send_order(buy_price,sell_price,sell_sl,mt5.ORDER_TYPE_SELL_LIMIT )

        time.sleep(10)
