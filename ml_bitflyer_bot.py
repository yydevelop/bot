import requests
from datetime import datetime, timedelta
import ccxt
import pandas as pd
import numpy as np
import talib
import settings
import time
import joblib
from logging import getLogger,Formatter,StreamHandler,FileHandler,INFO

# モデルの読み込み
model_y_buy = joblib.load('./model_y_buy.xz')
model_y_sell = joblib.load('./model_y_sell.xz')

# ccxtのパラメータ
symbol = 'BTC/JPY'	  # 購入予定のシンボル
product_code = 'FX_BTC_JPY'
bitflyer = ccxt.bitflyer()		 # 使用する取引所を記入
bitflyer.apiKey = settings.apiKey
bitflyer.secret = settings.secret

min_amount = 0.01
amount = 0.01
exit_max = 5
exit_cut = 20
ashi = 60 * 15

# 特徴量作成
def calc_features(df):
	open = df['op']
	high = df['hi']
	low = df['lo']
	close = df['cl']
	volume = df['volume']
	
	orig_columns = df.columns

	hilo = (df['hi'] + df['lo']) / 2
	df['BBANDS_upperband'], df['BBANDS_middleband'], df['BBANDS_lowerband'] = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
	df['BBANDS_upperband'] -= hilo
	df['BBANDS_middleband'] -= hilo
	df['BBANDS_lowerband'] -= hilo
	df['DEMA'] = talib.DEMA(close, timeperiod=30) - hilo
	df['EMA'] = talib.EMA(close, timeperiod=30) - hilo
	df['HT_TRENDLINE'] = talib.HT_TRENDLINE(close) - hilo
	df['KAMA'] = talib.KAMA(close, timeperiod=30) - hilo
	df['MA'] = talib.MA(close, timeperiod=30, matype=0) - hilo
	df['MIDPOINT'] = talib.MIDPOINT(close, timeperiod=14) - hilo
	df['SMA'] = talib.SMA(close, timeperiod=30) - hilo
	df['T3'] = talib.T3(close, timeperiod=5, vfactor=0) - hilo
	df['TEMA'] = talib.TEMA(close, timeperiod=30) - hilo
	df['TRIMA'] = talib.TRIMA(close, timeperiod=30) - hilo
	df['WMA'] = talib.WMA(close, timeperiod=30) - hilo

	df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
	df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
	df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
	df['AROON_aroondown'], df['AROON_aroonup'] = talib.AROON(high, low, timeperiod=14)
	df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
	df['BOP'] = talib.BOP(open, high, low, close)
	df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
	df['DX'] = talib.DX(high, low, close, timeperiod=14)
	df['MACD_macd'], df['MACD_macdsignal'], df['MACD_macdhist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
	# skip MACDEXT MACDFIX たぶん同じなので
	df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
	df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
	df['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)
	df['MOM'] = talib.MOM(close, timeperiod=10)
	df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
	df['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)
	df['RSI'] = talib.RSI(close, timeperiod=14)
	df['STOCH_slowk'], df['STOCH_slowd'] = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
	df['STOCHF_fastk'], df['STOCHF_fastd'] = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
	df['STOCHRSI_fastk'], df['STOCHRSI_fastd'] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
	df['TRIX'] = talib.TRIX(close, timeperiod=30)
	df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
	df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

	df['AD'] = talib.AD(high, low, close, volume)
	df['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
	df['OBV'] = talib.OBV(close, volume)

	df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
	df['NATR'] = talib.NATR(high, low, close, timeperiod=14)
	df['TRANGE'] = talib.TRANGE(high, low, close)

	df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
	df['HT_DCPHASE'] = talib.HT_DCPHASE(close)
	df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(close)
	df['HT_SINE_sine'], df['HT_SINE_leadsine'] = talib.HT_SINE(close)
	df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)

	df['BETA'] = talib.BETA(high, low, timeperiod=5)
	df['CORREL'] = talib.CORREL(high, low, timeperiod=30)
	df['LINEARREG'] = talib.LINEARREG(close, timeperiod=14) - close
	df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(close, timeperiod=14)
	df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(close, timeperiod=14) - close
	df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close, timeperiod=14)
	df['STDDEV'] = talib.STDDEV(close, timeperiod=5, nbdev=1)

	return df

features = sorted([
	'ADX',
	'ADXR',
	'APO',
	'AROON_aroondown',
	'AROON_aroonup',
	'AROONOSC',
	'CCI',
	'DX',
	'MACD_macd',
	'MACD_macdsignal',
	'MACD_macdhist',
	'MFI',
#	 'MINUS_DI',
#	 'MINUS_DM',
	'MOM',
#	 'PLUS_DI',
#	 'PLUS_DM',
	'RSI',
	'STOCH_slowk',
	'STOCH_slowd',
	'STOCHF_fastk',
#	 'STOCHRSI_fastd',
	'ULTOSC',
	'WILLR',
#	 'ADOSC',
#	 'NATR',
	'HT_DCPERIOD',
	'HT_DCPHASE',
	'HT_PHASOR_inphase',
	'HT_PHASOR_quadrature',
	'HT_TRENDMODE',
	'BETA',
	'LINEARREG',
	'LINEARREG_ANGLE',
	'LINEARREG_INTERCEPT',
	'LINEARREG_SLOPE',
	'STDDEV',
	'BBANDS_upperband',
	'BBANDS_middleband',
	'BBANDS_lowerband',
	'DEMA',
	'EMA',
	'HT_TRENDLINE',
	'KAMA',
	'MA',
	'MIDPOINT',
	'T3',
	'TEMA',
	'TRIMA',
	'WMA',
])


# ログの設定
logger = getLogger(__name__)
handlerSh = StreamHandler()
handlerFile = FileHandler("./crypt_bot.log")
handlerSh.setLevel(INFO)
handlerFile.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handlerSh)
logger.addHandler(handlerFile)

# LINEの設定
line_token = settings.token

bitflyer = ccxt.bitflyer()
bitflyer.apiKey = settings.apiKey
bitflyer.secret = settings.secret

# print文のかわりに使用
def print_log( text ):
	
	# コマンドラインへの出力とファイル保存
	logger.info( text )
	
	# LINEへの通知
	url = "https://notify-api.line.me/api/notify"
	data = {"message" : text}
	headers = {"Authorization": "Bearer " + line_token} 
	requests.post(url, data=data, headers=headers)


# ここからメイン
def get_init_flag():
	print("flag情報クリア")
	lFlag = {
		"init_flag": True,
		"buy_signal":0,
		"sell_signal":0,
		"entry_price":0,
		"exit_price":0,
		"position_price":0,
		"old_exit_price":0,
		"order":{
			"exist" : False,
			"side" : "",
			"count" : 0
		},
		"position":{
			"exist" : False,
			"side" : "",
			"count" : 0
		},
		"exit":{
			"exist" : False,
			"count" : 0
		}
	}
	return lFlag

# def get_init_flag_first():
# 	print("flag情報クリア")
# 	lFlag = get_init_flag()
# 	lFlag["init_flag"] = True
# 	return lFlag

def get_price(min, before=0, after=0):
	price = []
	params = {"periods" : min ,"limit":100}
	if before != 0:
		params["before"] = before
	if after != 0:
		params["after"] = after

	response = requests.get("https://api.cryptowat.ch/markets/bitflyer/btcfxjpy/ohlc",params)
	data = response.json()
	print(data['allowance']['remaining'],data['allowance']['cost'],)
	for i in data["result"][str(min)]:
		if i[1] != 0 and i[2] != 0 and i[3] != 0 and i[4] != 0:
			price.append({ "close_time" : i[0],
				"close_time_dt" : datetime.fromtimestamp(i[0]).strftime('%Y/%m/%d %H:%M'),
				"open_price" : i[1],
				"high_price" : i[2],
				"low_price" : i[3],
				"close_price": i[4],
				"volume": i[5] })
	return price

# 時間と始値・終値を表示する関数
def print_price( data ):
	print( "時間： " + datetime.fromtimestamp(data["close_time"]).strftime('%Y/%m/%d %H:%M') + " 始値： " + str(data["open_price"]) + " 終値： " + str(data["close_price"]) )

# 注文をキャンセルする関数
def cancel_exit( orders,flag ):
	try:
		for o in orders:
			bitflyer.cancel_order(
				symbol = product_code,
				id = o["id"],
				params = { "product_code" : "FX_BTC_JPY" })
		print("約定していないExit注文をキャンセルしました")
		flag["old_exit_price"] = flag["exit_price"]
		flag["exit"]["exist"] = False
		flag["exit"]["count"] = 0
		flag["exit_price"] = 0
		
		time.sleep(20)
		position = bitflyer.private_get_getpositions( params = { "product_code" : "FX_BTC_JPY" })
		if position:
			print("現在、まだ未決済の建玉があります")
			flag["position"]["exist"] = True
			flag["position"]["side"] = str(position[0]["side"])
			flag["position_price"] = np.float64(position[0]['price'])
		else:
			print("現在、未決済の建玉はありません")
			flag = get_init_flag()
	except ccxt.BaseError as e:
		print("BitflyerのAPIで問題発生 ： ", e)
	finally:
		return flag


def exit_sashi( flag ):
	if flag["position"]["side"].upper() == 'BUY':
		exit_side = 'SELL'
	else:
		exit_side = 'BUY'
	while True:
		try:
			order = bitflyer.create_order(
#				symbol = 'BTC/JPY',
				symbol = product_code,
				type='limit',
				side=exit_side,
				price= flag["exit_price"],
				amount=str(amount),
				params = { "product_code" : "FX_BTC_JPY" })
			flag["exit"]["exist"] = True
			flag["order"]["exist"] = False
			flag["order"]["count"] = 0
			flag["order"]["side"] = ''
			time.sleep(20)
			print("Exit注文が完了しました")
			break
		except ccxt.BaseError as e:
			print("BitflyerのAPIでエラー発生",e)
			print("Exit注文の通信が失敗しました。20秒後に再トライします")
			time.sleep(20)
	return flag


def cancel_position(position, orders, flag ):
	if position[0]['side'].upper() == 'BUY':
		exit_side = 'SELL'
	else:
		exit_side = 'BUY'
	while True:
		try:
			for o in orders:
				print("オーダーのキャンセル：",o["id"])
				bitflyer.cancel_order(
					symbol = product_code,
					id = o["id"],
					params = { "product_code" : "FX_BTC_JPY" })

			exit_amount = np.float64(position[0]['size'])
			if exit_amount < min_amount:
				order = bitflyer.create_order(
					symbol = product_code,
					type='market',
					side=position[0]['side'],
					amount=str(min_amount),
					params = { "product_code" : "FX_BTC_JPY" })
				exit_amount += min_amount
			time.sleep(10)
			
			order = bitflyer.create_order(
#				symbol = 'BTC/JPY',
				symbol = product_code,
				type='market',
				side=exit_side,
				amount=str(exit_amount),
				params = { "product_code" : "FX_BTC_JPY" })
			
			time.sleep(10)
			flag = get_init_flag()
			print("成行注文が完了しました")
			break
		except ccxt.BaseError as e:
			print("BitflyerのAPIでエラー発生",e)
			print("成行注文の通信が失敗しました。20秒後に再トライします")
			time.sleep(20)
	return flag

# サーバーに出した注文が約定したかどうかチェックする関数
def check_status( flag ):
	try:
		position = bitflyer.private_get_getpositions( params = { "product_code" : "FX_BTC_JPY" })
		orders = bitflyer.fetch_open_orders(
			symbol = product_code,
			params = { "product_code" : "FX_BTC_JPY" })
	except ccxt.BaseError as e:
		print("BitflyerのAPIで問題発生 : ",e)
	else:
		if position:
			flag["position"]["count"] += 1

			if flag["position"]["count"] >= exit_cut:
				print("ロスカット回数を超過したので成行注文でポジションを解消します")
				flag = cancel_position(position, orders, flag)
				return flag

			if (np.float64(position[0]["size"]) % amount) != 0:
				print("半端ポジションを解消します。")
				flag = cancel_position(position, orders, flag)
				return flag

			if orders:
				if orders[0]['amount'] == amount*2:
					print("ドテン注文が約定しなかったのでキャンセルします")
					flag["exit_price"] = flag["old_exit_price"]
					flag = cancel_order( orders,flag )

				elif str(position[0]['side']) == str(orders[0]['side']):
					print("同じ方向のexit注文があるのでキャンセルします")
					flag = cancel_exit( orders,flag )
				else:
					print("ポジションとExit注文の両方があります")
					# 既存の注文あり
					flag["order"]["exist"] = False
					flag["order"]["count"] = 0
					flag["order"]["side"] = ''
					flag["position"]["side"] = str(position[0]['side'])
					flag["position"]["exist"] = True
					flag["exit"]["exist"] = True
					flag["exit"]["count"] += 1
					flag["position_price"] = np.float64(position[0]['price'])
					if flag["exit_price"] == 0:
						flag["exit_price"] = np.float64(orders[0]['price'])
			else:
				print("ポジションを持っています")
				flag["order"]["exist"] = False
				flag["order"]["count"] = 0
				flag["order"]["side"] = ''
				flag["position"]["exist"] = True
				flag["position"]["side"] = str(position[0]['side'])
				flag["position_price"] = position[0]['price']

		else:
			if orders:
				print("まだ未約定の注文があります")
				for o in orders:
					print( o["id"] )
				flag["order"]["count"] += 1
				
				if flag["order"]["count"] > 0:
					print("注文が約定しなかったのでキャンセルします")
					flag = cancel_order( orders,flag )
			else:
				if flag["exit"]["exist"]:
					if flag["position"]["side"].upper() == 'BUY':
						print_log("Exit注文約定：損益＝{}".format(flag["exit_price"] - flag["position_price"]))
					else:
						print_log("Exit注文約定：損益＝{}".format(flag["position_price"] - flag["exit_price"]))

				flag["order"]["exist"] = False
				flag["order"]["count"] = 0
				flag["order"]["side"] = ''
				flag["position"]["exist"] = False
				flag["position"]["side"] = ""
				flag["exit"]["exist"] = False
				flag["exit"]["count"] = 0
				flag["exit_price"] = 0
				print("未約定の注文はありません")
	return flag

# サーバーに出した注文が約定したかどうかチェックする関数
def check_order( flag ):
	try:
		position = bitflyer.private_get_getpositions( params = { "product_code" : "FX_BTC_JPY" })
		orders = bitflyer.fetch_open_orders(
			symbol = product_code,
			params = { "product_code" : "FX_BTC_JPY" })
	except ccxt.BaseError as e:
		print("BitflyerのAPIで問題発生 : ",e)
	else:
		if position:
			if not orders:
				if flag["position"]["exist"]:
					print("ドテン注文が約定しました！")
					if flag["position"]["side"].upper() == 'BUY':
						print_log("ドテン注文：損益＝{}".format(flag["entry_price"] - flag["position_price"]))
					else:
						print_log("ドテン注文：損益＝{}".format(flag["position_price"] - flag["entry_price"]))

				else:
					print("注文が約定しました！")
				
				flag["position"]["count"] = 0
				flag["order"]["exist"] = False
				flag["order"]["count"] = 0
				flag["order"]["side"] = ''
				flag["position"]["exist"] = True
				flag["position"]["side"] = str(position[0]['side'])
				flag["position_price"] = np.float64(position[0]['price'])
	return flag

# サーバーに出したExit注文が約定したかどうかチェックする関数
def check_exit( flag , kbn):
	print("check_exit")
	try:
		position = bitflyer.private_get_getpositions( params = { "product_code" : "FX_BTC_JPY" })
		orders = bitflyer.fetch_open_orders(
#			symbol = "BTC/JPY",
			symbol = product_code,
			params = { "product_code" : "FX_BTC_JPY" })
	except ccxt.BaseError as e:
		print("BitflyerのAPIで問題発生 : ",e)
	else:
		if position:
			if orders:
				print("まだ未約定のExit注文があります")
				for o in orders:
					print( o["id"] )
				if kbn == "entry":
					if flag["exit"]["count"] >= exit_max:
						print("Exit注文をキャンセルします。")
						flag = cancel_exit( orders,flag )
				else:
						flag["exit"]["exist"] = True
						flag["exit"]["count"] += 1

			else:
				flag["order"]["exist"] = False
				flag["order"]["count"] = 0
				flag["position"]["exist"] = True
				flag["position"]["side"] = str(position[0]['side'])
				flag["position_price"] = np.float64(position[0]['price'])
				flag["exit"]["exist"] = False
				flag["exit"]["count"] = 0
				print("ポジションが残っているようです")
		else:
			if orders:
				print("Exit注文だけが残っています")
				for o in orders:
					print( o["id"] )
				
				if flag["exit"]["count"] >= exit_max:
					print("Exit注文をキャンセルします。")
					flag = cancel_exit( orders,flag )
			else:
				flag["order"]["exist"] = False
				flag["order"]["count"] = 0
				flag["position"]["exist"] = False
				flag["position"]["side"] = ''
				flag["position_price"] = 0
				flag["exit"]["exist"] = False
				flag["exit"]["count"] = 0
	return flag

# 注文をキャンセルする関数
def cancel_order( orders,flag ):
	try:
		for o in orders:
			bitflyer.cancel_order(
#				symbol = "BTC/JPY",
				symbol = product_code,
				id = o["id"],
				params = { "product_code" : "FX_BTC_JPY" })
		print("約定していない注文をキャンセルしました")
		flag["order"]["exist"] = False
		flag["order"]["count"] = 0
		flag["order"]["side"] = ''

		time.sleep(20)
		position = bitflyer.private_get_getpositions( params = { "product_code" : "FX_BTC_JPY" })
		if position:
			print("現在、まだ未決済の建玉があります")
			flag["position"]["exist"] = True
			flag["position"]["side"] = position[0]["side"]
			flag["position_price"] = position[0]['price']
			flag["exit"]["exist"] = False
		else:
			print("現在、未決済の建玉はありません")
			flag = get_init_flag()
	except ccxt.BaseError as e:
		print("BitflyerのAPIで問題発生 ： ", e)
	finally:
		return flag



# ============================================= main =============================================
flag = get_init_flag()
firstFlg = True
last_minute = 99

print_log("bytflyerBOT起動")

while True:
	try:
		now = datetime.now() + timedelta(hours=9)

		# 15分ごとにローソク足を取得して特徴量を作成し、モデルで予測します
		if (now.minute % 15 == 0 or firstFlg) and last_minute != now.minute:
			print(now)
			last_minute = now.minute
			firstFlg = False
			# while flag["order"]["exist"] or flag["init_flag"]:
			# 	flag["init_flag"] = False
			# 	flag = check_status( flag )
			flag = check_status( flag )

			print(flag)

			# 不要？
			# if flag["position"]["exist"]:
			# 	if flag["exit"]["exist"]:
			# 		flag = check_exit( flag , "check")
			# 	else:
			# 		if flag["exit_price"] != 0:
			# 			flag = exit_sashi( flag )
			# 			time.sleep(20)
			# 			continue

			# for _ in range(3):
			# 	price = get_price(60*15)
			# 	if price is None:
			# 		time.sleep(10)
			# 		continue
			# 	break
			price = get_price(60*15)

			df = pd.DataFrame(price)
			df=df.rename(columns={
					"open_price" : "op",
					"high_price" : "hi",
					"low_price" : "lo",
					"close_price": "cl",
					"volume": "volume",
			})
			df['close_time_dt'] = pd.to_datetime(df['close_time_dt'])
			df['datetime'] = df['close_time_dt']
			df['cl'] = df['cl'].astype(np.float64)
			df['op'] = df['op'].astype(np.float64)
			df['hi'] = df['hi'].astype(np.float64)
			df['lo'] = df['lo'].astype(np.float64)
			df['cl'] = df['cl'].astype(np.float64)

			df = df.set_index('datetime')

			# 特徴量作成
			df_features = calc_features(df)

			# 予測
			df_features['y_pred_buy'] = model_y_buy.predict(df_features[features])
			df_features['y_pred_sell'] = model_y_sell.predict(df_features[features])

			# 予測結果
			pred_buy = df_features['y_pred_buy'].iloc[-1]
			pred_sell = df_features['y_pred_sell'].iloc[-1]
			
			df_now = df_features.iloc[-1]

			# 呼び値 (取引所、取引ペアごとに異なるので、適切に設定してください)
			pips = 1

			# ATRで指値距離を計算します
			limit_price_dist = df_now['ATR'] * 0.5
			limit_price_dist = np.maximum(1, (limit_price_dist / pips).round()) * pips

			# 終値から両側にlimit_price_distだけ離れたところに、買い指値と売り指値を出します
			buy_price = df_now['cl'] - limit_price_dist
			sell_price = df_now['cl'] + limit_price_dist

			df[['close_time_dt','cl','y_pred_buy','y_pred_sell']].to_csv('./pred.csv')
			print(df_now['close_time_dt'],df_now['y_pred_buy'],df_now['y_pred_sell'],df_now['cl'])

			# 予測結果による売買
			buysell = ''
			weight = 0
			if pred_buy > 0 and pred_buy>=pred_sell:
				buysell = 'buy'
				entry_price = buy_price
				exit_price = sell_price
				weight = 1
			elif pred_sell > 0:
				buysell = 'sell'
				weight = -1
				entry_price = sell_price
				exit_price = buy_price
			
			if buysell != '' and flag["position"]["side"] != buysell.upper():
				# if flag["position"]["exist"] and flag["exit"]["exist"] == True:
				# 	print("Exit注文の約定を待ちます")
				# 	time.sleep(60)
				# 	continue

				try:
					print("Exit注文のチェック・キャンセル")
					flag = check_exit( flag , "entry")
					if not flag["exit"]["exist"]:

						#if flag["position"]["exist"] and flag["exit"]["exist"] == False:
						if flag["position"]["exist"]:
							print("ドテン注文を行います。")
							# ドテン用に２倍にする処理
							order_amount = amount * 2
						else:
							print("通常の注文を行います。")
							order_amount = amount

						print("注文：{}  {}  {}  {}  {}".format(df_now['close_time_dt'],buysell,df_now['cl'],entry_price,exit_price))
						order = bitflyer.create_order(
							symbol = product_code,
							type='limit',
							side=buysell,
							price= entry_price,
							#amount='0.01',
							amount=str(order_amount),
							params = { "product_code" : product_code })
						time.sleep(20)
						print("注文が完了しました。",order)
						flag["entry_price"] = entry_price
						flag["exit_price"] = exit_price
						flag["order"]["exist"] = True
						flag["order"]["side"] = buysell.upper()
				except ccxt.BaseError as e:
					print("BitflyerのAPIでエラー発生",e)
					print("注文が失敗しました")
			else:
				print("AIの判断により注文は行いません。")
				if flag["position"]["exist"]:
					if not flag["exit"]["exist"]:
						if flag["exit_price"] != 0:
							print("Exit注文を再度発行します")
							flag = exit_sashi( flag )
							time.sleep(20)
							continue

		time.sleep(10)
		if flag["order"]["exist"]:
			flag = check_order( flag )

			if not flag["order"]["exist"] and flag["position"]["exist"]:
				if not flag["exit"]["exist"]:
					if flag["exit_price"] != 0:
						flag = exit_sashi( flag )
			time.sleep(20)
			continue

	except Exception as e:
		print("エラー")
		print(e)
		flag = get_init_flag()
		time.sleep(10)
		continue