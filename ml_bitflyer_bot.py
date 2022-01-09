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
max_position = 0.03

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
		# "buy_signal":0,
		# "sell_signal":0,
		# "entry_price":0,
		# "exit_price":0,
		# "position_price":0,
		# "old_exit_price":0,
		"order":{
			"exist" : False,
			"side" : "",
			"id" : "",
			"amount" : 0,
			"price" : 0
		},
		# "position":{
		# 	"exist" : False,
		# 	"side" : "",
		# 	"count" : 0
		# },
		# "exit":{
		# 	"exist" : False,
		# 	"count" : 0
		# }
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

# ============================================= main =============================================
#flag = get_init_flag()
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

			orders = bitflyer.fetch_open_orders(symbol = product_code, params = { "product_code" : product_code })
			if orders:
				for order in orders:
					bitflyer.cancel_order(
						symbol = product_code,
						id = order["id"],
						params = { "product_code" : product_code })
					
				print("未約定の注文のキャンセル。")

			else:
				executions = bitflyer.private_get_getexecutions(params = { "product_code" : product_code })
				execution_buysell = ''
				execution_price = 0
				for execution in executions:
					if execution_buysell == '':
						execution_buysell = execution["side"]
						execution_price = np.float64(execution["price"])
					else:
						if execution_buysell != execution["side"]:
							if execution_buysell == 'BUY':
								profit = np.float64(execution["price"]) - execution_price
							else:
								profit = (execution_price - np.float64(execution["price"])) * amount

							print_log("約定履歴：{}　損益：{}".format(execution_buysell,profit))
							break
			
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
			positions = bitflyer.private_get_getpositions( params = { "product_code" : "FX_BTC_JPY" })
			pos = 0
			for position in positions:
				if position['side'].upper() == 'BUY':
					pos += np.float64(position['size'])
				else:
					pos -= np.float64(position['size'])

			buysell = ''
			entry_price = 0
			exit_flg = False

			# Exit注文
			if pos > 0:
				buysell = 'SELL'
				entry_price = sell_price
				exit_flg = True
				print("Exitの売り注文。")
			elif pos < 0:
				buysell = 'BUY'
				entry_price = buy_price
				exit_flg = True
				print("Exitの買い注文。")
			else:
				buysell = ''
				entry_price = 0
				exit_flg = False
				print("Exit注文は行いません。")
			
			if exit_flg:
				print("Exit注文：{}  {}  {}  {}  {}  {}".format(df_now['close_time_dt'],buysell,entry_price,pos,pred_buy,pred_sell))
				order = bitflyer.create_order(
					symbol = product_code,
					type='limit',
					side=buysell,
					price= entry_price,
					amount=str(amount),
					params = { "product_code" : product_code })

			# 自動売買
			entry_flg = False
			buysell = ''
			entry_price = 0
			
			if pred_buy > 0 and pos < max_position and (pred_buy>=pred_sell or pos <= (max_position*-1)):
				buysell = 'BUY'
				entry_price = buy_price
				entry_flg = True
				print("追加の買い注文。")
			elif pred_sell > 0 and pos > (max_position*-1):
				buysell = 'SELL'
				entry_flg = True
				entry_price = sell_price
				print("追加の売り注文。")
			else:
				buysell = ''
				entry_flg = False
				print("注文は行いません。")
				continue
			
			if entry_flg:
				print("エントリー注文：{}  {}  {}  {}  {}  {}".format(df_now['close_time_dt'],buysell,entry_price,pos,pred_buy,pred_sell))
				order = bitflyer.create_order(
					symbol = product_code,
					type='limit',
					side=buysell,
					price= entry_price,
					amount=str(amount),
					params = { "product_code" : product_code })

			print("注文が完了しました。")
				
		time.sleep(10)

	except Exception as e:
		print("エラー")
		print(e)
		time.sleep(10)
		continue