#%%
from typing import final
import numpy as np
import sys
import pandas as pd
import datetime
import os
import talib as ta
import MetaTrader5 as mt5
import settings
import requests
from logging import getLogger,Formatter,StreamHandler,FileHandler,INFO
# ログの設定
logger = getLogger(__name__)
handlerSh = StreamHandler()
handlerFile = FileHandler("./jp225_ratemodel.log")
handlerSh.setLevel(INFO)
handlerFile.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handlerSh)
logger.addHandler(handlerFile)

MAGIC=843221
SYMBOL = 'JP225'
LOT = 1.0

# LINEの設定
line_token = settings.token


# print文のかわりに使用
def print_log( text ):
	# コマンドラインへの出力とファイル保存
	logger.info( text )

def print_line(text):
	# LINEへの通知
	url = "https://notify-api.line.me/api/notify"
	data = {"message" : text}
	headers = {"Authorization": "Bearer " + line_token} 
	requests.post(url, data=data, headers=headers)


def get_225_per(today):
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
	code_list = df_nk225_sbi['code'].tolist()
	df_list = []
	count = 0

	for code in code_list:
		try:
			dfs = pd.read_html('https://m.finance.yahoo.co.jp/stock/historicaldata?code={}.T'.format(code))
			tmp_df = dfs[0]
			if '日付' in tmp_df.columns:
				tmp_df['日付'] = pd.to_datetime(tmp_df['日付'], format='%Y/%m/%d')
				tmp_df = tmp_df.sort_values('日付')
				tmp_df['SMA10'] = ta.SMA(tmp_df['終値'], timeperiod=10)
				tmp_df = tmp_df.dropna()
				if len(tmp_df[tmp_df['日付']==today]) == 0:
					count+=1
				df_list.append(tmp_df)
		except Exception:
			pass
		if count >=5:
			return None

	n225_df = pd.concat(df_list)
	n225_df_today = n225_df[n225_df['日付']==today]

	if len(n225_df_today) < 200:
		return None

	per = len(n225_df_today[n225_df_today['終値']>=n225_df_today['SMA10']])/ len(n225_df_today)
	return per


#%%
def iniMT5():
	if not mt5.initialize():
		print_log("initialize() failed, error code =",mt5.last_error())
		return False

	authorized=mt5.login(int(settings.number), password=settings.password,server=settings.server)
	if authorized:
		print_log("connected to account #{}".format(settings.number))
		return True
	else:
		print_log("failed to connect at account #{}, error code: {}".format(settings.number, mt5.last_error()))
		return False


def post_market_order(symbol, type, vol, price, dev, sl=None, tp=None, position=None):
	""" 注文を送信
	"""
	request = {
		'action': mt5.TRADE_ACTION_DEAL,
		'symbol': symbol,
		'volume': vol,
		'price': price,
		'deviation': dev,   # float型じゃだめ
		'magic': MAGIC,
		'comment': "python script open",    # 何でもOK
		'type_time': mt5.ORDER_TIME_GTC,
		'type': type,
		'type_filling': mt5.ORDER_FILLING_IOC, # ブローカーにより異なる
	}
	if sl is not None:
		request.update({"sl": sl,})
	if tp is not None:
		request.update({"tp": tp,})
	if position is not None:
		request.update({"position": position})

	result = mt5.order_send(request)
	return result

def get_position():
	# ポジションを確認する
	position = mt5.positions_get()
	if position is None:
		print_log("ポジションが存在しない")
		return None

	pos = position[0]
	if pos.magic != MAGIC:
		print_log("ポジションが存在しない")
		return None
	print_log(f"Open: {pos.price_open}")
	print_log(f"Current: {pos.price_current}")
	print_log(f"Swap: {pos.swap}")
	print_log(f"Profit: {pos.profit}")
	return pos

def send_order():
	# シンボルの情報を取得する
	symbol_info = mt5.symbol_info(SYMBOL)
	if symbol_info is None:
		print_log("Symbol not found")

	# # 注文を出す
	# point = symbol_info.point
	# result = post_market_order(
	# 	SYMBOL, 
	# 	type=mt5.ORDER_TYPE_BUY, 
	# 	vol=0.1, 
	# 	price=mt5.symbol_info_tick(SYMBOL).ask, 
	# 	dev=20, 
	# 	# sl=mt5.symbol_info_tick(SYMBOL).ask - point * 100,
	# 	# tp=mt5.symbol_info_tick(SYMBOL).ask + point * 100,
	# )

	point = mt5.symbol_info(SYMBOL).point
	price = mt5.symbol_info_tick(SYMBOL).ask
	deviation = 20
	request = {
	"action": mt5.TRADE_ACTION_DEAL,
	"symbol": SYMBOL,
	"volume": settings.lot,
	"type": mt5.ORDER_TYPE_BUY,
	"price": price,
	# "sl": price - 100 * point,
	# "tp": price + 100 * point,
	"deviation": deviation,
	"magic": MAGIC,
	"comment": "N225 rate model",
	"type_time": mt5.ORDER_TIME_GTC,
	"type_filling": mt5.ORDER_FILLING_RETURN,
	}

	# 取引リクエストを送信する
	result = mt5.order_send(request)

	# 実行結果を確認する
	code = result.retcode
	if code == 10009:
		print_log("注文完了")
	elif code == 10013:
		print_log("無効なリクエスト")
	elif code == 10018:
		print_log("マーケットが休止中")
	else:
		print_log("その他エラー")
	
	print_log("1. order_send(): by {} {} lots at {} with deviation={} points".format(SYMBOL,lot,price,deviation));
	if result.retcode != mt5.TRADE_RETCODE_DONE:
		print_log("2. order_send failed, retcode={}".format(result.retcode))
		# 結果をディクショナリとしてリクエストし、要素ごとに表示する
		result_dict=result._asdict()
		for field in result_dict.keys():
			print_log("   {}={}".format(field,result_dict[field]))
			# これが取引リクエスト構造体の場合は要素ごとに表示する
			if field=="request":
				traderequest_dict=result_dict[field]._asdict()
				for tradereq_filed in traderequest_dict:
					print_log("       traderequest: {}={}".format(tradereq_filed,traderequest_dict[tradereq_filed]))


def check_position():
	return_list = []
	# ポジションを確認する
	position = mt5.positions_get()
	if position is None:
		print_log("ポジションが存在しない")
		return return_list
	
	for pos in position:
		if pos.magic != MAGIC:
			return None
		print_log(f"Open: {pos.price_open}")
		print_log(f"Current: {pos.price_current}")
		print_log(f"Swap: {pos.swap}")
		print_log(f"Profit: {pos.profit}")
		print_log(f"Ticket: {pos.ticket}")
		return_list.append(pos)
	
	return return_list


def exit_position(position_ticket):
	# 決済する
	# 決済リクエストを作成する
	price=mt5.symbol_info_tick(SYMBOL).bid
	deviation=20
	request={
	"action": mt5.TRADE_ACTION_DEAL,
	"symbol": SYMBOL,
	"volume": settings.lot,
	"type": mt5.ORDER_TYPE_SELL,
	"position": position_ticket,
	"price": price,
	"deviation": deviation,
	"magic": MAGIC,
	"comment": "exit n225 model",
	"type_time": mt5.ORDER_TIME_GTC,
	"type_filling": mt5.ORDER_FILLING_RETURN,
	}
	# 取引リクエストを送信する
	result=mt5.order_send(request)
	# 実行結果を確認する
	print_log("3. close position #{}: sell {} {} lots at {} with deviation={} points".format(position_ticket,SYMBOL,settings.lot,price,deviation));
	if result.retcode != mt5.TRADE_RETCODE_DONE:
		print_log("4. order_send failed, retcode={}".format(result.retcode))
		print_log("   result",result)
	else:
		print_log("4. position #{} closed, {}".format(position_ticket,result))
		# 結果をディクショナリとしてリクエストし、要素ごとに表示する
		result_dict=result._asdict()
		for field in result_dict.keys():
			print_log("   {}={}".format(field,result_dict[field]))
			# これが取引リクエスト構造体の場合は要素ごとに表示する
			if field=="request":
				traderequest_dict=result_dict[field]._asdict()
				for tradereq_filed in traderequest_dict:
					print_log("       traderequest: {}={}".format(tradereq_filed,traderequest_dict[tradereq_filed]))



if __name__ == '__main__':
	os.chdir(os.path.dirname(os.path.abspath(__file__)))

	# lot = float(settings.lot)
	today = datetime.datetime.now().date()
	# today = datetime.datetime.strptime('2022-05-02','%Y-%m-%d')
	print_line('jp225_ratemodel:start:{}'.format(today))

	per = get_225_per(today)
	# per = 0.25
	if per is None:
		print_line('jp225_ratemodel:日経株価の更新が確認できませんでした。')
		sys.exit()
	else:
		print_line('jp225_ratemodel:percent:{}'.format(per))

	if not iniMT5():
		sys.exit()
	
	try:
		if per >= 0.75:
			position_list = check_position()
			if len(position_list) == 0:
				# send_order()
				result = post_market_order(
						SYMBOL, 
						type=mt5.ORDER_TYPE_BUY, 
						vol=LOT, 
						price=mt5.symbol_info_tick(SYMBOL).ask, 
						dev=20, 
						# sl=mt5.symbol_info_tick(SYMBOL).ask - point * 100,
						# tp=mt5.symbol_info_tick(SYMBOL).ask + point * 100,
					)
				print_log(result)
				print_line('jp225_ratemodel:ロングエントリー。{}'.format(result.price))
			else:
				print_line('jp225_ratemodel:すでにポジションを持っているためエントリーしない。')

		elif per <= 0.25:
			position_list = check_position()
			if len(position_list) >= 1:
				for pos in position_list:
					print(pos)
					ticket = pos.ticket
					result = post_market_order(
						SYMBOL, 
						type=mt5.ORDER_TYPE_SELL, 
						vol=LOT, 
						price=mt5.symbol_info_tick(SYMBOL).bid, 
						dev=20, 
						position=ticket,
					)
				print_line('jp225_ratemodel:手じまい。')
				print_line('jp225_ratemodel:open_price:{}'.format(pos.price_open))
				print_line('jp225_ratemodel:close_price:{}'.format(pos.price_current))
				print_line('jp225_ratemodel:利益:{}'.format((pos.price_current-pos.price_open)*LOT))
			else:
				print_line('jp225_ratemodel:ポジションを持っていないためエグジットしない。')
	except Exception as e:
		print_line(e)
	finally:
		mt5.shutdown()

