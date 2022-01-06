import ccxt
from datetime import datetime, timedelta
import dateutil.parser
from time import sleep
from logging import getLogger,INFO,FileHandler
import settings

logger = getLogger(__name__)

SYMBOL_BTCFX = 'FX_BTC_JPY'
ITV_SLEEP = 0.001
OHLC_INTERVAL_SEC = 15 * 60

def get_exec_datetime(d):
  exec_date = d["exec_date"].replace('T', ' ')[:-1]
  return dateutil.parser.parse(exec_date) + timedelta(hours=9)

def get_executions(bf, afterId, beforeId, count):
  executions = []
  while True:
    try:
      executions = bf.fetch2(path='executions', api='public', method='GET', params={"product_code": SYMBOL_BTCFX, "after": afterId, "before": beforeId, "count": count})
      break
    except Exception as e:
      print("{}: API call error".format(datetime.now()))
      print(e)
      sleep(1)
  return executions

def calc_next_date(dt: datetime) -> datetime:
   return dt + timedelta(seconds=OHLC_INTERVAL_SEC)

def is_empty_execution(executions_date: datetime, ohlc_date: datetime) -> bool:
   return ohlc_date + timedelta(seconds=OHLC_INTERVAL_SEC) <= executions_date

def fix_ohlc_date(executions_date: datetime, ohlc_date: datetime) -> datetime:
   if not is_empty_execution(executions_date, ohlc_date):
       return ohlc_date
   ohlc_date = ohlc_date + timedelta(seconds=OHLC_INTERVAL_SEC)
   return fix_ohlc_date(executions_date, ohlc_date)

bf = ccxt.bitflyer()
bf.apiKey = settings.apiKey
bf.secret = settings.secret
dateSince = datetime(2018, 9, 5, 8, 0, 0)
# dateSince = datetime(2017, 12, 1, 0, 0, 0)
dateUntil = datetime(2021, 12, 31, 10, 0, 0)
nextDate = calc_next_date(dateSince)

count = 500
afterId = 256645520
beforeId = afterId + count + 1
handler = FileHandler('./bitflyer_fx' + dateSince.strftime("%Y%m%d") + '.csv')
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
print("{}: Program start.".format(datetime.now()))

op, hi, lo, cl, vol = 0.0, 0.0, 0.0, 0.0, 0.0

exec_cnt = 0
loop = True
loop_cnt = 0
while loop:
  # 約定履歴を取得
  exs = get_executions(bf, afterId, beforeId, count)
  # たまに約定履歴がごっそりと無いことがある
  if(len(exs) == 0):
    print("no execs, ID count up: {} - {}, {}".format(afterId, beforeId, count))
    afterId += count
    beforeId += count
    continue
  afterId = exs[0]["id"]
  beforeId = afterId + count + 1
  date = get_exec_datetime(exs[-1])
  datePrev = date

  for ex in reversed(exs):
    date = get_exec_datetime(ex)
    if(dateSince <= date and date <= dateUntil):
      price = ex["price"]
      size = ex["size"]
      if(op == 0.0):
        op, hi, lo, cl, vol = price, price, price, price, size

      # 日付が変わったらファイルハンドラ変更
      if(date.day != datePrev.day):
        print("{}: {} finish, {} data. {} start, 1st id {}".format(datetime.now(), datePrev.strftime("%Y%m%d"), exec_cnt, date.strftime("%Y%m%d"), ex["id"]))
        handler.close()
        logger.removeHandler(handler)
        handler = FileHandler('data/ohlcv/' + date.strftime("%Y%m%d") + '.csv')
        handler.setLevel(INFO)
        logger.setLevel(INFO)
        logger.addHandler(handler)
        exec_cnt = 0

      # 秒が変わったらOHLCVリセット
      # ※ここをいじれば好きな解像度にできるはず、このコードは1秒足でデータ作成(抜けの補完はしてないので注意)
      if nextDate <= date:
        ohlcDate = fix_ohlc_date(date, nextDate)
        logger.info("{ohlcDate},{op},{hi},{lo},{cl},{vol}".format(
          ohlcDate=ohlcDate.strftime("%Y-%m-%d %H:%M:%S"),
          op=int(op),
          hi=int(hi),
          lo=int(lo),
          cl=int(cl),
          vol=vol,
        )
        )
        op, hi, lo, cl, vol = price, price, price, price, size
        nextDate = calc_next_date(ohlcDate)
      price = ex["price"]
      size = ex["size"]
      # ここから下は変更なし


      if(price > hi):
        hi = price
      if(price < lo):
        lo = price
      cl = price
      vol += size
      exec_cnt += 1

    if(date > dateUntil):
      loop = False
      print("{}: Collected all data, next ID {}".format(datetime.now(), ex["id"]))
      break
    datePrev = date

  # print("{}: loop end[{}]".format(loop_cnt+1, datetime.now()))
  # sleep(ITV_SLEEP)
  loop_cnt += 1