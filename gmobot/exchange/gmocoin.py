import requests
import json
import hmac
import hashlib
import time
from datetime import datetime

class GmoCoinExchange:

    apiKey = ''
    secretKey = ''
    
    def __init__(self, apiKey, secretKey):
        self.apiKey = apiKey
        self.secretKey = secretKey

    def create_limit_order(self,symbol,side,size,price):

        print("create_limit_order symbol:{0} side:{1} size:{2} price:{3}".format(symbol,side,size,price))

        timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
        method    = 'POST'
        endPoint  = 'https://api.coin.z.com/private'
        path      = '/v1/order'
        reqBody = {
            "symbol": symbol,
            "side": side,
            "executionType": "LIMIT",
            "size": str(size),
            "price": str(price)
        }

        text = timestamp + method + path + json.dumps(reqBody)
        sign = hmac.new(bytes(self.secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()

        headers = {
            "API-KEY": self.apiKey,
            "API-TIMESTAMP": timestamp,
            "API-SIGN": sign
        }

        res = requests.post(endPoint + path, headers=headers, data=json.dumps(reqBody))
        #print (json.dumps(res.json(), indent=2))
        return res.json()

    def create_limit_close_bulk_order(self,symbol,side,size,price):

        print("create_limit_close_bulk_order symbol:{0} side:{1} size:{2} price:{3}".format(symbol,side,size,price))

        timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
        method    = 'POST'
        endPoint  = 'https://api.coin.z.com/private'
        path      = '/v1/closeBulkOrder'
        reqBody = {
            "symbol": symbol,
            "side": side,
            "executionType": "LIMIT",
            "size": str(size),
            "price": str(price)
        }

        text = timestamp + method + path + json.dumps(reqBody)
        sign = hmac.new(bytes(self.secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()

        headers = {
            "API-KEY": self.apiKey,
            "API-TIMESTAMP": timestamp,
            "API-SIGN": sign
        }

        res = requests.post(endPoint + path, headers=headers, data=json.dumps(reqBody))
        #print (json.dumps(res.json(), indent=2))
        return res.json()

    def create_market_order(self,symbol,side,size):

        print("create_market_order symbol:{0} side:{1} size:{2}".format(symbol,side,size))

        timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
        method    = 'POST'
        endPoint  = 'https://api.coin.z.com/private'
        path      = '/v1/order'
        reqBody = {
            "symbol": symbol,
            "side": side,
            "executionType": "MARKET",
            "size": str(size)
        }

        text = timestamp + method + path + json.dumps(reqBody)
        sign = hmac.new(bytes(self.secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()

        headers = {
            "API-KEY": self.apiKey,
            "API-TIMESTAMP": timestamp,
            "API-SIGN": sign
        }

        res = requests.post(endPoint + path, headers=headers, data=json.dumps(reqBody))
        #print (json.dumps(res.json(), indent=2))
        return res.json()

    def create_market_close_bulk_order(self,symbol,side,size):

        print("create_market_close_bulk_order symbol:{0} side:{1} size:{2}".format(symbol,side,size))

        timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
        method    = 'POST'
        endPoint  = 'https://api.coin.z.com/private'
        path      = '/v1/closeBulkOrder'
        reqBody = {
            "symbol": symbol,
            "side": side,
            "executionType": "MARKET",
            "size": str(size)
        }

        text = timestamp + method + path + json.dumps(reqBody)
        sign = hmac.new(bytes(self.secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()

        headers = {
            "API-KEY": self.apiKey,
            "API-TIMESTAMP": timestamp,
            "API-SIGN": sign
        }

        res = requests.post(endPoint + path, headers=headers, data=json.dumps(reqBody))
        #print (json.dumps(res.json(), indent=2))
        return res.json()


    def create_cancel_all_order(self,symbol):

        print("create_cancel_all_order symbol:{0}".format(symbol))

        timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
        method    = 'POST'
        endPoint  = 'https://api.coin.z.com/private'
        path      = '/v1/cancelBulkOrder'
        reqBody = {
            "symbols": [symbol]
        }

        text = timestamp + method + path + json.dumps(reqBody)
        sign = hmac.new(bytes(self.secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()

        headers = {
            "API-KEY": self.apiKey,
            "API-TIMESTAMP": timestamp,
            "API-SIGN": sign
        }

        res = requests.post(endPoint + path, headers=headers, data=json.dumps(reqBody))
        #print (json.dumps(res.json(), indent=2))

    def get_active_orders(self,symbol):

        print("get_active_orders symbol:{0}".format(symbol))

        timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
        method    = 'GET'
        endPoint  = 'https://api.coin.z.com/private'
        path      = '/v1/activeOrders'

        text = timestamp + method + path
        sign = hmac.new(bytes(self.secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()
        parameters = {
            "symbol": symbol
        }

        headers = {
            "API-KEY": self.apiKey,
            "API-TIMESTAMP": timestamp,
            "API-SIGN": sign
        }

        res = requests.get(endPoint + path, headers=headers, params=parameters)
        # print (json.dumps(res.json(), indent=2))
        return res.json()

    def get_position(self,symbol):

        print("get_position symbol:{0}".format(symbol))

        timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
        method    = 'GET'
        endPoint  = 'https://api.coin.z.com/private'
        path      = '/v1/positionSummary'

        text = timestamp + method + path
        sign = hmac.new(bytes(self.secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()
        parameters = {
            "symbol": symbol
        }

        headers = {
            "API-KEY": self.apiKey,
            "API-TIMESTAMP": timestamp,
            "API-SIGN": sign
        }

        res = requests.get(endPoint + path, headers=headers, params=parameters)
        # print (json.dumps(res.json(), indent=2))

        position = {'buy':0.0, 'sell':0.0}

        if len(res.json()["data"]["list"]) > 0:
            positions = res.json()["data"]["list"]
            for p in positions:
                if p["side"] == "SELL":
                    position["sell"] = float(p['sumPositionQuantity'])
                elif p["side"] == "BUY":
                    position["buy"] =  float(p['sumPositionQuantity'])

        print(position)
        return position


