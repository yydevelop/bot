import os
from os.path import join, dirname
from dotenv import load_dotenv
import main
from exchange.gmocoin import GmoCoinExchange

if __name__  == '__main__':
   
   load_dotenv(verbose=True)

   apiKey    = os.getenv("API_KEY")
   secretKey = os.getenv("SECRET_KEY")
   lot = float(os.getenv("LOT"))
   max_lot = float(os.getenv("MAX_LOT"))
   interval = int(os.getenv("INTERVAL"))
   exchange = GmoCoinExchange(apiKey,secretKey)
   main.start(exchange, max_lot, lot, interval)