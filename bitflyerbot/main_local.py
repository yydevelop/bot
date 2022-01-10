import os
from os.path import join, dirname
from dotenv import load_dotenv
import main
import pybitflyer

if __name__  == '__main__':

   dotenv_path = join(dirname(__file__), './env/.env-bffx')
   load_dotenv(dotenv_path,verbose=True)

   exchangeName = os.getenv("EXCHANGE")
   apiKey    = os.getenv("API_KEY")
   secretKey = os.getenv("SECRET_KEY")
   lot = float(os.getenv("LOT"))
   max_lot = float(os.getenv("MAX_LOT"))
   interval = int(os.getenv("INTERVAL"))

   exchange = pybitflyer.API(api_key=apiKey, api_secret=secretKey)
   main.start(exchange, max_lot, lot, interval)