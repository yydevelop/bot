import configparser

inifile = configparser.SafeConfigParser()
inifile.read('settings.ini')

apiKey = inifile.get('BITFLYER', 'apiKey')
secret = inifile.get('BITFLYER', 'secret')

gmoApiKey = inifile.get('GMO', 'gmoApiKey')
gmoSecret = inifile.get('GMO', 'gmoSecret')

bybitApiKey = inifile.get('BYBIT', 'bybitApiKey')
bybitSecret = inifile.get('BYBIT', 'bybitSecret')

lot = inifile.get('LOT', 'lot')
max_lot = inifile.get('LOT', 'max_lot')
interval = inifile.get('LOT', 'interval')

token = inifile.get('LINE', 'token')
