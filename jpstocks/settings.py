import configparser

inifile = configparser.SafeConfigParser()
inifile.read('settings.ini')

number = inifile.get('OANDA', 'number')
password = inifile.get('OANDA', 'password')
server = inifile.get('OANDA', 'server')

lot = inifile.get('LOT', 'lot')
max_lot = inifile.get('LOT', 'max_lot')

token = inifile.get('LINE', 'token')
