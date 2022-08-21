import configparser

inifile = configparser.SafeConfigParser()
inifile.read('settings.ini')

lot = inifile.get('LOT', 'lot')
max_lot = inifile.get('LOT', 'max_lot')
interval = inifile.get('LOT', 'interval')

token = inifile.get('LINE', 'token')
