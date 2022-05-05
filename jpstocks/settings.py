import configparser

inifile = configparser.SafeConfigParser()
inifile.read('settings.ini')

number = inifile.get('XM', 'number')
password = inifile.get('XM', 'password')
server = inifile.get('XM', 'server')

lot = inifile.get('LOT', 'lot')
max_lot = inifile.get('LOT', 'max_lot')

token = inifile.get('LINE', 'token')
