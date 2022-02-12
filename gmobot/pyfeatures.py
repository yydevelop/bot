import talib
import numpy as np

features = sorted([
#     'ADX',
#     'ADXR',
#     'APO',
#     'AROON_aroondown',
#     'AROON_aroonup',
#     'AROONOSC',
#     'CCI',
#     'DX',
#     'MACD_macd',
#     'MACD_macdsignal',
#     'MACD_macdhist',
#     'MFI',
# #     'MINUS_DI',
# #     'MINUS_DM',
#     'MOM',
# #     'PLUS_DI',
# #     'PLUS_DM',
#     'RSI',
#     'STOCH_slowk',
#     'STOCH_slowd',
#     'STOCHF_fastk',
# #     'STOCHRSI_fastd',
#     'ULTOSC',
#     'WILLR',
# #     'ADOSC',
# #     'NATR',
#     'HT_DCPERIOD',
#     'HT_DCPHASE',
#     'HT_PHASOR_inphase',
#     'HT_PHASOR_quadrature',
#     'HT_TRENDMODE',
#     'BETA',
#     'LINEARREG',
#     'LINEARREG_ANGLE',
#     'LINEARREG_INTERCEPT',
#     'LINEARREG_SLOPE',
#     'STDDEV',
#     'BBANDS_upperband',
#     'BBANDS_middleband',
#     'BBANDS_lowerband',
#     'DEMA',
#     'EMA',
#     'HT_TRENDLINE',
#     'KAMA',
#     'MA',
#     'MIDPOINT',
#     'T3',
#     'TEMA',
#     'TRIMA',
#     'WMA',
    # 'MACD_macd',
    # 'RSI',
    # 'ATR',
    # 'volume',
    'VOL_3',
    'RETURN_3',
    'MA_GAP_3',
    'VOL_5',
    'RETURN_5',
    'MA_GAP_5',
    # 'VOL_21',
    # 'RETURN_21',
    # 'MA_GAP_21',
    # 'VOL_55',
    # 'RETURN_55',
    # 'MA_GAP_55',
    'PER_OPCL',
    'PER_HILO',
    'CL_LOG',
])


def calc_features(df):
    open = df['op']
    high = df['hi']
    low = df['lo']
    close = df['cl']
    volume = df['volume']
    
    orig_columns = df.columns

    hilo = (df['hi'] + df['lo']) / 2
    df['BBANDS_upperband'], df['BBANDS_middleband'], df['BBANDS_lowerband'] = talib.BBANDS(close, timeperiod=20, nbdevup=3, nbdevdn=3, matype=0)
    # df['BBANDS_upperband'] -= hilo
    # df['BBANDS_middleband'] -= hilo
    # df['BBANDS_lowerband'] -= hilo
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

    df['ATR'] = talib.ATR(high, low, close, timeperiod=3)
    df['NATR'] = talib.NATR(high, low, close, timeperiod=3)
    df['ATR14'] = talib.ATR(high, low, close, timeperiod=14)
    df['NATR14'] = talib.NATR(high, low, close, timeperiod=14)
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

    periods = [3,5,8,13,21,55,89]
    for period in periods:
        col = 'RETURN_{}'.format(period)
        df[col] = \
            df['cl'].pct_change(period)

        if period !=1:
            col = 'VOL_{}'.format(period)
            df[col] = \
                np.log(df['cl']).diff().rolling(period).std()
            
        col = 'MA_GAP_{}'.format(period)
        df[col] = \
            df['cl'] / (df['cl'].rolling(period).mean())
    
    df['hilo'] = df['hi'] - df['lo']
    df['PER_HILO'] = df['hilo'] / (hilo+1)
    df['opcl'] = df['op'] - df['cl']
    df['PER_OPCL'] = df['opcl'] / (hilo+1)
    df['CL_LOG'] = np.log(df['cl']).diff()

    df['SMA_5H'] = talib.SMA(close, timeperiod=12*5)
    df['SMA_11H'] = talib.SMA(close, timeperiod=12*11)
    df['SMA_22H'] = talib.SMA(close, timeperiod=12*22)
    df['SMA_SHORT'] = talib.SMA(close, timeperiod=12*5)
    df['SMA_MID'] = talib.SMA(close, timeperiod=12*11)
    df['SMA_LONG'] = talib.SMA(close, timeperiod=12*22)
    # df['LOG_OPCL'] = df['opcl'].apply(np.log)
    df['RSI11'] = talib.RSI(close, timeperiod=12*11)
    df['RSI22'] = talib.RSI(close, timeperiod=12*22)
    df['RSI55'] = talib.RSI(close, timeperiod=12*55)
    df['RSI11H'] = talib.RSI(close, timeperiod=12*11)
    df['RSI22H'] = talib.RSI(close, timeperiod=12*22)
    df['RSI55H'] = talib.RSI(close, timeperiod=12*55)

    return df