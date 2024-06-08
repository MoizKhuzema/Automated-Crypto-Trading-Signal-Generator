import talib
import numpy as np
from binance.client import Client

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


# Replace with your Binance API key
BINANCE_API_KEY = ''
# Replace with your Binance API secret
BINANCE_API_SECRET = ''
# Replace with your OpenAI key
OPENAI_API_KEY = ""
# Replace with OPENAI model
OPENAI_MODEL = 'text-davinci-003'

class CryptoGPT:
    # Class variables
    one_hour_crypto_data = None
    four_hour_crypto_data = None
    three_day_crypto_data = None
    break_out_template = None
    break_down_template = None

    def __init__(self):
        # Create Binance client
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, requests_params={"timeout": 30})
        # crypto data = {
        # coin,
        # historical_price_data = [open_prices, high_prices, low_prices, close_prices, volume],
        # technical_indicator_data
        # }
        CryptoGPT.one_hour_crypto_data = self.fetch_technical_indicator_data(Client.KLINE_INTERVAL_1HOUR) # short term traders
        CryptoGPT.four_hour_crypto_data = self.fetch_technical_indicator_data(Client.KLINE_INTERVAL_4HOUR) # medium term traders
        CryptoGPT.three_day_crypto_data = self.fetch_technical_indicator_data(Client.KLINE_INTERVAL_3DAY) # long term traders
        # Use template for break out strategy
        CryptoGPT.break_out_template = """
You are a crypto analyst assistant. I have provided you the real time data on {coin} crypto coin:

1- Time frame:  {time_frame}
2- High Price: {high_price}
3- Low Price: {low_price}
4- Close Price: {close_price}
5- Volume: {volume}
6- Trading Strategy: Breakout
7- Risk Tolerance: {risk_tolerance} percent of the investment
8- Minimum profit: {minimum_profit}%
9- Exit Strategy: No specific exit strategy
10- Technical indicators:
{indicator_data}

Respond with yes if:
- There is a breakout potential, with expected profit of atleast {minimum_profit} percent. Calculate profit with the formula: percentage_profit = ((expected sell_price - buy_price) / buy_price) * 100
- Probability of making a profit is atleast 75%
- If yes, respond with what price I should buy the coin, how many minutes/hours I should hold and the expected price to sell the coin to make a profit.

Else respond with a single word no.
"""
        # Use template for break down strategy
        CryptoGPT.break_down_template = """z
You are a crypto analyst assistant. I have provided you the real time data on {coin} crypto coin:

1- Time frame:  {time_frame}
2- High Price: {high_price}
3- Low Price: {low_price}
4- Close Price: {close_price}
5- Volume: {volume}
6- Trading Strategy: Breakdown
7- Risk Tolerance: {risk_tolerance} percent of the investment
8- Minimum increase in amount of coins: {minimum_profit}%
9- Exit Strategy: No specific exit strategy
10- Technical indicators:
{indicator_data}

Respond with yes if:
- There is a breakdown potential, with expected increase in amount of coins atleast {minimum_profit}%. Calculate increase with the formula: ((expected buy_price - sell_price) / sell_price) * 100
- Probability of success atleast 75%

Else respond with a single word no.
If yes, tell me at what price I should sell the coin, how many minutes/hours I should hold and the expected price to buy back the coin.
"""
        # Initialize model
        self.llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0.9)

    def fetch_crypto_data(self, symbol, interval):
            # Fetch real-time data from Binance API
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit = 50)
            timestamps = np.array([kline[0] // 1000 for kline in klines], dtype= 'double')  # Convert milliseconds to seconds
            open_prices = np.array([float(kline[1]) for kline in klines], dtype= 'double')
            high_prices = np.array([float(kline[2]) for kline in klines], dtype= 'double')
            low_prices = np.array([float(kline[3]) for kline in klines], dtype= 'double')
            close_prices = np.array([float(kline[4]) for kline in klines], dtype= 'double')
            volumes = np.array([float(kline[5]) for kline in klines], dtype= 'double')
            return timestamps, open_prices, high_prices, low_prices, close_prices, volumes

    def fetch_technical_indicator_data(self, interval):
        # Fetch all trading symbols available on Binance
        exchange_info = self.client.get_exchange_info()
        symbols = [symbol['symbol'] for symbol in exchange_info['symbols']]
        # Set interval
        interval = interval
        output = []

        # Iterate through symbols
        loop_counter = 1
        for symbol in symbols:
            # Get the latest OHLC data
            timestamps, open_prices, high_prices, low_prices, close_prices, volume = self.fetch_crypto_data(symbol, interval)
            
            # Overlap Studies
            upper_band, middle_band, lower_band = talib.BBANDS(close_prices, timeperiod=20, matype=0)
            dema = talib.DEMA(close_prices, timeperiod=30)
            ema = talib.EMA(close_prices, timeperiod=14)
            trendline = talib.HT_TRENDLINE(close_prices)
            kama = talib.KAMA(close_prices, timeperiod=30)
            ma = talib.MA(close_prices, timeperiod=20, matype=0)
            mama, fama = talib.MAMA(close_prices, fastlimit=0.5, slowlimit=0.05)
            mavp = talib.MAVP(close_prices, periods=timestamps, minperiod=2, maxperiod=30, matype=0)
            midpoint = talib.MIDPOINT(close_prices, timeperiod=14)
            midprice = talib.MIDPRICE(high_prices, low_prices, timeperiod=14)
            sar = talib.SAR(high_prices, low_prices, acceleration=0.02, maximum=0.2)
            sarext = talib.SAREXT(high_prices, low_prices, startvalue=0, offsetonreverse=0, accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2, accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2)
            sma = talib.SMA(close_prices, timeperiod=20)
            t3 = talib.T3(close_prices, timeperiod=5, vfactor=0.7)
            tema = talib.TEMA(close_prices, timeperiod=30)
            trima = talib.TRIMA(close_prices, timeperiod=30)
            wma = talib.WMA(close_prices, timeperiod=30)

            # Momentum Indicators
            adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            adxr = talib.ADXR(high_prices, low_prices, close_prices, timeperiod=14)
            apo = talib.APO(close_prices, fastperiod=12, slowperiod=26, matype=0)
            aroon_down, aroon_up = talib.AROON(high_prices, low_prices, timeperiod=14)
            aroonosc = talib.AROONOSC(high_prices, low_prices, timeperiod=14)
            bop = talib.BOP(open_prices, high_prices, low_prices, close_prices)
            cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod=20)
            cmo = talib.CMO(close_prices, timeperiod=14)
            dx = talib.DX(high_prices, low_prices, close_prices, timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            macdext, macd_signal_ext, macd_hist_ext = talib.MACDEXT(close_prices, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
            macdfix, macd_signal_fix, macd_hist_fix = talib.MACDFIX(close_prices, signalperiod=9)
            mfi = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14)
            minus_di = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
            minus_dm = talib.MINUS_DM(high_prices, low_prices, timeperiod=14)
            mom = talib.MOM(close_prices, timeperiod=10)
            plus_di = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
            plus_dm = talib.PLUS_DM(high_prices, low_prices, timeperiod=14)
            ppo = talib.PPO(close_prices, fastperiod=12, slowperiod=26, matype=0)
            roc = talib.ROC(close_prices, timeperiod=10)
            rocp = talib.ROCP(close_prices, timeperiod=10)
            rocr = talib.ROCR(close_prices, timeperiod=10)
            rocr100 = talib.ROCR100(close_prices, timeperiod=10)
            rsi = talib.RSI(close_prices, timeperiod=14)
            slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            fastk, fastd = talib.STOCHF(high_prices, low_prices, close_prices, fastk_period=5, fastd_period=3, fastd_matype=0)
            fastk_rsi, fastd_rsi = talib.STOCHRSI(close_prices, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
            trix = talib.TRIX(close_prices, timeperiod=30)
            ultosc = talib.ULTOSC(high_prices, low_prices, close_prices, timeperiod1=7, timeperiod2=14, timeperiod3=28)
            willr = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)

            # Volume Indicators
            ad = talib.AD(high_prices, low_prices, close_prices, volume)
            adosc = talib.ADOSC(high_prices, low_prices, close_prices, volume, fastperiod=3, slowperiod=10)
            obv = talib.OBV(close_prices, volume)

            # Cycle Indicators
            ht_dcperiod = talib.HT_DCPERIOD(close_prices)
            ht_dcphase = talib.HT_DCPHASE(close_prices)
            ht_phasor_inphase, ht_phasor_quadrature = talib.HT_PHASOR(close_prices)
            ht_sine, ht_leadsine = talib.HT_SINE(close_prices)
            ht_trendmode = talib.HT_TRENDMODE(close_prices)

            # Price Transform
            avgprice = talib.AVGPRICE(open_prices, high_prices, low_prices, close_prices)
            medprice = talib.MEDPRICE(high_prices, low_prices)
            typprice = talib.TYPPRICE(high_prices, low_prices, close_prices)
            wclprice = talib.WCLPRICE(high_prices, low_prices, close_prices)

            # Volatility Indicators
            atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            natr = talib.NATR(high_prices, low_prices, close_prices, timeperiod=14)
            trange = talib.TRANGE(high_prices, low_prices, close_prices)

            # Pattern Recognition
            cdl_2crows = talib.CDL2CROWS(open_prices, high_prices, low_prices, close_prices)
            cdl_3blackcrows = talib.CDL3BLACKCROWS(open_prices, high_prices, low_prices, close_prices)
            cdl_3inside = talib.CDL3INSIDE(open_prices, high_prices, low_prices, close_prices)
            cdl_3linestrike = talib.CDL3LINESTRIKE(open_prices, high_prices, low_prices, close_prices)
            cdl_3outside = talib.CDL3OUTSIDE(open_prices, high_prices, low_prices, close_prices)
            cdl_3starsinsouth = talib.CDL3STARSINSOUTH(open_prices, high_prices, low_prices, close_prices)
            cdl_3whitesoldiers = talib.CDL3WHITESOLDIERS(open_prices, high_prices, low_prices, close_prices)
            cdl_abandonedbaby = talib.CDLABANDONEDBABY(open_prices, high_prices, low_prices, close_prices)
            cdl_advanceblock = talib.CDLADVANCEBLOCK(open_prices, high_prices, low_prices, close_prices)
            cdl_belthold = talib.CDLBELTHOLD(open_prices, high_prices, low_prices, close_prices)
            cdl_breakaway = talib.CDLBREAKAWAY(open_prices, high_prices, low_prices, close_prices)
            cdl_closingmarubozu = talib.CDLCLOSINGMARUBOZU(open_prices, high_prices, low_prices, close_prices)
            cdl_concealbabyswall = talib.CDLCONCEALBABYSWALL(open_prices, high_prices, low_prices, close_prices)
            cdl_counterattack = talib.CDLCOUNTERATTACK(open_prices, high_prices, low_prices, close_prices)
            cdl_darkcloudcover = talib.CDLDARKCLOUDCOVER(open_prices, high_prices, low_prices, close_prices)
            cdl_doji = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
            cdl_dojistar = talib.CDLDOJISTAR(open_prices, high_prices, low_prices, close_prices)
            cdl_dragonflydoji = talib.CDLDRAGONFLYDOJI(open_prices, high_prices, low_prices, close_prices)
            cdl_engulfing = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            cdl_eveningdojistar = talib.CDLEVENINGDOJISTAR(open_prices, high_prices, low_prices, close_prices)
            cdl_eveningstar = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
            cdl_gapsidesidewhite = talib.CDLGAPSIDESIDEWHITE(open_prices, high_prices, low_prices, close_prices)
            cdl_gravestonedoji = talib.CDLGRAVESTONEDOJI(open_prices, high_prices, low_prices, close_prices)
            cdl_hammer = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
            cdl_hangingman = talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)
            cdl_harami = talib.CDLHARAMI(open_prices, high_prices, low_prices, close_prices)
            cdl_haramicross = talib.CDLHARAMICROSS(open_prices, high_prices, low_prices, close_prices)
            cdl_highwave = talib.CDLHIGHWAVE(open_prices, high_prices, low_prices, close_prices)
            cdl_hikkake = talib.CDLHIKKAKE(open_prices, high_prices, low_prices, close_prices)
            cdl_hikkakemod = talib.CDLHIKKAKEMOD(open_prices, high_prices, low_prices, close_prices)
            cdl_homingpigeon = talib.CDLHOMINGPIGEON(open_prices, high_prices, low_prices, close_prices)
            cdl_identical3crows = talib.CDLIDENTICAL3CROWS(open_prices, high_prices, low_prices, close_prices)
            cdl_inneck = talib.CDLINNECK(open_prices, high_prices, low_prices, close_prices)
            cdl_invertedhammer = talib.CDLINVERTEDHAMMER(open_prices, high_prices, low_prices, close_prices)
            cdl_kicking = talib.CDLKICKING(open_prices, high_prices, low_prices, close_prices)
            cdl_kickingbylength = talib.CDLKICKINGBYLENGTH(open_prices, high_prices, low_prices, close_prices)
            cdl_ladderbottom = talib.CDLLADDERBOTTOM(open_prices, high_prices, low_prices, close_prices)
            cdl_longleggeddoji = talib.CDLLONGLEGGEDDOJI(open_prices, high_prices, low_prices, close_prices)
            cdl_longline = talib.CDLLONGLINE(open_prices, high_prices, low_prices, close_prices)
            cdl_marubozu = talib.CDLMARUBOZU(open_prices, high_prices, low_prices, close_prices)
            cdl_matchinglow = talib.CDLMATCHINGLOW(open_prices, high_prices, low_prices, close_prices)
            cdl_mathold = talib.CDLMATHOLD(open_prices, high_prices, low_prices, close_prices)
            cdl_morningdojistar = talib.CDLMORNINGDOJISTAR(open_prices, high_prices, low_prices, close_prices)
            cdl_onneck = talib.CDLONNECK(open_prices, high_prices, low_prices, close_prices)
            cdl_piercing = talib.CDLPIERCING(open_prices, high_prices, low_prices, close_prices)
            cdl_rickshawman = talib.CDLRICKSHAWMAN(open_prices, high_prices, low_prices, close_prices)
            cdl_risefall3methods = talib.CDLRISEFALL3METHODS(open_prices, high_prices, low_prices, close_prices)
            cdl_separatinglines = talib.CDLSEPARATINGLINES(open_prices, high_prices, low_prices, close_prices)
            cdl_shootingstar = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)
            cdl_shortline = talib.CDLSHORTLINE(open_prices, high_prices, low_prices, close_prices)
            cdl_spinningtop = talib.CDLSPINNINGTOP(open_prices, high_prices, low_prices, close_prices)
            cdl_stalledpattern = talib.CDLSTALLEDPATTERN(open_prices, high_prices, low_prices, close_prices)
            cdl_sticksandwich = talib.CDLSTICKSANDWICH(open_prices, high_prices, low_prices, close_prices)
            cdl_takuri = talib.CDLTAKURI(open_prices, high_prices, low_prices, close_prices)
            cdl_tasukigap = talib.CDLTASUKIGAP(open_prices, high_prices, low_prices, close_prices)
            cdl_thrusting = talib.CDLTHRUSTING(open_prices, high_prices, low_prices, close_prices)
            cdl_tristar = talib.CDLTRISTAR(open_prices, high_prices, low_prices, close_prices)
            cdl_unique3river = talib.CDLUNIQUE3RIVER(open_prices, high_prices, low_prices, close_prices)
            cdl_upsidegap2crows = talib.CDLUPSIDEGAP2CROWS(open_prices, high_prices, low_prices, close_prices)
            cdl_xsidegap3methods = talib.CDLXSIDEGAP3METHODS(open_prices, high_prices, low_prices, close_prices)

            print(f'fetched data of {loop_counter} /  {len(symbols)} coins')
            loop_counter += 1

            indicator_data = {
                'coin': symbol,
                'historical_price_data': [open_prices, high_prices, low_prices, close_prices, volume],
                'technical_indicator_data': 
f"""
'Upper Band': {upper_band[-1]},
'Middle Band': {middle_band[-1]},
'Lower Band': {lower_band[-1]},
'DEMA': {dema[-1]},
'EMA': {ema[-1]},
'HT Trendline': {trendline[-1]},
'KAMA': {kama[-1]},
'MA': {ma[-1]},
'MAMA': {mama[-1]},
'FAMA': {fama[-1]},
'MAVP': {mavp[-1]},
'Midpoint': {midpoint[-1]},
'Midprice': {midprice[-1]},
'SAR': {sar[-1]},
'SAREXT': {sarext[-1]},
'SMA': {sma[-1]},
'T3': {t3[-1]},
'TEMA': {tema[-1]},
'TRIMA': {trima[-1]},
'WMA': {wma[-1]},
'ADX': {adx[-1]},
'ADXR': {adxr[-1]},
'APO': {apo[-1]},
'Aroon Down': {aroon_down[-1]},
'Aroon Up': {aroon_up[-1]},
'Aroon Oscillator': {aroonosc[-1]},
'BOP': {bop[-1]},
'CCI': {cci[-1]},
'CMO': {cmo[-1]},
'DX': {dx[-1]},
'MACD': {macd[-1]},
'MACD Signal': {macd_signal[-1]},
'MACD Histogram': {macd_hist[-1]},
'MACDEXT': {macdext[-1]},
'MACD Signal Ext': {macd_signal_ext[-1]},
'MACD Hist Ext': {macd_hist_ext[-1]},
'MACDFIX': {macdfix[-1]},
'MACD Signal Fix': {macd_signal_fix[-1]},
'MACD Hist Fix': {macd_hist_fix[-1]},
'MFI': {mfi[-1]},
'Minus DI': {minus_di[-1]},
'Minus DM': {minus_dm[-1]},
'Momentum': {mom[-1]},
'Plus DI': {plus_di[-1]},
'Plus DM': {plus_dm[-1]},
'PPO': {ppo[-1]},
'ROC': {roc[-1]},
'ROCP': {rocp[-1]},
'ROCR': {rocr[-1]},
'ROCR100': {rocr100[-1]},
'RSI': {rsi[-1]},
'SlowK': {slowk[-1]},
'SlowD': {slowd[-1]},
'FastK': {fastk[-1]},
'FastD': {fastd[-1]},
'FastK RSI': {fastk_rsi[-1]},
'FastD RSI': {fastd_rsi[-1]},
'TRIX': {trix[-1]},
'ULTOSC': {ultosc[-1]},
'WILLR': {willr[-1]},
'AD': {ad[-1]},
'ADOSC': {adosc[-1]},
'OBV': {obv[-1]},
'HT DCPeriod': {ht_dcperiod[-1]},
'HT DCPhase': {ht_dcphase[-1]},
'HT Phasor Inphase': {ht_phasor_inphase[-1]},
'HT Phasor Quadrature': {ht_phasor_quadrature[-1]},
'HT Sine': {ht_sine[-1]},
'HT LeadSine': {ht_leadsine[-1]},
'HT TrendMode': {ht_trendmode[-1]},
'Average Price': {avgprice[-1]},
'Median Price': {medprice[-1]},
'Typical Price': {typprice[-1]},
'Weighted Close Price': {wclprice[-1]},
'ATR': {atr[-1]},
'NATR': {natr[-1]},
'True Range': {trange[-1]},
'CDL 2 Crows': {cdl_2crows[-1]},
'CDL 3 Black Crows': {cdl_3blackcrows[-1]},
'CDL 3 Inside': {cdl_3inside[-1]},
'CDL 3 Line Strike': {cdl_3linestrike[-1]},
'CDL 3 Outside': {cdl_3outside[-1]},
'CDL 3 Stars in the South': {cdl_3starsinsouth[-1]},
'CDL 3 White Soldiers': {cdl_3whitesoldiers[-1]},
'CDL Abandoned Baby': {cdl_abandonedbaby[-1]},
'CDL Advance Block': {cdl_advanceblock[-1]},
'CDL Belt Hold': {cdl_belthold[-1]},
'CDL Breakaway': {cdl_breakaway[-1]},
'CDL Closing Marubozu': {cdl_closingmarubozu[-1]},
'CDL Conceal Baby Swall': {cdl_concealbabyswall[-1]},
'CDL Counterattack': {cdl_counterattack[-1]},
'CDL Dark Cloud Cover': {cdl_darkcloudcover[-1]},
'CDL Doji': {cdl_doji[-1]},
'CDL Doji Star': {cdl_dojistar[-1]},
'CDL Dragonfly Doji': {cdl_dragonflydoji[-1]},
'CDL Engulfing': {cdl_engulfing[-1]},
'CDL Evening Doji Star': {cdl_eveningdojistar[-1]},
'CDL Evening Star': {cdl_eveningstar[-1]},
'CDL Gap Side Side White': {cdl_gapsidesidewhite[-1]},
'CDL Gravestone Doji': {cdl_gravestonedoji[-1]},
'CDL Hammer': {cdl_hammer[-1]},
'CDL Hanging Man': {cdl_hangingman[-1]},
'CDL Harami': {cdl_harami[-1]},
'CDL Harami Cross': {cdl_haramicross[-1]},
'CDL High Wave': {cdl_highwave[-1]},
'CDL Hikkake': {cdl_hikkake[-1]},
'CDL Hikkake Mod': {cdl_hikkakemod[-1]},
'CDL Homing Pigeon': {cdl_homingpigeon[-1]},
'CDL Identical 3 Crows': {cdl_identical3crows[-1]},
'CDL In Neck': {cdl_inneck[-1]},
'CDL Inverted Hammer': {cdl_invertedhammer[-1]},
'CDL Kicking': {cdl_kicking[-1]},
'CDL Kicking By Length': {cdl_kickingbylength[-1]},
'CDL Ladder Bottom': {cdl_ladderbottom[-1]},
'CDL Long Legged Doji': {cdl_longleggeddoji[-1]},
'CDL Long Line': {cdl_longline[-1]},
'CDL Marubozu': {cdl_marubozu[-1]},
'CDL Matching Low': {cdl_matchinglow[-1]},
'CDL Mat Hold': {cdl_mathold[-1]},
'CDL Morning Doji Star': {cdl_morningdojistar[-1]},
'CDL On Neck': {cdl_onneck[-1]},
'CDL Piercing': {cdl_piercing[-1]},
'CDL Rickshaw Man': {cdl_rickshawman[-1]},
'CDL Rise Fall 3 Methods': {cdl_risefall3methods[-1]},
'CDL Separating Lines': {cdl_separatinglines[-1]},
'CDL Shooting Star': {cdl_shootingstar[-1]},
'CDL Short Line': {cdl_shortline[-1]},
'CDL Spinning Top': {cdl_spinningtop[-1]},
'CDL Stalled Pattern': {cdl_stalledpattern[-1]},
'CDL Stick Sandwich': {cdl_sticksandwich[-1]},
'CDL Takuri': {cdl_takuri[-1]},
'CDL Tasuki Gap': {cdl_tasukigap[-1]},
'CDL Thrusting': {cdl_thrusting[-1]},
'CDL Tristar': {cdl_tristar[-1]},
'CDL Unique 3 River': {cdl_unique3river[-1]},
'CDL Upside Gap 2 Crows': {cdl_upsidegap2crows[-1]},
'CDL X Side Gap 3 Methods': {cdl_xsidegap3methods[-1]}
"""
            }

            output.append(indicator_data)
        return  output
    
    def predict(self, template, coin, time_frame, high_price, low_price, close_price, volume, risk_tolerance, minimum_profit, indicator_data):
        # generate prompt (input)
        prompt = PromptTemplate(
            input_variables=["coin", "time_frame", "high_price", "low_price", "close_price", "volume", "risk_tolerance", "minimum_profit", "indicator_data"],
            template=template
        )
        # generate llm chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        # Run the chain only specifying the input variable.
        result = chain.run({
            'coin': coin,
            'time_frame': time_frame,
            'high_price': high_price,
            'low_price':  low_price,
            'close_price': close_price,
            'volume': volume,
            'risk_tolerance': risk_tolerance,
            'minimum_profit': minimum_profit,
            'indicator_data': indicator_data
            })
        
        return result
    
        
def main():
    model = CryptoGPT()
    main_exit_flag = True
    predict_exit_flag = True
    #  short term = 0
    #  medium term = 1
    #  long term = 2
    time_frame_flag = None
    # break out trading = 0
    # break down trading = 1
    trading_stat_flag = None
    crypto_data = None
    template = None
    time_frame = None
    risk_tolerance = None
    minimum_profit = None


    while main_exit_flag:
        time_frame_flag = int(input("""
Enter time frame
1- short term
2- medium term
3- long term
Answer:   """))
        trading_stat_flag = int(input("""
Enter trading strategy,
1- breakout
2- breakdown
Answer:   """))       
        risk_tolerance = int(input("Enter the risk tolerance (5-20 %):   "))

        if time_frame_flag == 1:
            crypto_data = model.one_hour_crypto_data
            time_frame = '1 hour'
            minimum_profit = int(input("Enter the minimum profit between 5 to 20 %:   "))
        elif time_frame_flag == 2:
            crypto_data = model.four_hour_crypto_data
            time_frame = '4 hours'
            minimum_profit = int(input("Enter the minimum profit between 10 to 50 %:   "))
        elif time_frame_flag == 3:
            crypto_data = model.three_day_crypto_data
            time_frame = '3 days'
            minimum_profit = int(input("Enter the minimum profit between 20 to 200 %:   "))

        if trading_stat_flag == 1:
            template = model.break_out_template
        elif trading_stat_flag == 2:
            template = model.break_down_template

        loop_counter = 0
        while predict_exit_flag:
            output = model.predict(template, 
                                       crypto_data[loop_counter]['coin'],
                                       time_frame,
                                       crypto_data[loop_counter]['historical_price_data'][1],
                                       crypto_data[loop_counter]['historical_price_data'][2],
                                       crypto_data[loop_counter]['historical_price_data'][3],
                                       crypto_data[loop_counter]['historical_price_data'][4],
                                       risk_tolerance,
                                       minimum_profit,
                                       crypto_data[loop_counter]['technical_indicator_data']
                                       )
            
            if 'no' not in output.lower():
                predict_exit_flag = False
            
            loop_counter += 1
            if loop_counter > len(crypto_data):
                predict_exit_flag = False
        
        print(f'Coin indentified: {crypto_data[loop_counter]["coin"]}')
        print(output)
        print('-------------------------------------------------------')
        main_exit_flag = int(input('exit = 0, try again = 1:   '))
        predict_exit_flag = True


# driver code
main()