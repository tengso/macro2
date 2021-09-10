import pandas as pd
import datetime
import functools as ft


# J: April
# K: May
# M: June
# N: July
# Q: August
# U: Sept

symbol_map = {
    3: 'H',
    4: 'J',
    5: 'K',
    6: 'M',
    7: 'N',
    8: 'Q',
    9: 'U',
}


@ft.lru_cache(maxsize=None)
def get_spot_price(month):
    columns = [
        'open_time',
        'open',
        'high',
        'low',
        'close',
        'volume',
        'close_time',
        'quote_asset_volume',
        'number_of_trades',
        'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume',
        'ignore',
    ]

    spot_path = fr'/Users/song/projects/PycharmProjects/pythonProject/macro/data/BTCUSDT-1m-2021-{month:02}.csv'

    btc_spot = pd.read_csv(spot_path, names=columns)
    btc_spot['open_time'] = btc_spot.open_time.apply(lambda t: datetime.datetime.utcfromtimestamp(t / 1000))
    btc_spot['close_time'] = btc_spot.close_time.apply(lambda t: datetime.datetime.utcfromtimestamp(t / 1000))
    btc_spot = btc_spot.set_index('open_time')
    btc_spot.index = btc_spot.index.tz_localize('UTC')
    btc_spot.index = btc_spot.index.tz_convert('America/New_York')
    btc_spot = btc_spot.close

    return btc_spot


def get_discount_band(discount, window_size, band_size):
    discount = discount.rolling(window_size)
    discount_mean = discount.mean()
    discount_std = discount.std()
    discount_upper = discount_mean + band_size * discount_std
    discount_lower = discount_mean - band_size * discount_std
    discount_band = pd.DataFrame(dict(mv=discount_mean, upper=discount_upper, lower=discount_lower, discount=discount))
    return discount_band


@ft.lru_cache(maxsize=None)
def get_official_daily_discount():
    # daily_premium
    p = r'/Users/song/projects/PycharmProjects/pythonProject/macro/data/grayscale-premium.csv'
    discount = pd.read_csv(p)
    # discount.head()
    discount['timestamp'] = pd.to_datetime(discount['timestamp'])
    discount['date'] = discount.timestamp.apply(lambda t: t.date())
    # discount.head()
    discount = discount.set_index('date').value
    return discount


@ft.lru_cache(maxsize=None)
def get_gbtc_intraday_price():
    path = r'/Users/song/projects/PycharmProjects/pythonProject/macro/data/intraday.xlsx'

    gbtc = pd.read_excel(path, sheet_name='GBTC 1M')

    gbtc['datetime'] = pd.to_datetime(gbtc.Dates)
    gbtc = gbtc.set_index('datetime', drop=False)
    gbtc.index = gbtc.index.tz_localize('Asia/Hong_Kong').tz_convert('America/New_York')
    gbtc['datetime'] = gbtc.index
    gbtc['date'] = gbtc.datetime.apply(lambda d: d.date())
    gbtc['time'] = gbtc.datetime.apply(lambda d: d.time())

    gbtc = gbtc[['Close', 'Volume', 'datetime', 'date', 'time']]
    gbtc.columns = ['close', 'volume', 'datetime', 'date', 'time']

    return gbtc


@ft.lru_cache(maxsize=None)
def get_future_intraday_price(month):
    # month = 3
    path = r'/Users/song/projects/PycharmProjects/pythonProject/macro/data/intraday.xlsx'

    code = symbol_map[month]
    btc = pd.read_excel(path, sheet_name=f'BTC{code}1 1M')
    # btc.head()
    btc['datetime'] = pd.to_datetime(btc.Dates)
    btc = btc.set_index('Dates', drop=False)
    btc.index = btc.index.tz_localize('Asia/Hong_Kong').tz_convert('America/New_York')
    btc['datetime'] = btc.index
    btc['date'] = btc.datetime.apply(lambda d: d.date())
    btc['time'] = btc.datetime.apply(lambda d: d.time())

    btc = btc[['Close', 'Volume', 'datetime', 'date', 'time']]
    btc.columns = ['close', 'volume', 'datetime', 'date', 'time']

    # btc.head()
    # btc[btc.date == datetime.date(2021, 3, 7)]
    return btc


def compute_daily_price(prices, start_time=datetime.time(9, 30), end_time=datetime.time(16)):
    # prices = get_future_intraday_price(3)
    # start_time = datetime.time(9, 30)
    # end_time = datetime.time(16)
    # prices['2021-03-03 09:30:00':].head()
    # prices['2021-03-01 09:30:00':].head()
    # snap_time = datetime.time(9, 30)
    prices = prices[(prices.time <= end_time) & (prices.time >= start_time)]
    volume = prices.groupby('date').volume.sum()
    open_price = prices.groupby('date').first().close
    close_price = prices.groupby('date').last().close
    daily = pd.DataFrame(dict(open=open_price, close=close_price, volume=volume))
    return daily


def compute_intraday_discount(start_date, end_date):
    result = []
    for asof in pd.date_range(start_date, end_date):
        print(asof)
        # asof = datetime.date(2021, 7, 15)
        start_time = datetime.datetime.combine(asof, datetime.time(9, 30))
        end_time = datetime.datetime.combine(asof, datetime.time(16, 0))

        future = get_future_intraday_price(asof.month)
        if not len(future):
            continue
        future = future[start_time:end_time]
        # future.head()

        gbtc = get_gbtc_intraday_price()
        if not len(gbtc):
            continue
        gbtc = gbtc[start_time:end_time]
        # gbtc.head()

        combined = pd.DataFrame(dict(gbtc=gbtc.close, future=future.close))
        # combined.head()

        btc_per_share = 0.000937966
        # btc_per_share = 0.0010

        combined['gbtc_fv'] = btc_per_share * combined.future
        combined['discount'] = (combined.gbtc - combined.gbtc_fv) / combined.gbtc_fv

        result.append(combined)

    if len(result):
        return pd.concat(result)
    else:
        return result


def compute_daily_discount(intraday_discount):
    discount = intraday_discount.copy()
    discount['datetime'] = discount.index
    discount['date'] = discount.datetime.apply(lambda dt: dt.date())

    daily_last = discount.groupby('date').discount.last()
    daily_mean = discount.groupby('date').discount.mean()
    daily_first = discount.groupby('date').discount.first()
    daily_min = discount.groupby('date').discount.min()
    daily_max = discount.groupby('date').discount.max()
    daily_std = discount.groupby('date').discount.std()

    daily = pd.DataFrame(dict(last=daily_last, mean=daily_mean, first=daily_first, min=daily_min, max=daily_max, std=daily_std))

    official_discount = get_official_daily_discount()
    daily['official'] = official_discount
    daily = daily.dropna()

    return daily


def compute_continuous_futures_pnl(month_range):
    data = {}
    for month in month_range:
        # month = 3
        prices = get_future_intraday_price(month)
        prices = compute_daily_price(prices)

        data[f'{month}_close'] = prices['close']
        data[f'{month}_volume'] = prices['volume']

    data = pd.DataFrame(data)
    data = data.fillna(0)

    # month_range = [3, 4, 5, 6, 7, 8]

    result_ret = {}
    result_pnl = {}
    for i, asof in enumerate(data.index):
        if i == 0:
            continue
        row = data.loc[asof]
        last_row = data.loc[data.index[i - 1]]
        volumes = [row[f'{month}_volume'] for month in month_range]
        max_volumes = max(volumes)
        month = month_range[volumes.index(max_volumes)]
        last_price = last_row[f'{month}_close']
        price = row[f'{month}_close']
        ret = (price - last_price) / last_price
        pnl = price - last_price
        # print(asof, volumes, max_volumes, month_range[month])
        # print(asof, month, pnl)
        result_ret[asof] = ret
        result_pnl[asof] = pnl

    return pd.Series(result_ret).sort_index(), pd.Series(result_pnl).sort_index()

