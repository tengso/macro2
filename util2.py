import pandas as pd
import pyfolio as pf


# p = r'C:\Users\steng\PycharmProjects\pythonProject\macro\data\prices.xlsx'
p = r'/Users/song/projects/PycharmProjects/pythonProject/macro/data/prices.xlsx'
# p = '/Users/song/projects/PycharmProjects/pythonProject/macro/data/prices.xlsx'

# 代码	名称	日期	开盘价(元)	最高价(元)	最低价(元)	收盘价(元)	成交额(百万)	成交量(股)
columns = ['symbol', 'name', 'date', 'open', 'high', 'low', 'close', 'notional', 'volume']
bbg_columns = ['date', 'open', 'close', 'high', 'low', 'mid', 'volume']
eua_columns = ['date', 'close']


def read_price(symbol, start_date=None, end_date=None):
    # symbol = 'Adjusted KRBN'
    # start_date = '2015-01-01'
    # end_date = '2017-07-01'

    prices = pd.read_excel(p, sheet_name=symbol)
    prices.columns = eua_columns
    if len(prices.columns) == len(columns):
        prices.columns = columns
    elif len(prices.columns) == len(bbg_columns):
        prices.columns = bbg_columns
    elif len(prices.columns) == len(eua_columns):
        prices.columns = eua_columns

    prices = prices.set_index('date', drop=False).sort_index()
    prices['ret'] = prices.close.pct_change()

    if start_date is not None and end_date is not None:
        prices = prices[start_date:end_date]

    prices = prices.dropna()
    return prices


def single_price_momentum(prices, look_back, long_weight=1, short_weight=-1, threshold=0):
    assert look_back > 1, 'look back window must > 1'
    assert long_weight >= 0, 'long weight must > 0'
    assert short_weight <= 0, 'short weight must < 0'

    p = pd.DataFrame({
        't0': prices,
        't_m_b': prices.shift(look_back),
        't_p_1': prices.shift(-1),
        't_m_1': prices.shift(1)
    })
    p['look_back_return'] = (p.t_m_1 - p.t_m_b) / p.t_m_b
    p['signal'] = p['look_back_return'].apply(lambda x: long_weight if x >= threshold else short_weight if x <= -threshold else 0)
    p['ret'] = (p.t_p_1 - p.t0) / p.t0 * p.signal

    return p[['signal', 'ret']].shift(1).dropna()


def simulate_portfolio(port):
    port_table = []
    sharpe = pf.ep.sharpe_ratio(port)
    sortino = pf.ep.sortino_ratio(port)
    vol = pf.ep.annual_volatility(port)
    max_dd = pf.ep.max_drawdown(port)
    annual = pf.ep.annual_return(port)
    port_table.append(dict(sharpe=sharpe, sortino=sortino, vol=vol, max_drawdown=max_dd, annual=annual))
    port_table = pd.DataFrame(port_table)
    return port_table

