import pandas as pd

import util as ut1, util2 as ut2
import matplotlib.pyplot as plt
import pyfolio as pf


def generate_cme_price():
    cme = ut1.compute_continuous_futures_pnl(range(3, 9))
    cme_ret, cme_pnl = cme
    cme_price = (cme_ret + 1).cumprod()
    return cme_price


def generate_signal(price, rolling):
    # rolling = 10
    data = pd.DataFrame(dict(price=price))
    data['t_m_1'] = data.price.shift(1)
    data['moving_average'] = data.t_m_1.rolling(rolling).mean()
    data['standard_error'] = data.t_m_1.rolling(rolling).std()
    data['upper'] = data.moving_average + 2 * data.standard_error
    data['lower'] = data.moving_average - 2 * data.standard_error
    data = data.dropna()
    data['signal'] = data.apply(lambda row: 1 if row.price > row.moving_average else -1, axis=1)
    data['last_signal'] = data.signal.shift(1)

    return data


def simulate(data):
    trades = []
    for i in range(0, len(data)):
        row = data.iloc[i]
        if i == 0:
            trades.append(dict(date=data.index[i], cash=-row.price * row.signal))
        elif row.signal != row.last_signal:
            trades.append(dict(date=data.index[i], cash=row.price * row.last_signal))
            trades.append(dict(date=data.index[i], cash=-row.price * row.signal))
        elif i == len(data) - 1:
            trades.append(dict(date=data.index[i], cash=row.price * row.signal))

    trades = pd.DataFrame(trades)
    trades['last_cash'] = trades.cash.shift(1)
    trades = trades.dropna()
    trades = trades[trades.cash != trades.last_cash]
    trades['pnl'] = trades.cash + trades['last_cash']

    return trades


price = ut2.read_price('GDAX ETH')
data = generate_signal(price.close, 20)
trades = simulate(data)
trades.pnl.sum()
daily_pnl = trades.set_index('date').pnl
daily_pnl.cumsum().plot()
plt.show()

# cme_price.plot()
# cme_price.index = pd.to_datetime(cme_price.index)
# cme = pd.DataFrame(dict(price=btc['2021'].close))
# plt.show()
