import pandas as pd
import matplotlib.pyplot as plt
import datetime
import util as ut
import functools as ft


# combined = pd.DataFrame(dict(gbtc=gbtc[start_time:end_time].Close, btc=btc[start_time:end_time].Close, btc_spot=btc_spot[start_time:end_time]))
# combined['gbtc_fair_value'] = combined.btc_spot * btc_per_share
# combined['discount'] = (combined.gbtc - combined.gbtc_fair_value) / combined.gbtc_fair_value
# # combined.head()
# combined['gbtc_ret'] = combined.gbtc.pct_change()
# combined['btc_ret'] = combined.btc.pct_change()
# combined['btc_spot_ret'] = combined.btc_spot.pct_change()
# combined.tail()
# combined.head()
#
# combined[['gbtc_ret', 'btc_ret', 'btc_spot_ret']].cumsum().plot()
# combined.discount.plot()
# plt.show()
# GBTC Prices

# daily discount stats

start_date = datetime.date(2021, 3, 1)
end_date = datetime.date(2021, 8, 31)

intraday_discount = ut.compute_intraday_discount(start_date, end_date)

daily_discount = ut.compute_daily_discount(intraday_discount)
daily_discount.head()

official_daily_discount = daily_discount.official

official_std = official_daily_discount.rolling(20).std()

futures_prices = ut.get_future_intraday_price(3)
ut.compute_daily_price(futures_prices)

month_range = [3, 4, 5, 6, 7, 8]
daily_ret, daily_pnl = ut.compute_continuous_futures_pnl(month_range)

# daily_ret.cumsum().plot()
# plt.show()
# daily_pnl.cumsum().plot()
# plt.show()

# my_daily['official_mean'] = my_daily.official.rolling(10).mean()
# my_daily['official_upper'] = my_daily.official_mean + 1.5 * official_std
# my_daily['official_lower'] = my_daily.official_mean - 1.5 * official_std

# my_daily[['official', 'official_mean', 'official_lower', 'official_upper']].plot()

trades = pd.DataFrame(dict(future_pnl=))
trade_in_dates = ut.my_daily[my_daily.official <= my_daily.official_lower]['official']

plt.show()

future_prices = [get_future_price(month) for month in [3, 4, 5, 6, 7, 8]]
future_prices = pd.concat(future_prices)
# future_prices.tail()
# future_prices.head()
future_daily_price = get_daily_price(future_prices)
# future_daily_price.head()

gbtc_prices = get_gbtc_price()
# gbtc_prices.head()
gbtc_daily_price = get_daily_price(gbtc_prices)

gbtc_daily_price['official_discount'] = my_daily.official
gbtc_daily_price['mean_discount'] = my_daily['mean']
gbtc_daily_price['future_open'] = future_daily_price['open']
gbtc_daily_price['future_close'] = future_daily_price['close']
gbtc_daily_price.head()

prices = gbtc_daily_price[['open', 'close', 'official_discount', 'mean_discount', 'future_open', 'future_close']]
prices.columns = ['gbtc_open', 'gbtc_close', 'official_discount', 'mean_discount', 'future_open', 'future_close']
prices['gbtc_close_t_m_1'] = prices.gbtc_close.shift(1)
prices['future_close_t_m_1'] = prices.future_close.shift(1)
prices['mean_discount_t_m_1'] = prices.mean_discount.shift(1)
prices['official_discount_t_m_1'] = prices.official_discount.shift(1)

prices[['official_discount', 'mean_discount']].plot()
plt.show()

# prices['gbtc_trading_pnl'] = (prices.gbtc_close - prices.gbtc_open) / prices.gbtc_open
# prices['future_trading_pnl'] = -(prices.future_close - prices.future_open) / prices.future_open
# prices['trading_pnl'] = prices.gbtc_trading_pnl + prices.future_trading_pnl

prices['gbtc_position_pnl'] = (prices.gbtc_close - prices.gbtc_close_t_m_1) / prices.gbtc_close_t_m_1
prices['future_position_pnl'] = -(prices.future_close - prices.future_close_t_m_1) / prices.future_close_t_m_1
prices['position_pnl'] = prices.gbtc_position_pnl + prices.future_position_pnl

prices[prices.official_discount_t_m_1 <= -0.16].position_pnl.sum()

prices[prices.official_discount_t_m_1 <= -0.15].g_pnl.sum()

prices[prices.official_discount_t_m_1 <= -0.15].trading_pnl.cumsum().plot()
plt.show()

prices.head()

prices[prices.official_discount <= -0.1]

trade_in_dates.index[0]
trade_in_date = trade_in_dates.index[0]
trade_out_date = trade_in_date + datetime.timedelta(days=15)

pnls = prices.loc[trade_in_date:trade_out_date][['official_discount', 'position_pnl']]
pnls
pnls.sum()

prices[['official_discount', 'mean_discount']].plot()
plt.show()


my_daily[['mean', 'official']].plot()
plt.show()

rolling_std = my_daily['mean'].rolling(20).std()
rolling_mean = my_daily['mean'].rolling(20).mean()
# upper = rolling_mean + 1 * rolling_std
# lower = rolling_mean - 1 * rolling_std
# rolling_discount = pd.DataFrame(dict(mean=rolling_mean, upper=upper, lower=lower, discount=my_daily['mean']))
rolling_discount = pd.DataFrame(dict(mean=rolling_mean, discount=my_daily['mean']))
# rolling_discount.head(20)
rolling_discount.plot()
plt.show()
my_daily[my_daily['mean'] < -0.12]

(my_daily['max'] - my_daily['min']).plot(kind='bar')
(my_daily['max'] - my_daily['min']).mean()
my_daily['std'].plot(kind='bar')
plt.show()

discount.discount.rolling(20).std().plot.bar()
plt.show()

discount.discount.rolling('20d').std().plot()
plt.show()

discount['month'] = discount.datetime.apply(lambda dt: dt.month())
discount.head()


def simulate(trade_in_date):
    # trade_in_date = datetime.date(2021, 3, 25)
    trade_out_date = trade_in_date + datetime.timedelta(days=20)

    future_prices = get_future_price(trade_in_date.month)
    # future_prices.head()
    future_daily_price = get_daily_price(future_prices)

    gbtc_prices = get_gbtc_price()
    # gbtc_prices.head()
    gbtc_daily_price = get_daily_price(gbtc_prices)

    daily_combined = pd.DataFrame(dict(future_open=future_daily_price.open, gbtc_open=gbtc_daily_price.open, future_close=future_daily_price.close,
                                       gbtc_close=gbtc_daily_price.close))
    trades = daily_combined[trade_in_date:trade_out_date]

    future_trade_in_price = trades.loc[trade_in_date].future_open
    gbtc_trade_in_price = trades.loc[trade_in_date].gbtc_open

    trades['gbtc_pnl'] = (trades.gbtc_close - gbtc_trade_in_price) / gbtc_trade_in_price
    trades['future_pnl'] = -(trades.future_close - future_trade_in_price) / future_trade_in_price
    trades['pnl'] = trades.gbtc_pnl + trades.future_pnl
    trades

gbtc_prices['notional'] = gbtc_prices.Volume * gbtc_prices.Close
gbtc_daily_volume = gbtc_prices.groupby('date').notional.sum().astype(int)
gbtc_daily_volume.mean() * 0.02
gbtc_daily_volume.index[0]
gbtc_daily_volume.plot()
plt.show()
gbtc_prices['2021-08-18'].Volume.plot(kind='bar')
plt.show()


result_ret = pd.Series(result_ret)
result_ret = result_ret.sort_index()
(result_ret + 1).cumprod().tail()

march = result_ret[datetime.date(2021, 3, 1): datetime.date(2021, 3, 25)]
march.head()
((march + 1).cumprod() - 1).tail()
march.cumsum().tail()

(52215 - 48795) / 48795

# (1 + result).cumprod().plot()
# result.cumsum().plot()
# plt.show()

pd.DataFrame(dict(sum=result.cumsum(), prod=((1 + result).cumprod() - 1))).plot()
plt.show()
result.tail()

result_pnl = pd.Series(result_pnl)
result_pnl = result_pnl.sort_index()
result_pnl.sum()
# result_pnl.cumsum().plot()
# plt.show()

(46830 - 48795)
result.sum()

official_discount = get_official_discount()

pnl = pd.DataFrame(dict(gbtc_ret=gbtc_ret, ret=result_ret))
pnl['pre_discount'] = gbtc_daily_price.official_discount.shift(1)
pnl['discount'] = official_discount.discount
# pnl['mean_discount'] = gbtc_daily_price.mean_discount
pnl['discount_diff'] = pnl.discount - pnl.pre_discount
pnl['total_pnl'] = pnl.gbtc_ret / 2 - pnl.ret / 2
pnl['mean_discount'] = pnl.discount.rolling(20).mean()

pnl.head(30)

daily = [col for col in result.columns if 'daily' in col]
result[daily].mean().mean()

pnl.total_pnl.corr(pnl.discount_diff)

pnl.total_pnl.cumsum().plot()
plt.show()

gbtc_daily_price.mean_discount.plot()
plt.show()

pnl.head(30)

my_daily

official_discount
