import pandas as pd
import matplotlib.pyplot as plt
import datetime
import util as ut


def prepare_trades():
    month_range = [3, 4, 5, 6, 7, 8]
    daily_futures_ret, daily_futures_pnl = ut.compute_continuous_futures_pnl(month_range)

    gbtc_intraday_price = ut.get_gbtc_intraday_price()
    gbtc_daily_price = ut.compute_daily_price(gbtc_intraday_price)
    daily_gbtc_ret = gbtc_daily_price['close'].pct_change()

    trades = pd.DataFrame(dict(gbtc_ret=daily_gbtc_ret, futures_ret=daily_futures_ret))
    trades['total_pnl'] = trades.gbtc_ret / 2 - trades.futures_ret / 2

    return trades


def get_volume(prices, asof, start_time, end_time, multiplier=1):
    prices = prices[prices.date == asof]
    prices = prices[(prices.time <= end_time) & (prices.time >= start_time)]
    # print(prices)
    return int((prices.close * prices.volume).sum() * multiplier)


def get_average_price(prices, asof, start_time, end_time):
    prices = prices[prices.date == asof]
    prices = prices[(prices.time <= end_time) & (prices.time >= start_time)]
    price = (prices.close * prices.volume).sum() / prices.volume.sum()
    return price


def simulate_intraday(trade_in_date, start_trade_in_time, end_trade_in_time, trade_out_date, start_trade_out_time, end_trade_out_time):
    # trade_in_date = datetime.date(2021, 5, 12)
    # trade_out_date = datetime.date(2021, 5, 20)

    # start_trade_in_time = datetime.time(15, 45)
    # end_trade_in_time = datetime.time(16, 0)
    #
    # start_trade_out_time = datetime.time(15, 45)
    # end_trade_out_time = datetime.time(16)

    # start_trade_in_time = datetime.time(9, 30)
    # end_trade_in_time = datetime.time(9, 45)

    # start_trade_out_time = datetime.time(9, 30)
    # end_trade_out_time = datetime.time(9, 45)

    assert trade_in_date.month == trade_out_date.month

    futures_prices = ut.get_future_intraday_price(trade_in_date.month)

    futures_trade_in_price = get_average_price(futures_prices, trade_in_date, start_trade_in_time, end_trade_in_time)
    futures_trade_out_price = get_average_price(futures_prices, trade_out_date, start_trade_out_time, end_trade_out_time)

    btc_prices = ut.get_gbtc_intraday_price()
    btc_trade_in_price = get_average_price(btc_prices, trade_in_date, start_trade_in_time, end_trade_in_time)
    btc_trade_out_price = get_average_price(btc_prices, trade_out_date, start_trade_out_time, end_trade_out_time)

    pnl = (btc_trade_out_price - btc_trade_in_price) / btc_trade_in_price - (futures_trade_out_price - futures_trade_in_price) / futures_trade_in_price
    return pnl


def simulate(all_trades):
    trading_days = all_trades[all_trades.pre_day_discount <= all_trades.trade_in_threshold].index
    print(trading_days)

    all_result = []
    for trading_day in trading_days:
        result = {}
        trades = all_trades[trading_day:].head(10)
        t = trades[trades.trade_out_threshold <= trades.discount]
        if len(t):
            trade_out_date = t.index[0]
        else:
            trade_out_date = trades.index[-1]

        trades = trades[:trade_out_date]
        # t.idxmin()
        # trades.iloc[0]
        # trades['pnl'] = trades.total_pnl.cumsum()
        result[f'{trading_day}_cum'] = trades.total_pnl.cumsum()
        result[f'{trading_day}_daily'] = trades.total_pnl
        result[f'{trading_day}_discount'] = trades.discount
        result[f'{trading_day}_pre_day_discount'] = trades.pre_day_discount
        result[f'{trading_day}_trade_in_threshold'] = trades.trade_in_threshold
        result[f'{trading_day}_trade_out_threshold'] = trades.trade_out_threshold
        result[f'{trading_day}_pre_date'] = trades.pre_date
        # print(trades)

        result = pd.DataFrame(result)
        all_result.append((trading_day, result))

    return all_result


# slide: Futures Continuous Pnl
def compute_continuous_futures_pnl():
    month_range = [3, 4, 5, 6, 7, 8]
    ret, pnl = ut.compute_continuous_futures_pnl(month_range)
    ((1 + ret).cumprod() - 1).plot(title='Cumulative Daily Return')
    plt.show()

    (pnl * 5).cumsum().plot(title='Cumulative PnL')
    plt.show()


# slide: Discount diff vs. Return Regression
def compute_official_discount_return_regression():
    trades = prepare_trades()
    official_daily_discount = ut.get_official_daily_discount()
    trades['pre_day_discount'] = official_daily_discount.shift(1)
    trades['discount'] = official_daily_discount
    trades['discount_diff'] = trades.discount - trades.pre_day_discount
    trades.head()

    import statsmodels.api as sm
    x = trades["discount_diff"]
    y = trades["total_pnl"]
    data = pd.DataFrame(dict(x=x, y=y))
    data = data.dropna()
    # Note the difference in argument order
    # X = sm.add_constant(data.x)
    model = sm.OLS(data.y, data.x).fit()
    model.summary()
    # data.head(40)
    # predictions = model.predict(x) # make the predictions by the model
    # Print out the statistics


# slide: simulate
def simulate_with_official_discount():
    trades = prepare_trades()
    trades['date'] = trades.index
    trades['pre_date'] = trades.date.shift(1)
    official_daily_discount = ut.get_official_daily_discount()
    trades['discount'] = official_daily_discount
    trades['pre_day_discount'] = official_daily_discount.shift(1)
    trades['pre_day_discount'] = official_daily_discount.shift(1)
    mean_discount = official_daily_discount.rolling(20).mean()
    std_discount = official_daily_discount.rolling(20).std()
    trades['trade_in_threshold'] = mean_discount - 1.5 * std_discount
    # trades['trade_in_threshold'] = -0.15
    trades['trade_out_threshold'] = mean_discount

    result = simulate(trades)
    summary = []
    for trading_day, r in result:
        # print(trading_day)
        # print(r)
        r.head()
        asof = trading_day.strftime('%Y-%m-%d')
        pnl = r.tail(1)[f'{asof}_cum']
        trade_in_discount = r.head(1)[f'{asof}_pre_day_discount']
        trade_out_discount = r.tail(1)[f'{asof}_discount']
        trade_in_date = r.head(1)[f'{asof}_pre_date']
        trade_out_date = r.tail(1).index[0]
        record = dict(
            trade_in_date=trade_in_date.iloc[0],
            trade_out_date=trade_out_date,
            pnl=pnl.iloc[0],
            trade_in_discount=trade_in_discount.iloc[0],
            trade_out_discount=trade_out_discount.iloc[0],
        )
        summary.append(record)
        # print(asof, pnl.iloc[0], trade_in_discount.iloc[0], trade_out_discount.iloc[0])

    summary = pd.DataFrame(summary)
    print(summary)


start_date = datetime.date(2021, 3, 1)
end_date = datetime.date(2021, 8, 29)
intraday_discount = ut.compute_intraday_discount(start_date, end_date)
intraday_discount['datetime'] = intraday_discount.index
intraday_discount['date'] = intraday_discount.datetime.apply(lambda dt: dt.date())
intraday_discount['time'] = intraday_discount.datetime.apply(lambda dt: dt.time())

official_daily_discount = ut.get_official_daily_discount()
official_daily_discount.index = pd.to_datetime(official_daily_discount.index)

around_open = intraday_discount[intraday_discount.time <= datetime.time(9, 45)]
around_open = around_open.groupby('date').discount.mean()

around_close = intraday_discount[intraday_discount.time >= datetime.time(15, 45)]
# around_close.head()
around_close = around_close.groupby('date').discount.mean()

# around_close.head()
full_day = intraday_discount.groupby('date').discount.mean()
# intraday_discount.head()
# full_day.head()
r = pd.DataFrame(dict(
    official=official_daily_discount,
    around_close=around_close,
    around_open=around_open,
    full_day=full_day,
))

r = r.dropna()
r.head()
(r[['official', 'around_close', 'around_open', 'full_day']] * -1).plot(kind='bar')
plt.show()
r.head()
r[['official', 'around_close', 'around_open', 'full_day']].std()
plt.show()
r['diff'] = r.intraday - r.official
r['diff'].max()
r['diff'].min()

r.index = [i.date() for i in r.index]
r.index[0]
r.loc[datetime.date(2021, 5, 12)]
r.loc[datetime.date(2021, 5, 20)]
r.index

intraday_discount['2021-05-21'].discount.plot()
plt.show()
intraday_discount['2021-05-24'].discount.plot()
plt.show()
official_daily_discount['2021-05-21']

intraday_discount['2021-08-17'].discount.tail(10).mean()

intraday_discount.min()
intraday_discount.max()

trade_in_date = datetime.date(2021, 8, 3)
trade_out_date = datetime.date(2021, 8, 25)

r.loc[trade_in_date], r.loc[trade_out_date]

# start_trade_in_time = datetime.time(15, 45)
# end_trade_in_time = datetime.time(16, 0)
#
# start_trade_out_time = datetime.time(15, 45)
# end_trade_out_time = datetime.time(16)

start_trade_in_time = datetime.time(9, 30)
end_trade_in_time = datetime.time(9, 45)

start_trade_out_time = datetime.time(9, 30)
end_trade_out_time = datetime.time(9, 45)

simulate_intraday(trade_in_date, start_trade_in_time, end_trade_in_time, trade_out_date, start_trade_out_time, end_trade_out_time)

futures_prices = ut.get_future_intraday_price(8)
gbtc_prices = ut.get_gbtc_intraday_price()
futures_prices.head()

get_volume(futures_prices, trade_out_date, start_trade_out_time, end_trade_out_time, 5)
get_volume(gbtc_prices, trade_out_date, start_trade_out_time, end_trade_out_time)
