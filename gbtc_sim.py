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
    # print(trade_in_date, trade_out_date)
    # trade_in_date = datetime.date(2021, 3, 12)
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

    c = ut.select_contract(ut.month_range)
    # contracts.head()
    # contracts = contracts[trade_in_date]

    contracts = pd.DataFrame(dict(contract=c))
    contracts['next_contract'] = contracts.contract.shift(-1)
    contracts['roll'] = contracts.contract != contracts.next_contract

    trading_days = contracts[trade_in_date:trade_out_date]

    trading_days = trading_days.head(1).index.to_list() + trading_days[trading_days.roll == True].index.to_list() + \
                   trading_days.tail(1).index.to_list()
    trading_days = pd.DataFrame(dict(trading_date=list(set(trading_days))))
    trading_days = trading_days.sort_values('trading_date')
    trading_days['contract'] = trading_days.trading_date.apply(lambda d: c.get(d))
    # trading_days

    trade_in_prices = []
    trade_out_prices = []
    trade_in_dates = []
    trade_out_dates = []
    trade_in_contract = []
    trade_out_contract = []

    for i in range(0, len(trading_days)):
        trading_day = trading_days.iloc[i]
        # print(trading_day)
        contract = int(trading_day.contract)
        trading_date = trading_day.trading_date
        if i == 0:
            # trade in
            prices = ut.get_future_intraday_price(contract)
            price = get_average_price(prices, trading_date, start_trade_in_time, end_trade_in_time)
            trade_in_prices.append(price)
            trade_in_dates.append(trading_date)
            trade_in_contract.append(contract)
        elif i == len(trading_days) - 1:
            # trade out
            prices = ut.get_future_intraday_price(contract)
            price = get_average_price(prices, trading_date, start_trade_in_time, end_trade_in_time)
            trade_out_prices.append(price)
            trade_out_dates.append(trading_date)
            trade_out_contract.append(contract)
        else:
            # roll
            prices = ut.get_future_intraday_price(contract)
            price = get_average_price(prices, trading_date, start_trade_in_time, end_trade_in_time)
            trade_out_prices.append(price)
            trade_out_dates.append(trading_date)
            trade_out_contract.append(contract)

            contract += 1
            prices = ut.get_future_intraday_price(contract)
            price = get_average_price(prices, trading_date, start_trade_in_time, end_trade_in_time)
            trade_in_prices.append(price)
            trade_in_dates.append(trading_date)
            trade_in_contract.append(contract)

    if len(trade_in_prices) == len(trade_out_prices):
        result = pd.DataFrame(dict(
            trade_in_price=trade_in_prices,
            trade_out_price=trade_out_prices,
            trade_in_date=trade_in_dates,
            trade_out_date=trade_out_dates,
            trade_in_contract=trade_in_contract,
            trade_out_contract=trade_out_contract,
        ))

        result['ret'] = (result.trade_out_price - result.trade_in_price) / result.trade_in_price
        # ((result.ret + 1).cumprod() - 1)
        futures_return = result.ret.sum()

        # trade_in_contract = contracts.get(trade_in_date)
        # trade_out_contract = contracts.get(trade_out_date)
        # trading_contracts = list(range(int(trade_in_contract), int(trade_out_contract) + 1))
        # contracts.head()
        # futures_trade_in_price = get_average_price(futures_prices, trade_in_date, start_trade_in_time, end_trade_in_time)
        # futures_trade_out_price = get_average_price(futures_prices, trade_out_date, start_trade_out_time, end_trade_out_time)

        btc_prices = ut.get_gbtc_intraday_price()
        btc_trade_in_price = get_average_price(btc_prices, trade_in_date, start_trade_in_time, end_trade_in_time)
        btc_trade_out_price = get_average_price(btc_prices, trade_out_date, start_trade_out_time, end_trade_out_time)

        pnl = (btc_trade_out_price - btc_trade_in_price) / btc_trade_in_price - futures_return

        return pnl
    else:
        return None


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
    ((1 + ret).cumprod() - 1).plot(title='CME Bitcoin Future Cumulative Daily Return')
    plt.show()

    (pnl * 5).cumsum().plot(title='CME Bitcoin Future Cumulative PnL')
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
    import numpy as np
    x = np.array(trades["discount_diff"].to_list())
    y = np.array(trades["total_pnl"].to_list())
    #m = np.polyfit(x, y, 1)
    fig = plt.figure()
    plt.scatter(x, y)
    fig.suptitle('daily return vs. overnight discount change')
    plt.ylabel('daily return')
    plt.xlabel('overnight discount change')
    plt.show()
    data = pd.DataFrame(dict(x=x, y=y))
    data = data.dropna()
    # Note the difference in argument order
    # X = sm.add_constant(data.x)
    model = sm.OLS(data.y, data.x).fit()
    s = model.summary()

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
    summary.to_clipboard()
    print(summary)


# slide: compute intraday volume
def compute_intraday_volume():
    month_range = [3, 4, 5, 6, 7, 8]

    contracts = ut.select_contract(month_range)
    # len(contracts)

    data = {}
    for month in month_range:
        # month = 3
        prices = ut.get_future_intraday_price(month)
        # prices.head()
        data[f'{month}_close'] = prices['close']
        data[f'{month}_volume'] = prices['volume']

    data = pd.DataFrame(data)
    data = data.fillna(0)
    # len(data.date.unique())

    data['datetime'] = data.index
    data['date'] = data.datetime.apply(lambda dt: dt.date())
    data['contract'] = data.date.apply(lambda d: contracts.get(d))

    data = data.dropna()

    data['active_volume'] = data.apply(lambda d: d[f'{int(d.contract)}_volume'], axis=1)
    data['active_price'] = data.apply(lambda d: d[f'{int(d.contract)}_close'], axis=1)
    data['active_notional'] = data.active_volume * data.active_price * 5 / 1E6
    data['hour'] = data.datetime.apply(lambda dt: dt.hour)

    data['2021-03-04'].active_notional.cumsum().plot(title='Trading Notional ($Million)')
    plt.show()

    days = len(data.date.unique())
    futures_hourly_volume = data.groupby('hour').active_notional.sum() / days
    futures_hourly_volume.plot(kind='bar', title='CME BTC Hourly Trading Notional Distribution ($Million)')
    plt.show()

    gbtc = ut.get_gbtc_intraday_price()
    days = len(gbtc.date.unique())
    gbtc['notional'] = gbtc.close * gbtc.volume
    gbtc['hour'] = gbtc.time.apply(lambda t: t.hour)
    gbt_hourly_volume = gbtc.groupby('hour').notional.sum() / days / 1E6
    gbt_hourly_volume[[9, 10, 11, 12, 13, 14, 15]].plot(kind='bar', title='GBTC Hourly Trading Notional Distribution ($Million)')
    plt.show()


# slide: intraday discount vs. official discount
def compute_intraday_discount():
    start_date = datetime.date(2021, 3, 1)
    end_date = datetime.date(2021, 8, 29)

    discount = ut.compute_intraday_discount(start_date, end_date)
    discount['datetime'] = discount.index
    discount['time'] = discount.datetime.apply(lambda dt: dt.time())
    discount['date'] = discount.datetime.apply(lambda dt: dt.date())

    around_open = discount[(discount.time >= datetime.time(9, 30)) & (discount.time <= datetime.time(9, 45))]
    around_close = discount[(discount.time >= datetime.time(15, 45)) & (discount.time <= datetime.time(16))]
    around_close = around_close.groupby('date').discount.mean()
    around_open = around_open.groupby('date').discount.mean()

    official_discount = ut.get_official_daily_discount()
    combined = pd.DataFrame(dict(around_open=around_open, around_close=around_close, official=official_discount))
    combined = combined.dropna()
    combined.index = pd.to_datetime(combined.index)
    combined['next_open'] = combined.around_open.shift(-1)

    discount_may = combined['2021-05']
    discount_may[['around_open', 'around_close', 'official']].plot(title='GBTC Intraday Discount vs. Official in May 2021 ')
    plt.show()

    discount_june = combined['2021-06']
    discount_june[['around_open', 'around_close', 'official']].plot(title='GBTC Intraday Discount vs. Official in June 2021 ')

    discount_july = combined['2021-07']
    discount_july[['around_open', 'around_close', 'official']].plot(title='GBTC Intraday Discount vs. Official in July 2021 ')
    plt.show()

    discount_august = combined['2021-08']
    discount_august[['around_open', 'around_close', 'official']].plot(title='GBTC Intraday Discount vs. Official in August 2021 ')
    plt.show()

    # discount distribution by hour
    daily_mean = discount.groupby('date').discount.mean()
    daily_std = discount.groupby('date').discount.std()
    discount['daily_mean'] = discount.apply(lambda row: daily_mean[row.date], axis=1)
    discount['daily_std'] = discount.apply(lambda row: daily_std[row.date], axis=1)
    discount['demean'] = (discount.discount - discount.daily_mean) / discount.daily_std
    discount['hour'] = discount.time.apply(lambda t: t.hour)
    hourly = discount.groupby('hour').demean.mean()
    hourly[[9, 10, 11, 12, 13, 14, 15]].plot(kind='bar', title='Hourly Discount Z-score Distribution')
    plt.show()

    # compare around close vs. next open
    (combined.next_open - combined.around_close).mean()
    (combined.next_open - combined.around_close).std()

    d = discount_august[['around_close', 'next_open']]
    d.index = [dd.date() for dd in d.index.to_list()]
    d.plot(kind='bar', title='Discount Around Today Close vs. Discount Around Next Open (August)')
    plt.show()

    d = discount_july[['around_close', 'next_open']]
    d.index = [dd.date() for dd in d.index.to_list()]
    d.plot(kind='bar', title='Discount Around Today Close vs. Discount Around Next Open (July)')
    plt.show()

    # daily range
    daily_range = discount.groupby('date').discount.max() - discount.groupby('date').discount.min()
    daily_range.index = pd.to_datetime(daily_range.index)

    daily_range_may = daily_range['2021-05']
    daily_range_may.index = [d.date() for d in daily_range_may.index.to_list()]
    daily_range_may.plot(kind='bar', title='Intraday Max Range (May)')
    plt.show()

    daily_range_june = daily_range['2021-06']
    daily_range_june.index = [d.date() for d in daily_range_june.index.to_list()]
    daily_range_june.plot(kind='bar', title='Intraday Max Range (June)')
    plt.show()

    daily_range_july = daily_range['2021-07']
    daily_range_july.index = [d.date() for d in daily_range_july.index.to_list()]
    daily_range_july.plot(kind='bar', title='Intraday Max Range (July)')
    plt.show()

    daily_range_august = daily_range['2021-08']
    daily_range_august.index = [d.date() for d in daily_range_august.index.to_list()]
    daily_range_august.plot(kind='bar', title='Intraday Max Range (August)')
    plt.show()


def simulate_pnl(trade_in_days, trade_out_days, daily_discount):
    result = []

    for i, trade_in_date in enumerate(trade_in_days):
        trade_out_date = trade_out_days[i]

        # print(daily_discount.loc[trade_in_date:trade_out_date])

        # start_trade_in_time = datetime.time(15, 45)
        # end_trade_in_time = datetime.time(16, 0)
        #
        # start_trade_out_time = datetime.time(15, 45)
        # end_trade_out_time = datetime.time(16)

        start_trade_in_time = datetime.time(9, 30)
        end_trade_in_time = datetime.time(9, 45)

        start_trade_out_time = datetime.time(9, 30)
        end_trade_out_time = datetime.time(9, 45)

        pnl = simulate_intraday(trade_in_date, start_trade_in_time, end_trade_in_time, trade_out_date,
                                start_trade_out_time, end_trade_out_time)
        result.append(dict(
            trade_in_date=trade_in_date,
            trade_out_date=trade_out_date,
            pnl=pnl,
            trade_in_discount=daily_discount.loc[trade_in_date].around_open,
            trade_out_discount=daily_discount.loc[trade_out_date].around_open,
        ))

        # print(trade_in_date.date(), trade_out_date.date(), pnl)

    result = pd.DataFrame(result)
    return result


# slide: intraday simulation
def simulate_with_intraday():
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
    around_close = around_close.groupby('date').discount.mean()

    # around_close.head()
    # intraday_discount.head()
    # full_day.head()
    daily_discount = pd.DataFrame(dict(
        around_close=around_close,
        around_open=around_open,
        official=official_daily_discount,
    ))
    daily_discount = daily_discount.dropna()

    daily_discount['last_around_close'] = daily_discount.around_close.shift(1)
    daily_discount['around_close_mean'] = daily_discount.around_close.rolling(20).mean()
    daily_discount['around_close_std'] = daily_discount.around_close.rolling(20).std()
    daily_discount['around_close_upper'] = daily_discount.around_close_mean + 1 * daily_discount.around_close_std
    daily_discount['around_close_lower'] = daily_discount.around_close_mean - 1 * daily_discount.around_close_std
    daily_discount.index = pd.to_datetime(daily_discount.index)

    def find_trade_out_days(trade_in_days, trade_out_days):
        result = []
        for trade_in_date in trade_in_days:
            for trade_out_date in trade_out_days:
                if trade_out_date > trade_in_date:
                    result.append(trade_out_date)
                    break
            else:
                print(f'not found: {trade_in_date}')
        return result

    def moving_average_strategy():
        trade_in_days = daily_discount[daily_discount.last_around_close <= daily_discount.around_close_lower].index.to_list()
        trade_out_days = daily_discount[daily_discount.last_around_close >= daily_discount.around_close_mean].index.to_list()

        trade_out_days = find_trade_out_days(trade_in_days, trade_out_days)
        length = min(len(trade_in_days), len(trade_out_days))

        trade_in_days = trade_in_days[0:length]
        trade_out_days = trade_out_days[0:length]
        # trading_days = pd.DataFrame(dict(trade_in=trade_in_days, trade_out=trade_out_days))

        result = simulate_pnl(trade_in_days, trade_out_days, daily_discount)
        # print(result)
        return result

    def hold_strategy():
        trading_days = daily_discount.index.to_list()
        all_trade_in_days = daily_discount[daily_discount.last_around_close <= daily_discount.around_close_lower].index.to_list()

        summary = []
        for hold in range(2, 15):
            # hold = 2
            trade_in_days = []
            trade_out_days = []

            for trade_in_date in all_trade_in_days:
                i = trading_days.index(trade_in_date)
                trade_out_date = trading_days[min(i + hold, len(trading_days) - 1)]
                trade_in_days.append(trade_in_date)
                trade_out_days.append(trade_out_date)

            result = simulate_pnl(trade_in_days, trade_out_days, daily_discount)
            result['hold'] = hold
            summary.append(result)
            print(result)

        summary = pd.concat(summary)

        return summary

    summary = moving_average_strategy()
    summary.to_clipboard()

    summary = hold_strategy()
    print(summary.groupby('hold').pnl.mean())
    print(summary)

