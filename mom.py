import pandas as pd

import util as ut1, util2 as ut2
import matplotlib.pyplot as plt
import pyfolio as pf

import statsmodels.api as sm

sbtc = ut1.get_spot_price(7)
sbtc = pd.DataFrame(dict(close=sbtc))
sbtc['ret'] = sbtc.close.pct_change()
for plus in range(1, 20):
    sbtc[f'ret_p_{plus}'] = sbtc.ret.shift(-plus)
sbtc.head()

gbtc = ut2.read_price('Adjusted GBTC')
gbtc.head()

btc = ut2.read_price('GDAX BTC')
for plus in range(1, 20):
    btc[f'ret_p_{plus}'] = btc.ret.shift(-plus)

btc.head()

eth = ut2.read_price('GDAX ETH')
eth.head()

spy = ut2.read_price('Adjusted SPY')
spy.head()

for year in ['2015', '2016', '2017', '2018', '2019', '2020', '2021']:
    print(year, btc[year].ret.corr(spy[year].ret.shift(1)))

for year in ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07']:
    print(year, btc[year].ret.corr(spy[year].ret.shift(1)))

pf.ep.sharpe_ratio(btc.ret)
pf.ep.sharpe_ratio(eth.ret)

pf.ep.annual_volatility(btc.ret)
pf.ep.annual_volatility(eth.ret)

port = eth.ret - 0.76 * btc.ret
port = port.dropna()
port_price = (port + 1).cumprod()
port_price.plot()
plt.show()

mom = ut2.single_price_momentum(port_price, 10)
mom.ret.cumsum().plot()
pf.ep.sharpe_ratio(mom.ret)
mom.ret.corr(eth.ret)
plt.show()

pf.ep.sharpe_ratio(mom.ret['2021-07':])
pf.ep.max_drawdown(mom.ret['2018':])

# etc['2020:01']
port.cumsum().plot()
plt.show()

pf.ep.sharpe_ratio(port['2020':])
pf.ep.sharpe_ratio(btc.ret['2020':])
pf.ep.sharpe_ratio(eth.ret['2020':])

pf.ep.annual_return(port)

eth['2020':].ret.cumsum().plot()
port['2020':].cumsum().plot()
plt.show()

data = pd.DataFrame(dict(x=btc.ret, y=eth.ret))
data = data.dropna()
model = sm.OLS(data.y, data.x).fit()
s = model.summary()
s
start_date = '2020-05'
(eth[start_date:].ret.cumsum() - btc[start_date:].ret.cumsum()).plot()
plt.show()

month_range = [3, 4, 5, 6, 7, 8]
cme_ret, cme_pnl = ut1.compute_continuous_futures_pnl(month_range)
cme_ret.index = pd.to_datetime(cme_ret.index)
cme_price = (cme_ret + 1).cumprod()

pf.ep.sharpe_ratio(btc.ret)
pf.ep.max_drawdown(btc.ret)
pf.ep.sharpe_ratio(btc.ret)
pf.ep.max_drawdown(btc.ret)


btc.close.plot()
plt.show()

start_date = '2015-01'
for name, ret in [('gbtc', gbtc.ret), ('btc', btc.ret), ('cme', cme_ret), ('eth', eth.ret)]:
    ret = ret[start_date:]
    sp = pf.ep.sharpe_ratio(ret)
    md = pf.ep.max_drawdown(ret)
    print(name, sp, md, ret.sum())


for look_back in [2, 3, 4, 5, 10, 20, 30, 35, 40, 45, 50, 55, 60]:
    mom = ut2.single_price_momentum(eth[start_date:].close, look_back)
    sp = pf.ep.sharpe_ratio(mom.ret)
    md = pf.ep.max_drawdown(mom.ret)
    print(look_back, sp, md)

mom.ret.cumsum().plot()
plt.show()

price = 12.97
nav = 14.89
(price - nav) / nav

c = range(1, 500)
c = [[cc] * 10 for cc in c]
c = [ccc for cc in c for ccc in cc]
c
len(eth)

eth['period'] = c[0:len(eth)]
btc['period'] = c[0:len(btc)]

eth.head(20)

weekly_eth = eth.groupby('period').ret.sum()
weekly_btc = btc.groupby('period').ret.sum()

c
c * 5
weekly_eth.corr(weekly_btc)
weekly_eth.corr(weekly_btc.shift(1))

btc_ret = pd.DataFrame(dict(t0=btc.ret))
btc_ret['t_1'] = btc_ret.t0.shift(1)
btc_ret['t_2'] = btc_ret.t0.shift(2)
btc_ret['t_3'] = btc_ret.t0.shift(3)
btc_ret['t_5'] = btc_ret.t0.shift(5)
btc_ret['t_10'] = btc_ret.t0.shift(10)
btc_ret['t_20'] = btc_ret.t0.shift(20)
btc_ret['t_50'] = btc_ret.t0.shift(50)

btc_ret.head(10)

btc_ret.t0.corr(btc_ret.t_1)
btc_ret.t0.corr(btc_ret.t_10)
btc_ret.t0.corr(btc_ret.t_20)
btc_ret.t0.corr(btc_ret.t_50)

combined = pd.DataFrame(dict(btc=btc.ret, eth=eth.ret))
combined = combined.dropna()
combined.head()

combined['both'] = combined.apply(lambda c: c.btc * c.eth >= 0, axis=1)
combined['both'] = combined.apply(lambda c: c.btc >= 0 and c.eth >= 0, axis=1)
len(combined)
len(combined[combined.both == True])
combined['both_1'] = combined.both.shift(1)
combined['eth_1'] = combined.eth.shift(1)
combined[(combined.both_1 == False) & (combined.eth_1 < 0)].btc.cumsum().plot()
plt.show()
combined[combined.both_1 == True].eth.cumsum().plot()
plt.show()
combined[combined.both_1 == False].eth.sum()


def simulate(btc, rolling, band, hold=1):
    btc['ma'] = btc.rolling(rolling).close.mean()
    btc['ms'] = btc.rolling(rolling).close.std()
    btc['upper'] = btc.ma + band * btc.ms
    btc['lower'] = btc.ma - band * btc.ms
    btc['signal'] = btc.apply(lambda btc: 1 if btc.close < btc.lower else -1 if btc.close > btc.upper else 0, axis=1)

    cols = [f'ret_p_{col}' for col in range(1, hold + 1)]
    total_ret = btc[cols].sum(axis=1)
    simulation = (btc[btc.signal == 1])[['close', 'ma', 'upper', 'lower', 'ret']]
    simulation['total_ret'] = total_ret

    return simulation


def evaluate(simulation):
    win = len(simulation[simulation.total_ret > 0])
    loss = len(simulation[simulation.total_ret < 0])
    sp = pf.ep.sharpe_ratio(simulation.total_ret)
    md = pf.ep.max_drawdown(simulation.total_ret)
    return dict(sp=sp, md=md, size=len(simulation), ratio=win / (win + loss), total_return=simulation.total_ret.sum())


btc = pd.DataFrame(dict(close=sbtc))
prices = sbtc['2021-07']
s = simulate(prices, rolling=20, band=3, hold=10)
s.total_ret.sum()
len(s)
len(s) * 0.0002
s['datetime'] = s.index
s['date'] = s.datetime.apply(lambda dt: dt.date())
daily_s = s.groupby('date').total_ret.sum()
daily_s.cumsum().plot()
plt.show()
pf.ep.sharpe_ratio(daily_s)

p = evaluate(s)
p
s.total_ret.cumsum().plot()
plt.show()

for rolling in [10, 20, 30, 40, 50]:
    # rolling = 5
    s = simulate(prices, rolling=rolling, band=2)
    perf = evaluate(s)
    print(rolling, perf)


for hold in [1, 2, 3, 4, 5, 6, 7, 8 , 9, 10]:
    s = simulate(btc, 10, 1, hold=hold)
    perf = evaluate(s)
    print(hold, perf)

# pf.ep.sharpe_ratio(simulation.ret_p_1)
# len(simulation[simulation.ret_p_1 > 0])
# len(simulation[simulation.ret_p_1 < 0])

for band in [1, 1.5, 2]:
    for rolling in [10, 20, 30, 40]:
        s = simulate(btc, rolling, band, hold=1)
        p = evaluate(s)
        print(band, rolling, len(s), p['sp'], p['ratio'])

btc[['close', 'ma', 'upper', 'lower']]['2021'].plot()
plt.show()

(btc.ret_p_1 > 0

btc['ma'] = btc.rolling(rolling).close.mean()

# btc['ms'] = btc.rolling(rolling).close.std()
# btc['upper'] = btc.ma + band * btc.ms
# btc['lower'] = btc.ma - band * btc.ms
# btc['signal'] = btc.apply(lambda btc: 1 if btc.close < btc.lower else -1 if btc.close > btc.upper else 0, axis=1)
#
# cols = [f'ret_p_{col}' for col in range(1, hold + 1)]
# total_ret = btc[cols].sum(axis=1)
# simulation = (btc[btc.signal == 1])[['close', 'ma', 'upper', 'lower', 'ret']]
