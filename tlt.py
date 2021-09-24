from util2 import read_price
import pyfolio as pf
import matplotlib.pyplot as plt
import datetime

start_date = datetime.date(2005, 1, 1).strftime('%Y-%m-%d')
end_date = datetime.date(2021, 12, 31).strftime('%Y-%m-%d')

tlt = read_price('Adjusted TLT', start_date, end_date)
spy = read_price('Adjusted SPY', start_date, end_date)

eth = read_price('GDAX ETH', start_date, end_date)
eth = read_price('GDAX ETH', start_date, end_date)

port = tlt.ret * 0.6 + spy.ret * 0.4

tlt.head()

tlt.ret

pf.plotting.plot_annual_returns(tlt.ret)
pf.plotting.plot_annual_returns(spy.ret)
plt.show()

pf.ep.sharpe_ratio(tlt.ret)
pf.ep.max_drawdown(tlt.ret)
pf.ep.annual_return(tlt.ret)

pf.ep.sharpe_ratio(spy.ret)
pf.ep.max_drawdown(spy.ret)
pf.ep.annual_return(spy.ret)

pf.ep.sharpe_ratio(port)
pf.ep.max_drawdown(port)
pf.ep.annual_return(port)
