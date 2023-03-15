from q2 import preprocess
import numpy as np
import matplotlib.pyplot as plt
from options import Put, Call, FindAnswer

AAPL_mar31 = """Mar 31	13.50	0.95 	13.65	13.85	5	422	140.00	0.64	-0.55 	0.64	0.65	371	3216
Mar 31	9.50	1.36 	9.35	9.60	7	10727	145.00	1.32	-0.94 	1.30	1.31	1678	3678
Mar 31	5.69	0.84 	5.70	5.80	202	2741	150.00	2.60	-1.30 	2.57	2.59	841	3217
Mar 31	2.91	0.51 	2.91	2.92	1432	6552	155.00	4.90	-1.60 	4.65	4.85	34	884
Mar 31	1.16	0.19 	1.14	1.16	729	6468	160.00	8.30	-0.60 	7.90	8.15	11	328
Mar 31	0.37	0.02 	0.36	0.38	110	7623	165.00	14.21	--	12.15	12.55	--	82"""

time_mar31 = 18/365

AAPL_apr28 = """Apr 28	14.45	--	14.60	15.90	--	13	140.00	1.84	-0.69 	1.76	1.87	171	426
Apr 28	11.76	0.66 	11.10	11.70	1	14	145.00	3.00	-0.60 	2.80	2.95	10	191
Apr 28	8.00	0.28 	7.85	8.30	13	37	150.00	4.45	-1.25 	4.35	4.50	46	266
Apr 28	5.25	0.30 	5.15	5.35	23	270	155.00	6.58	-0.82 	6.50	6.70	4	32
Apr 28	2.94	0.14 	2.97	3.15	6	378	160.00	9.74	-0.26 	9.05	10.45	2	5
Apr 28	1.55	-0.04 	1.54	1.67	3	104	165.00	14.25	--	12.05	13.95	--	4"""

time_apr28 = 46/365

AAPL_jun16 = """Jun 16	18.50	0.30 	18.25	18.45	19	35423	140.00	4.10	-1.00 	4.05	4.15	339	50162
Jun 16	14.77	0.66 	14.65	14.85	19	20167	145.00	5.50	-1.20 	5.45	5.55	156	26301
Jun 16	11.60	0.80 	11.45	11.65	317	34463	150.00	7.23	-1.37 	7.20	7.30	208	61822
Jun 16	8.75	0.70 	8.70	8.75	376	30655	155.00	9.36	-1.64 	9.35	9.45	445	20164
Jun 16	6.35	0.50 	6.30	6.40	498	36973	160.00	12.30	-1.07 	11.95	12.10	16	12083
Jun 16	4.48	0.19 	4.40	4.45	40	20906	165.00	15.25	-1.24 	15.00	15.25	1	6499"""

time_jun16 = 95/365

AAPL_jul21 = """Jul 21	19.80	-0.10 	19.80	20.05	1	2449	140.00	5.00	-1.05 	4.90	5.00	81	9673
Jul 21	16.45	0.80 	16.30	16.65	8	3779	145.00	6.60	-0.90 	6.40	6.45	15	6334
Jul 21	13.21	0.32 	13.10	13.50	71	5205	150.00	8.20	-1.35 	8.15	8.25	120	6840
Jul 21	10.40	0.35 	10.35	10.45	139	8432	155.00	10.40	-1.25 	10.30	10.40	47	4079
Jul 21	8.10	0.35 	7.90	8.00	79	12593	160.00	13.45	--	12.80	13.10	--	2215
Jul 21	5.85	0.35 	5.85	5.95	177	9842	165.00	15.70	-0.80 	15.80	16.20	2	2893"""

time_jul21 = 129/365

colors = ["#ecffc1", "#c3febf", "#76fdbc", "#ecdffd", "#c3affd", "#7682fc"]

stock_price = 153.12

risk_free = 0.02

def iv_call(cost, strike_price, stock_price, time_maturity):
    def f(s):
        return Call(strike_price).cost(stock_price, risk_free, s, time_maturity)
    return FindAnswer(0.00001, 300, cost, f).answer

def iv_put(cost, strike_price, stock_price, time_maturity):
    def f(s):
        return Put(strike_price).cost(stock_price, risk_free, s, time_maturity)
    return FindAnswer(0.00001, 300, cost, f).answer

def q5():
    datas = np.zeros((4, 6, 5))
    datas[0] =  preprocess(AAPL_mar31)
    datas[1] =  preprocess(AAPL_apr28)
    datas[2] =  preprocess(AAPL_jun16)
    datas[3] =  preprocess(AAPL_jul21)

    times = [time_mar31, time_apr28, time_jun16, time_jul21]

    # Plots for call
    plt.figure()

    # Set the x axis text
    x = np.array([0, 1, 2, 3])
    xticks = ['Mar 24','Apr 28','Jun 16','Sep 15']
    plt.xticks(x, xticks)

    for i in range(6):
        strike_price = datas[0, i, 0]
        mid_prices = (datas[:, i, 1] + datas[:, i, 2])/2
        y = []
        for (price, t) in zip(mid_prices, times):
            implied_vol = iv_call(price, strike_price, stock_price, t)
            y.append(implied_vol)
        plt.plot(x, y, color=colors[i], label=f"Strike price: {strike_price}")

    plt.legend(loc="upper right")
    plt.show()

    # Plots for put
    plt.figure()

    # Set the x axis text
    plt.xticks(x, xticks)

    for i in range(6):
        strike_price = datas[0, i, 0]
        mid_prices = (datas[:, i, 3] + datas[:, i, 4])/2
        y = []
        for (price, t) in zip(mid_prices, times):
            implied_vol = iv_put(price, strike_price, stock_price, t)
            y.append(implied_vol)
        plt.plot(x, y, color=colors[i], label=f"Strike price: {strike_price}")

    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    q5()