from q2 import preprocess
import numpy as np
import matplotlib.pyplot as plt
from options import Put, Call, FindAnswer

GOOG_mar24 = """Mar 24	4.40	0.40 	4.35	4.45	667	328	89.00	1.59	-0.28 	1.57	1.65	560	600
Mar 24	3.90	0.55 	3.70	3.80	607	907	90.00	1.93	-0.26 	1.91	2.12	1150	1671
Mar 24	3.15	0.43 	3.10	3.20	647	556	91.00	2.34	-0.32 	2.30	2.39	650	1320
Mar 24	2.56	0.29 	2.56	2.66	810	434	92.00	2.80	-0.24 	2.56	2.84	855	816
Mar 24	2.09	0.30 	2.08	2.13	1026	545	93.00	3.30	-0.38 	3.25	3.35	878	1608
Mar 24	1.69	0.28 	1.66	1.71	355	1196	94.00	3.85	-0.44 	3.80	3.95	407	762"""

time_mar24 = 11/365

GOOG_apr28 = """Apr 28	7.01	--	6.45	7.45	--	78	89.00	3.10	--	3.35	3.80	--	2
Apr 28	6.25	-0.02 	5.65	6.55	8	4	90.00	3.70	-0.44 	3.75	4.20	3	27
Apr 28	5.80	0.30 	5.15	5.85	1	11	91.00	4.06	-0.15 	4.15	4.65	5	--
Apr 28	5.57	0.37 	4.95	5.30	103	152	92.00	4.28	-0.92 	4.65	5.10	14	25
Apr 28	4.91	0.42 	4.45	4.80	3	12	93.00	4.62	-0.68 	5.10	5.60	9	52
Apr 28	4.60	0.42 	3.90	4.35	25	3	94.00	5.35	0.55 	5.35	6.30	97	50"""

time_apr28 = 46/365

GOOG_jun16 = """Jun 16	8.95	0.12 	8.75	9.45	8	396	89.00	4.63	-0.67 	5.00	5.10	71	1884
Jun 16	8.60	0.75 	8.15	8.80	50	6621	90.00	4.95	-0.40 	5.40	5.50	157	15295
Jun 16	7.95	0.40 	7.60	7.75	33	1316	91.00	5.75	-0.45 	5.80	6.05	215	3415
Jun 16	7.30	0.55 	7.05	7.20	101	2573	92.00	6.30	0.10 	6.25	6.60	70	1941
Jun 16	6.70	0.15 	6.55	6.65	46	3379	93.00	6.65	-0.45 	6.75	6.85	137	3152
Jun 16	6.20	0.46 	6.05	6.15	84	1410	94.00	7.00	-0.39 	7.25	7.35	317	1622"""

time_jun16 = 95/365

GOOG_sep15 = """Sep 15	13.80	--	11.55	12.05	--	804	89.00	6.45	-0.75 	6.80	6.95	226	1019
Sep 15	11.25	0.53 	10.95	11.35	5	1703	90.00	6.95	-0.50 	7.25	7.35	308	5072
Sep 15	10.40	--	10.40	11.05	--	891	91.00	7.55	-0.30 	7.60	7.80	23	881
Sep 15	10.50	0.75 	9.60	10.45	10	862	92.00	7.90	-0.45 	7.90	8.25	47	954
Sep 15	10.05	0.80 	9.35	9.90	37	684	93.00	8.39	-0.46 	8.60	8.75	13	1025
Sep 15	9.65	0.75 	8.85	9.00	46	846	94.00	8.55	-0.72 	9.05	9.20	46	971"""

time_sep15 = 186/365

colors = ["#ecffc1", "#c3febf", "#76fdbc", "#ecdffd", "#c3affd", "#7682fc"]

stock_price = 91.66

risk_free = 0.02

def iv_call(cost, strike_price, stock_price, time_maturity):
    def f(s):
        return Call(strike_price).cost(stock_price, risk_free, s, time_maturity)
    return FindAnswer(0.00001, 300, cost, f).answer

def iv_put(cost, strike_price, stock_price, time_maturity):
    def f(s):
        return Put(strike_price).cost(stock_price, risk_free, s, time_maturity)
    return FindAnswer(0.00001, 300, cost, f).answer

def q4():
    datas = np.zeros((4, 6, 5))
    datas[0] =  preprocess(GOOG_mar24)
    datas[1] =  preprocess(GOOG_apr28)
    datas[2] =  preprocess(GOOG_jun16)
    datas[3] =  preprocess(GOOG_sep15)

    times = [time_mar24, time_apr28, time_jun16, time_sep15]

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
    q4()