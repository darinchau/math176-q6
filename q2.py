import numpy as np
import matplotlib.pyplot as plt
from options import FindAnswer, Call, Put

# Copied straight from the provided website as of 9:25pm California time, 13 Mar 2023
AAPL_data = """Apr 6	9.10	2.00 	8.45	8.75	23	125	145.00	2.62	-0.68 	2.56	2.64	585	1071
Apr 6	8.70	2.20 	7.75	8.05	58	287	146.00	2.92	-0.66 	2.61	2.94	215	468
Apr 6	7.69	2.04 	6.90	7.35	18	374	147.00	3.05	-0.91 	3.15	3.25	365	390
Apr 6	6.50	1.25 	6.45	6.70	109	146	148.00	3.65	-0.90 	3.50	3.60	401	273
Apr 6	6.33	1.85 	5.80	6.05	383	481	149.00	3.80	-1.20 	3.85	4.00	209	774
Apr 6	5.30	1.04 	5.00	5.40	463	1640	150.00	4.35	-0.95 	3.85	4.40	691	850
Apr 6	3.95	0.90 	3.95	4.10	609	1026	152.50	5.50	-1.70 	5.15	5.80	528	261
Apr 6	2.88	0.65 	2.65	2.91	770	2686	155.00	6.85	-1.50 	6.55	7.05	159	402
Apr 6	2.18	0.80 	1.76	2.20	637	892	157.50	7.05	-2.85 	8.15	8.80	8	183
Apr 6	1.31	0.33 	1.13	1.46	757	3683	160.00	9.20	-2.81 	10.15	10.65	32	2584
Apr 6	0.83	0.26 	0.81	0.95	521	2110	162.50	11.00	-1.25 	12.15	12.70	2	32
Apr 6	0.56	0.17 	0.50	0.56	514	1957	165.00	14.25	-1.76 	14.30	15.65	7	6"""

GOOG_data = """Apr 6	8.05	1.63 	6.95	8.35	1	12	86.00	1.23	-0.34 	1.28	1.52	58	280
Apr 6	6.70	-3.19 	5.80	6.70	1	88	87.00	1.46	-0.33 	1.59	1.78	104	446
Apr 6	6.56	--	5.15	6.05	--	30	88.00	1.90	-0.25 	1.87	2.09	46	262
Apr 6	5.30	-0.45 	4.55	5.90	2	85	89.00	1.76	-0.64 	2.03	2.25	18	427
Apr 6	4.77	0.72 	4.45	5.10	19	216	90.00	2.47	-0.30 	2.34	2.61	133	251
Apr 6	4.12	0.52 	3.50	4.10	132	246	91.00	2.96	-0.49 	2.68	3.05	168	87
Apr 6	3.55	0.15 	3.05	3.45	68	214	92.00	3.30	-0.11 	3.35	3.50	130	270
Apr 6	3.00	0.39 	2.62	3.25	71	268	93.00	3.50	-0.40 	3.85	4.00	164	49
Apr 6	2.52	0.10 	2.22	2.48	109	139	94.00	4.46	-0.29 	3.90	5.10	39	52
Apr 6	2.12	0.33 	1.87	2.24	113	1246	95.00	4.74	-0.46 	4.35	5.50	21	141
Apr 6	1.76	0.06 	1.67	1.72	141	198	96.00	5.34	0.19 	4.90	6.00	10	99
Apr 6	1.62	0.42 	1.36	1.42	163	245	97.00	5.79	-0.81 	5.45	7.45	9	26"""

risk_free = 0.02

# Calculated as of 13 Mar
time_maturity = 24/365

# String processing magic
def preprocess(st):
    data = [row.split("\t") for row in st.split("\n")]
    datas = []
    # Call bid is row 3, call ask is row 4, strike price is row 7, put bid is row 10, put ask is row 11
    for row in data:
        x = [row[7], row[3], row[4], row[10], row[11]]
        x = [float(a) for a in x]
        datas.append(x)
    datas = np.array(datas)
    return datas

def implied_vol_call(cost: float, strike_price: float, stock_price: float):
    def f(s):
        return Call(strike_price).cost(stock_price, risk_free, s, time_maturity)
    return FindAnswer(0.01, 300, cost, f).answer

def implied_vol_put(cost: float, strike_price: float, stock_price: float):
    def f(s):
        return Put(strike_price).cost(stock_price, risk_free, s, time_maturity)
    return FindAnswer(0.01, 300, cost, f).answer

def q2(data, stock_price, hist_vol):
    mid_call = (data[:, 1] + data[:, 2])/2
    mid_put = (data[:, 3] + data[:, 4])/2

    # Call plot
    plt.figure()
    x = []
    y = []
    for (strike_price, cost) in zip(data[:, 0], mid_call):
        x.append(strike_price)
        y.append(implied_vol_call(cost, strike_price, stock_price))
    # Mint color
    plt.plot(x, y, color="#3eb489")

    # Put plot
    x = []
    y = []
    for (strike_price, cost) in zip(data[:, 0], mid_put):
        x.append(strike_price)
        y.append(implied_vol_put(cost, strike_price, stock_price))
    # Turquoise color
    plt.plot(x, y, color="#30d5c8")

    # Also plot the sd from question 1
    x = []
    y = []
    for (strike_price, _) in zip(data[:, 0], mid_put):
        x.append(strike_price)
        y.append(hist_vol)
    # Crimson red color
    plt.plot(x, y, '--', color="#b90e0a")

    plt.show()

if __name__ == "__main__":
    aapl = preprocess(AAPL_data)
    goog = preprocess(GOOG_data)

    # My implementations for calculating option price is in percentage, so artificially multiply volatility by 100
    q2(aapl, 150.47, 25.534)
    q2(goog, 91.66, 49.816)
