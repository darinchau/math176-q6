from math import log as ln
from math import sqrt

def q1(data):
    # Turn the thing into a list of prices. This list is reversed because most recent prices comes first in the csv
    S = [float(d[1:]) for d in data.split("\n")]
    S.reverse()

    hist_returns = []
    for i in range(len(S) - 1):
        hist_returns.append((S[i+1] - S[i])/S[i])

    mean_returns = 1/len(S) * sum(map(lambda x: ln(1 + x), hist_returns))
    sigma_returns = sqrt(1/(len(S) - 1) * sum(map(lambda x: (ln(1 + x) - mean_returns) ** 2, hist_returns)))

    return (252 * mean_returns, sqrt(252) * sigma_returns)

GOOG_data = """$90.10
$89.35
$91.07
$91.80
$92.05
$94.59
$95.78
$97.10
$94.95
$95.00
$94.86
$95.46
$100.00
$108.04
$103.47
$105.22
$108.80
$101.43
$99.87
$97.95"""

AAPL_data = """$147.92
$146.71
$149.40
$148.91
$148.48
$152.55
$153.71
$155.33
$153.20
$153.85
$151.01
$150.87
$151.92
$154.65
$151.73
$154.50
$150.82
$145.43
$144.29
$143.00"""

if __name__ == "__main__":
    print(q1(GOOG_data))
    print(q1(AAPL_data))