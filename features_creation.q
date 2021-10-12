t: ("DFFFFFI"; enlist ",") 0: `:/Users/lapkeitang/Documents/PycharmProjects/Nathan_Demonstration/csv/AAPL_20160104_20210930.csv;

t: xcol[`$ssr[;" ";"_"]each string cols t;t];

/ cols t

t: select Date, Open, High, Low, Close, Close_EWMA: (2%1+10) ema Close, Close_50_MA: 50 mavg Close, Adj_Close, Volume from t;

`:/Users/lapkeitang/Documents/PycharmProjects/Nathan_Demonstration/csv/AAPL_20160104_20210930_FROM_Q.csv 0: csv 0: t;

/ t: ("DFFFFFFFI"; enlist ",") 0: `:/Users/lapkeitang/Documents/PycharmProjects/Nathan_Demonstration/csv/AAPL_20160104_20210930_FROM_Q.csv;

exit 0;