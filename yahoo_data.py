import yfinance as yf
import datetime
import pandas as pd
import os

from dateutil.relativedelta import relativedelta

######################################################  USER INPUT  ######################################################
TICKER = 'AAPL'
startDate_String = '2016-01-01'
endDate_String = '2021-09-30'

######################################################  FUNCTION  ######################################################
def create_directory(directory):
    if not os.path.exists(directory):
        print("Creating directory: {}".format(directory))
        os.makedirs(directory)

######################################################  PROGRAM START  ######################################################
# startDate = pd.to_datetime(startDate_String)
# startDate = datetime.datetime.fromisoformat(startDate_String)
startDate = datetime.datetime.strptime(startDate_String, '%Y-%m-%d')
print('Start Date: {}'.format(startDate))

endDate = pd.to_datetime(endDate_String)
print('endDate: {}'.format(endDate))

# start: str
#     Download start date string (YYYY-MM-DD) or _datetime.
#     Default is 1900-01-01
# end: str
#     Download end date string (YYYY-MM-DD) or _datetime.
#     Default is now
df_data = yf.download(TICKER, startDate + relativedelta(days=-10), endDate + relativedelta(days=10))

#####   FILTER   #####
df_data = df_data[(df_data.index >= startDate) & (df_data.index <= endDate)]
# df_data.reset_index(drop=True, inplace=True)
df_data.sort_index(ascending=True, axis=0, inplace=True)
actual_StartDate_String = df_data.index[0].strftime('%Y%m%d')
actual_EndDate_String = df_data.index[-1].strftime('%Y%m%d')

#####   EXPORT   #####
print('length of {0} stock records: {1}'.format(TICKER, len(df_data)))
filename = '{0}_{1}_{2}.csv'.format(TICKER, actual_StartDate_String, actual_EndDate_String)
create_directory('csv')
df_data.to_csv('csv/{}'.format(filename))
print('File ({}) has been exported to directory /csv'.format(filename))
print('Done')