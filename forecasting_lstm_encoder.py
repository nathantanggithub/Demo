import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
import datetime
import holidays
import threading
import time

from queue import Queue
from matplotlib.pylab import rcParams
# from pmdarima.model_selection import train_test_split
from dateutil.relativedelta import relativedelta
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import adfuller, acf

from tensorflow.python.keras.models import Sequential, load_model, Model
from tensorflow.python.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout, Flatten, Input
from tensorflow.python.keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.merge import concatenate

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_curve, roc_auc_score, auc
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

######################################################  USER INPUT  ######################################################
# CSV_FILE = 'AAPL_20160104_20210930.csv'
CSV_FILE = 'AAPL_20160104_20210930_FROM_Q.csv'

LAGS_NUM = 60
EPOCHS = 1
FEATURE_LIST = ['Close', 'Open', 'High', 'Low', 'Close', 'Close_EWMA', 'Close_50_MA', 'Adj_Close', 'IsMondayFriday']  # 'Open', 'High', 'Low', 'Close', 'Close_EWMA', 'Close_50_MA', 'Adj_Close', 'IsMondayFriday', 'Volume'    Remark: Response field must be the first left most item placed in the list
TRAIN_TO_TEST_RATIO = 0.75
FORECAST_STEP = 5
MAKE_X_EWMA = False
SMOOTH_PREDICTION = False
LOWESS_FRACTION = 0.25
PLOT = True

######################################################  GENERAL  ######################################################

#####   USER INPUT CHECK   #####
if SMOOTH_PREDICTION is True and FORECAST_STEP <= 1:
    print('FORECAST_STEP must be > 1 if SMOOTH_PREDICTION is True')
    exit()

# pd.options.mode.chained_assignment = None  # default='warn'
np.set_printoptions(threshold=sys.maxsize)

# lock to serialize console output
lock = threading.Lock()

rcParams['figure.figsize'] = 20, 10

# print(os.path.basename(your_path))
TICKER = CSV_FILE.split('_')[0].replace('.csv', '')

# or:
# us_holidays = holidays.US()
# or:
# us_holidays = holidays.CountryHoliday('US')
# or, for specific prov / states:
# us_holidays = holidays.CountryHoliday('US', prov=None, state='CA')
# print(len(us_holidays))
# print(datetime.datetime(2021,1,2) in us_holidays)
us_holidays = holidays.UnitedStates()

#####   EWMA - Alpha   #####
# com = 2  # com >= 0
# alpha = 1/(1+com)  # for pandas` center-of-mass parameter

# halflife = 2  # halflife > 0
# alpha = 1 - np.exp(np.log(0.5)/halflife)  # for pandas` half-life parameter

span = 2  # span >= 1
alpha = 2 / (span + 1)  # for pandas` span parameter


######################################################  FUNCTION  ######################################################
def title(string='', length=150, symbol='-'):
    if len(string) > 0:
        string = ' ' + string + ' '
    print('\n{0:{1}^{2}}'.format(string, symbol, length))


def create_directory(directory):
    if not os.path.exists(directory):
        print("Creating directory: {}".format(directory))
        os.makedirs(directory)


def get_time_elapsed(startTime, is_print=True):
    time_string = time.strftime('%H:%M:%S', time.gmtime(time.time() - startTime))
    if is_print:
        print('\nTime cost - {}'.format(time_string))
    return time_string


def create_supervised_dataset(arr, window=60, step=0, sequence=False, label_feature=0, make_x_ewma=False):
    arr_ewma = np.zeros(arr.shape)
    xlist = []
    ylist = []
    if make_x_ewma:
        for k in range(arr.shape[1]):  # Vertical
            # if np.unique(arr[:, k], axis=0) >= 20:
            arr_ewma[:, k] = exponential_weighted_moving_average_array(arr=arr[:, k], alpha=alpha)
        # for k in range(arr.shape[0]):  # Horizontal
        #     # if np.unique(arr[k, :], axis=0) >= 20:
        #     arr_ewma[k, :] = exponential_weighted_moving_average_array(arr=arr[k, :], alpha=alpha)
    for j in range(arr.shape[1]):
        x = []
        y = []
        for i in range(window, arr.shape[0] - step):
            if sequence:
                x.append(arr_ewma[i - window:i, j] if make_x_ewma else arr[i - window:i, j])
                y.append(arr_ewma[i:i + step + 1, j] if make_x_ewma else arr[i:i + step + 1, j])
            else:
                x.append(arr_ewma[i - window:i, j] if make_x_ewma else arr[i - window:i, j])
                y.append(arr_ewma[i + step, j] if make_x_ewma else arr[i + step, j])
        xArray = np.array(x)
        yArray = np.array(y)
        xlist.append(xArray)
        ylist.append(yArray)
    xOutput = np.concatenate(xlist, axis=1)
    yOutput = np.array(ylist[label_feature]).reshape(-1, step + 1 if sequence else 1)
    return xOutput, yOutput


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)  # ME
    mae = np.mean(np.abs(forecast - actual))  # MAE
    mpe = np.mean((forecast - actual) / actual)  # MPE
    rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE
    corr = np.corrcoef(forecast, actual)[0, 1]  # corr
    mins = np.amin(np.hstack([forecast[:, None], actual[:, None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:, None], actual[:, None]]), axis=1)
    minmax = 1 - np.mean(mins / maxs)  # minmax
    acf1 = acf(rolling_forecast_array - rolling_test_array)[1] if FORECAST_STEP > 1 else np.nan  # ACF1
    return ({'mape': mape, 'me': me, 'mae': mae,
             'mpe': mpe, 'rmse': rmse, 'acf1': acf1,
             'corr': corr, 'minmax': minmax})


def moving_average_df(df, field, window):
    outputField = '{}_MA'.format(field.upper())
    df[outputField] = df[field].rolling(window=window, min_periods=1).mean()  # The purpose of min_periods=1 is to keep the very first value
    return outputField


def moving_average_array(arr, window):
    return np.convolve(arr, np.ones(window), 'valid') / window


def exponential_weighted_moving_average_df(df, field, alpha):
    outputField = '{}_EWMA'.format(field.upper())
    df[outputField] = df[field].ewm(alpha=alpha, min_periods=1).mean()  # The purpose of min_periods=1 is to keep the very first value, but ewm seems to have default value of min_periods=1
    return outputField


def exponential_weighted_moving_average_array(arr, alpha):
    '''
    Returns the exponentially weighted moving average of x.

    Parameters:
    -----------
    x : array-like
    alpha : float {0 <= alpha <= 1}

    Returns:
    --------
    ewma: numpy array
          the exponentially weighted moving average
    '''
    # Coerce x to an array
    arr = np.array(arr)
    n = arr.size

    # Create an initial weight matrix of (1-alpha), and a matrix of powers
    # to raise the weights by
    w0 = np.ones(shape=(n, n)) * (1 - alpha)
    p = np.vstack([np.arange(i, i - n, -1) for i in range(n)])

    # Create the weight matrix
    w = np.tril(w0 ** p, 0)

    # Calculate the ewma
    return np.dot(w, arr[::np.newaxis]) / w.sum(axis=1)


######################################################  PROGRAM START  ######################################################
startTime = time.time()
# startTime = time.perf_counter()

title(string='PROGRAM START', length=150, symbol='-')

####################    DATA    ####################
print("Reading csv file ({}) ...".format(CSV_FILE))
df = pd.read_csv('csv/{}'.format(CSV_FILE))
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df.index = df['Date']
    df.drop(labels='Date', axis=1, inplace=True)
df.sort_index(ascending=True, axis=0, inplace=True)
startDate_string = df.index[0].strftime('%Y%m%d')
endDate_string = df.index[-1].strftime('%Y%m%d')
startDate = datetime.datetime.strptime(startDate_string, '%Y%m%d')
endDate = datetime.datetime.strptime(endDate_string, '%Y%m%d')
# print(df.head().to_string())
# print(len(df))
# plt.figure(figsize=(16, 8))
# plt.plot(df['Close'], label='Close Price history')
# plt.show()

####################    EXOGENOUS_FEATURE    ####################
df['Date'] = df.index
df['IsMondayFriday'] = df['Date'].apply(lambda dt: 1 if dt.dayofweek in [0, 4] else 0)
df.drop(labels='Date', axis=1, inplace=True)

title(string='FEATURES LIST', length=25, symbol='-')
print('{} (RESPONSE FIELD)'.format(FEATURE_LIST[0]))
for f in FEATURE_LIST[1:]:
    print(f)

####################    PREDICTION INDEX    ####################
prediction_index_list = []
# weekends = [6, 7]
for dt in pd.date_range(endDate + relativedelta(days=1), endDate + relativedelta(days=2 * FORECAST_STEP)):
    # print(type(dt))  # <class 'pandas._libs.tslibs.timestamps.Timestamp'>
    # print(type(dt.to_pydatetime()))  # <class 'datetime.datetime'>
    # if dt.isoweekday() not in weekends:
    if dt.dayofweek <= 4 and dt not in us_holidays:  # Remove weekends and public holidays
        # print(dt.strftime("%Y-%m-%d"))
        prediction_index_list.append(dt.to_pydatetime().date())
        if len(prediction_index_list) == FORECAST_STEP:
            break
# for x in prediction_index_list:
#     print(x)
# exit()

####################    TRAIN DATASET    ####################
title(string='TRAIN TEST SPLIT', length=25, symbol='-')
df_feature = df[FEATURE_LIST].copy()
feature_array = df_feature.values
df_train, df_test = train_test_split(df_feature, train_size=TRAIN_TO_TEST_RATIO, shuffle=False).copy()
train_array = df_train.values
test_array = df_test.values
print('Train dataset: {} rows'.format(len(train_array)))
print('Test dataset: {} rows'.format(len(test_array)))

x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))
# x_scaler = StandardScaler()
# y_scaler = StandardScaler()

X_train, y_train = create_supervised_dataset(train_array, window=LAGS_NUM, step=FORECAST_STEP - 1, sequence=True, label_feature=0, make_x_ewma=MAKE_X_EWMA)
if train_array.shape[1] > 1:
    X_train_scaled = x_scaler.fit_transform(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
else:
    X_train_scaled = x_scaler.fit_transform(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
y_train_scaled = y_scaler.fit_transform(y_train).reshape(X_train_scaled.shape[0], y_train.shape[1], 1)

####################    TEST DATASET    ####################
inputs_array = feature_array[len(feature_array) - len(test_array) - LAGS_NUM:]
X_test, y_test = create_supervised_dataset(inputs_array, window=LAGS_NUM, step=FORECAST_STEP - 1, sequence=True, label_feature=0, make_x_ewma=MAKE_X_EWMA)

if test_array.shape[1] > 1:
    X_test_scaled = x_scaler.transform(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])
else:  # test_array.shape[1] == 1
    X_test_scaled = x_scaler.transform(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
y_test_scaled = y_scaler.transform(y_test).reshape(y_test.shape[0], y_test.shape[1], 1)

####################    MODEL   ####################
title(string='MODEL', length=150, symbol='-')

model = Sequential()
model.add(LSTM(units=100, activation='relu', input_shape=((X_train_scaled.shape[1], X_train_scaled.shape[2]) if X_train_scaled.shape[2] != 1 else (X_train_scaled.shape[1], 1)), recurrent_dropout=0))
model.add(RepeatVector(y_train_scaled.shape[1]))
model.add(LSTM(units=50, activation='relu', return_sequences=True, recurrent_dropout=0))
model.add(TimeDistributed(Dense(units=1)))  # activation='linear'

model.compile(loss='mean_squared_error', optimizer='adam')  # loss='mae'
model.fit(X_train_scaled, y_train_scaled, epochs=EPOCHS, batch_size=1, verbose=0)  # verbose=2

####################    FORECAST    ####################
predicted_specific_step_array_scaled = model.predict(X_test_scaled)
predicted_specific_step_array = y_scaler.inverse_transform(predicted_specific_step_array_scaled.reshape(predicted_specific_step_array_scaled.shape[0], predicted_specific_step_array_scaled.shape[1]))  # For example, reshape from (361, 1, 1) to (361, 1)
predicted_specific_step_array_smoothed = np.zeros(predicted_specific_step_array.shape)
for i in range(predicted_specific_step_array.shape[0]):
    predicted_specific_step_array_smoothed[i] = lowess(predicted_specific_step_array[i], range(predicted_specific_step_array.shape[1]), frac=LOWESS_FRACTION)[:, 1].reshape(predicted_specific_step_array[i].shape)

####################    ACCURACY METRICS   ####################
title(string='ACCURACY METRICS (PREDICTION VS ACTUAL)', length=150, symbol='-')
print("Remark: The first few predictions might be more accurate then the others relatively, it is because no walking forward update of the model have been taken place, and this work is just a demonstration only.")

test_index_list = list(df_test.index)
forecast_accuracy_dict = {}
metrics_name_list = ['mape', 'corr', 'minmax']
for metrics_name in metrics_name_list:
    forecast_accuracy_dict[metrics_name] = []
for i, arr in enumerate(predicted_specific_step_array_smoothed if SMOOTH_PREDICTION else predicted_specific_step_array):
    print('\n*** Prediction of {0} on date {1} (i.e. Day T) ***'.format(FEATURE_LIST[0], df_feature.index[len(train_array) + i - 1].date()))
    rolling_test_index_list = test_index_list[i:i + FORECAST_STEP]
    rolling_test_array = test_array[i:i + FORECAST_STEP, 0]
    rolling_forecast_array = arr

    #####   PREDICTION VS ACTUAL    #####
    df_result = pd.DataFrame({'Day': ['T+{}'.format(i + 1) for i in range(FORECAST_STEP)]
                                 , 'Actual': rolling_forecast_array
                                 , 'Prediction': rolling_test_array}, index=rolling_test_index_list)
    print(df_result.to_string())

    #####  ACCURACY METRICS  #####
    # The commonly used accuracy metrics to judge forecasts are:
    # Mean Absolute Percentage Error (MAPE)
    # Mean Error (ME)
    # Mean Absolute Error (MAE)
    # Mean Percentage Error (MPE)
    # Root Mean Squared Error (RMSE)
    # Lag 1 Autocorrelation of Error (ACF1)
    # Correlation between the Actual and the Forecast (corr)
    # Min-Max Error (minmax)
    # Typically, if you are comparing forecasts of two different series, the MAPE, Correlation and Min-Max Error can be used.
    for k, v in forecast_accuracy(forecast=rolling_forecast_array, actual=rolling_test_array).items():
        if k in metrics_name_list:
            print('{0}: {1}'.format(k, v))
            forecast_accuracy_dict[k].append(v)

    #####    PLOT   #####
    if PLOT:
        fig = plt.figure()
        fig.suptitle(TICKER, fontsize=20)
        # plt.plot(df_train.index, df_train['Close'], label="Train Close")
        plt.plot(pd.Series(rolling_test_index_list), rolling_test_array, label="Test Close", marker='o' if len(rolling_test_array) == 1 else '')
        plt.plot(pd.Series(rolling_test_index_list), rolling_forecast_array, label="Prediction Close", marker='o' if len(rolling_forecast_array) == 1 else '')
        plt.title('Prediction of {0} on {1}'.format(FEATURE_LIST[0], df_feature.index[len(train_array) + i - 1].date()))
        plt.legend(loc="upper left")
        plt.show()

    if i == (len(test_array) - FORECAST_STEP):
        break

title(string='AVERAGE ACCURACY METRICS', length=150, symbol='-')
for metrics_name in metrics_name_list:
    print('Average {0}: {1}'.format(metrics_name, np.average(forecast_accuracy_dict[metrics_name])))

get_time_elapsed(startTime)
