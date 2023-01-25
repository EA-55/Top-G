import logging
import time

import ccxt
import numpy as np
import pandas as pd
import pyti
import talib
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def connect_to_exchange(exchange_id):
    try:
      
        exchange.load_markets()
        return exchange
    except Exception as e:
        print("Error connecting to exchange: ", e)




logging.basicConfig(filename='bot.log', level=logging.INFO, format='%(asctime)s %(message)s')

def fetch_ohlcv_data(exchange_id, symbol, timeframe, periods):
    try:
        exchange = connect_to_exchange(exchange_id)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    except Exception as e:
        print("Error connecting to exchange: {}".format(e))
        return None
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    try:
        file_name = '{}_{}_ohlcv.json'.format(symbol, timeframe)
        with open(file_name, 'w') as f:
            df.to_json(f)
    except Exception as e:
        print("Error saving data to json: {}".format(e))

    return df

def scale_and_train_nn(df):
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    # Split the data into training and testing sets
    X = df_scaled[:, :-1]
    y = df_scaled[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Define and train the neural network
    nn = MLPRegressor()
    nn.fit(X_train, y_train)
    return nn, scaler

exchange = ccxt.binance()
exchange.load_markets()
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d')

df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df.to_csv('ohlcv_data.csv',index=False)

def calculate_indicators(df):
    df.to_csv('ohlcv_data.csv',index=False)
    if set(['close', 'high', 'low']).issubset(df.columns):
        close = df['close'].values
        df.dropna(inplace=True)
        if df.empty:
            return pd.DataFrame()
        df.fillna(df.mean(), inplace=True)
        scaler = MinMaxScaler()
        df_normalized = scaler.fit_transform(df)
        rsi = talib.RSI(close, timeperiod=14)
        df['rsi'] = rsi
        df['ma5'] = talib.SMA(close, timeperiod=5)
        df['ma10'] = talib.SMA(close, timeperiod=10)
        df['ma20'] = talib.SMA(close, timeperiod=20)
        upper_band, middle_band, lower_band = talib.BBANDS(close)
        df['upper_band'] = upper_band
        df['middle_band'] = middle_band
        df['lower_band'] = lower_band
        high = df['high'].values
        low = df['low'].values
        df['fib_retracement'] = pyti.fibonacci_retracement(high, low, [0, 23.6, 38.2, 50, 61.8, 100])
        macd, macd_signal, macd_hist = talib.MACD(close)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        df['adx'] = talib.ADX(high, low, close)
        df['stochastic_oscillator_k'], df['stochastic_oscillator_d'] = talib.STOCH(df['high'].values, df['low'].values, df['close'].values, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        print("File created.")
        return df
    else:
        return pd.DataFrame()

# Load data
df = calculate_indicators(df)


# Split the data into inputs and outputs
X = df[['open', 'high', 'low', 'close', 'volume']]
y = df['close'].values
# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# Normalize the data
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


data = pd.read_csv('stock_data.csv')
X_train, X_test, y_train, y_test = train_test_split(data.drop('output_column', axis=1), data['output_column'], test_size=0.2, random_state=0)

# Initialize the NeuralNetwork
class NeuralNetwork:
    def __init__(self):
        # Define the model
        self.model = Sequential()
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=10)
    def evaluate(self, X_test, y_test):
        # evaluate the model on the test data
        return self.model.evaluate(X_test, y_test)

nn = NeuralNetwork()
nn.fit(X_train, y_train)
test_loss, test_acc = nn.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Load and check the data
data = pd.read_csv('stock_data.csv')
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Split data into inputs and outputs
X = data.drop('output_column', axis=1)
y = data['output_column']

# Preprocess the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Cross-validation
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=0)

# Define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(20, input_shape=(X.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
param_grid = {'batch_size': [32, 64, 128],
              'epochs': [50, 100, 200]}
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold)
grid_result = grid.fit(X, y)

# Print the best parameters and the corresponding score
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Train the model on the full dataset
model.fit(X, y)



test_days = 90
n_folds = 10
stop_loss = 0.1
initial_balance = 1000

# Create a KFold object for cross-validation
kf = KFold(n_splits=n_folds, shuffle=True)

# Initialize a list to store the results of each fold
results = []

# Loop through the folds
for train_index, test_index in kf.split(X_normalized):
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model on the training data
    nn.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = nn.predict(X_test)

    # Initialize a variable to store the total profit for this fold
    total_profit = 0

    # Initialize the current date and balance
    current_date = df_test.index[0]
    balance = initial_balance

    for i in range(test_days):
        # Calculate indicators for the current date
        df_current = calculate_indicators(df_test.loc[current_date])
        # Make a prediction for the current date
        prediction = nn.predict(scaler.transform(df_current))
        # Check if the prediction is a buy or sell signal
        if prediction == 1:
            # Buy
            # Check if enough funds are available
            if balance >= df_current['close'].iloc[-1]:
                # Calculate the amount of shares that can be bought
                shares = balance // df_current['close'].iloc[-1]
                # Update the balance
                balance -= shares * df_current['close'].iloc[-1]
                # Add the details of the new position to the open_positions dictionary
                open_positions[i] = {'timestamp': current_date, 'price': df_current['close'].iloc[-1], 'shares': shares}
                # Log the trade
                log_trade({'timestamp': current_date, 'price': df_current['close'].iloc[-1], 'amount': shares}, symbol,
                          'buy')
        elif prediction == -1:
            # Sell
            # Loop through the open positions
            for pos_id, pos_details in open_positions.items():
                # Sell the shares
                balance += pos_details['shares'] * df_current['close'].iloc[-1]
                # Log the trade
                log_trade({'timestamp': current_date, 'id': pos_id, 'price': df_current['close'].iloc[-1],
                           'amount': pos_details['shares']}, symbol, 'sell')
                # Remove the position from the open_positions dictionary
                del open_positions[pos_id]

                # Update the current date
                current_date += pd.Timedelta(1, 'D')
                # Calculate the total profit for this fold
                total_profit = balance - initial_balance
                # Append the total profit to the results list
                results.append(total_profit)

                # Print the average profit across all folds
                print("Average profit:", np.mean(results))

                # Calculate the accuracy of the model
                accuracy = accuracy_score(y_test, predictions)
                print("Accuracy:", accuracy)

                # Calculating the mean squared error
                mse = mean_squared_error(y_test, predictions)
                print("Mean Squared Error:", mse)

                # Printing the classification report
                print(classification_report(y_test, predictions))

                # Printing the confusion matrix
                print(confusion_matrix(y_test, predictions))



def log_trade(trade_data, symbol, trade_type):
    # Create a dataframe from the trade data
    trade_df = pd.DataFrame(trade_data, index=[0])
    # Add a column for the symbol and trade type
    trade_df['symbol'] = symbol
    trade_df['trade_type'] = trade_type
    # Append the data to the trades.csv file
    trade_df.to_csv('trades.csv', mode='a', header=False)


def trading_strategy(exchange, symbol, df, nn, scaler, stop_loss=0.05):
    prediction = nn.predict(scaler.transform(df))
    fundamental_decision = fundamental_analysis(symbol)
    if df['rsi'].iloc[-1] < 30 and df['ma5'].iloc[-1] > df['ma10'].iloc[-1] > df['ma20'].iloc[-1] and \
            df['close'].iloc[-1] < df['lower_band'].iloc[-1] and df['fib_retracement'].iloc[-1] < 50 and \
            df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] and buy_or_sell == "buy":
        try:
            # Place a buy order
            order = exchange.create_order(symbol, 'limit', 'buy', df['close'].iloc[-1], 1)
            logging.info('Buy order placed: ' + str(order))
            # Place a stop loss order
            stop_loss_order = exchange.create_order(symbol, 'stop_loss', 'sell', df['close'].iloc[-1] * (1 - stop_loss),
                                                    1)
            logging.info('Stop loss order placed: ' + str(stop_loss_order))
        except Exception as e:
            logging.error('Error placing buy order: ' + str(e))
    elif df['rsi'].iloc[-1] > 70 and df['ma5'].iloc[-1] < df['ma10'].iloc[-1] < df['ma20'].iloc[-1] and \
            df['close'].iloc[-1] > df['upper_band'].iloc[-1] and df['fib_retracement'].iloc[-1] > 50 and \
            df['macd'].iloc[-1] < df['macd_signal'].iloc[-1] and buy_or_sell == "sell":
        try:
            # Place a sell order
            order = exchange.create_order(symbol, 'limit', 'sell', df['close'].iloc[-1], 1)
            logging.info('Sell order placed: ' + str(order))
            # Place a stop loss order
            stop_loss_order = exchange.create_order(symbol, 'stop_loss', 'buy', df['close'].iloc[-1] * (1 + stop_loss),
                                                    1)
            logging.info('Stop loss order placed: ' + str(stop_loss_order))
        except Exception as e:
            logging.error('Error placing sell order: ' + str(e))
        else:
            logging.info('No trading opportunity found')

    def main():
        exchange_id = 'binance'
        symbol = 'BTC/USDT'
        timeframe = '1d'
        periods = 100

        exchange = connect_to_exchange(exchange_id)
        df = fetch_ohlcv_data(exchange, symbol, timeframe, periods)
        df = calculate_indicators(df)

        nn = MLPRegressor()
        scaler = MinMaxScaler()
        kf = KFold(n_splits=10)

        trading_strategy(exchange, symbol, df, nn, scaler)

    if __name__ == '__main__':main()


def trading_strategy(exchange, symbol, df, nn, scaler, stop_loss=0.05):
    prediction = nn.predict(scaler.transform(df))
    if df['rsi'].iloc[-1] < 30 and df['ma5'].iloc[-1] > df['ma10'].iloc[-1] > df['ma20'].iloc[-1] and \
            df['close'].iloc[-1] < df['lower_band'].iloc[-1] and df['fib_retracement'].iloc[-1] < 50 and \
            df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] and prediction > 0.8:
        try:
            # Place a buy order
            order = exchange.create_order(symbol, 'limit', 'buy', df['close'].iloc[-1], 1)
            logging.info('Buy order placed: ' + str(order))
            # Place a stop loss order
            stop_loss_order = exchange.create_order(symbol, 'stop_loss', 'sell', df['close'].iloc[-1] * (1-stop_loss), 1)
            logging.info('Stop loss order placed: ' + str(stop_loss_order))
        except Exception as e:
            logging.error('Error placing buy order: ' + str(e))
    elif df['rsi'].iloc[-1] > 70 and df['ma5'].iloc[-1] < df['ma10'].iloc[-1] < df['ma20'].iloc[-1] and \
            df['close'].iloc[-1] > df['upper_band'].iloc[-1] and df['fib_retracement'].iloc[-1] > 50 and \
            df['macd'].iloc[-1] < df['macd_signal'].iloc[-1] and prediction < 0.2:
        try:
            # Place a sell order
            order = exchange.create_order(symbol, 'limit', 'sell', df['close'].iloc[-1], 1)
            logging.nfo('Sell order placed: ' + str(order))
            # Place a stop loss order
            stop_loss_order = exchange.create_order(symbol, 'stop_loss', 'buy', df['close'].iloc[-1] * (1+stop_loss), 1)
            logging.info('Stop loss order placed: ' + str(stop_loss_order))
        except Exception as e:
            logging.error('Error placing sell order: ' + str(e))
        else:logging.info('No trading opportunity found')

def manage_risk(exchange, symbol, nn, scaler, stop_loss=0.05, max_loss=1000):
    stop_trading = False
    balance = exchange.fetch_balance()
    while not stop_trading:
        updated_balance = exchange.fetch_balance()
        current_loss = balance - updated_balance
        if current_loss > max_loss:
            logging.warning(f"Maximum loss of {max_loss} reached. Closing all positions and pausing trading.")
            close_all_positions(exchange, symbol)
            stop_trading = True
        else:
            time.sleep(3600)

# This function can be called to stop the trading loop
def stop_trading():
    stop_trading = True


class RiskManager:
    def __init__(self, exchange, symbol, stop_loss=0.05, max_loss=1000):
        self.exchange = exchange
        self.symbol = symbol
        self.stop_loss = stop_loss
        self.max_loss = max_loss
        self.is_paused = False

    def pause_trading(self):
        self.is_paused = True
        logging.warning("Trading is now paused.")

    def continue_trading(self):
        self.is_paused = False
        logging.info("Trading has been resumed.")

    def close_all_positions(self):
        try:
            open_positions = self.exchange.fetch_open_orders(self.symbol)
            for position in open_positions:
                self.exchange.cancel_order(position['id'])
            logging.info("All open positions have been closed.")
        except Exception as e:
            logging.error("Error closing open positions: " + str(e))

    def monitor_risk(self):
        balance = self.exchange.fetch_balance()
        while True:
            if self.is_paused:
                time.sleep(3600)
                continue

            updated_balance = self.exchange.fetch_balance()
            current_loss = balance - updated_balance
            if current_loss > self.max_loss:
                logging.warning(f"Maximum loss of {self.max_loss} reached. Closing all positions and pausing trading.")
                self.close_all_positions()
                self.pause_trading()
            else:
                time.sleep(3600)


def test_strategy(exchange, symbol_list, timeframes):
    results = {}
    for symbol in symbol_list:
        for timeframe in timeframes:
            df = fetch_ohlcv_data(exchange, symbol, timeframe)
            df = calculate_indicators(df)
            nn, scaler = scale_and_train_nn(df)
            performance = trading_strategy(exchange, symbol, df, nn, scaler)
            risk_management = RiskManagement(exchange, symbol, nn, scaler)
            risk_management.monitor_risk()
            results[symbol + '-' + timeframe] = performance
    return results



def backtest_strategy(exchange, symbol, start_date, end_date, nn, scaler):
    historical_data = fetch_historical_data(exchange, symbol, start_date, end_date)
    historical_data = calculate_indicators(historical_data)
    performance = evaluate_performance(historical_data, nn, scaler)
    return performance
def backtest(exchange, symbol, nn, scaler, periods=100, start_date=None, end_date=None):
    data = fetch_ohlcv_data(exchange, symbol, '1d', periods, start_date, end_date)
    data = calculate_indicators(data)
    predictions = nn.predict(scaler.transform(data))
    backtest_results = evaluate_performance(data, predictions)
    return backtest_results

def optimize_parameters(exchange, symbol, nn, scaler, periods=100, start_date=None, end_date=None):
    best_params = None
    best_performance = None
    for param_set in generate_parameter_combinations():
        updated_nn, updated_scaler = scale_and_train_nn(param_set)
        results = backtest(exchange, symbol, updated_nn, updated_scaler, periods, start_date, end_date)
        if not best_performance or results > best_performance:
            best_params = param_set
            best_performance = results
    return best_params


def optimize_parameters(exchange, symbol, nn, scaler, periods):
    # Define the parameters to optimize
    parameters = {'hidden_layer_sizes': [(100,), (50,50), (20,20,20)],
                  'activation': ['relu', 'tanh'],
                  'solver': ['adam', 'sgd']}
    # Use GridSearchCV to optimize the parameters
    clf = GridSearchCV(nn, parameters)
    clf.fit(scaler.transform(fetch_ohlcv_data(exchange, symbol, '1d', periods)), return_train_score=True)
    best_params = clf.best_params_
    nn.set_params(**best_params)
    return nn


def evaluate_performance(df, nn, scaler):
    df['prediction'] = nn.predict(scaler.transform(df))
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = np.where(df['prediction'] > 0.8, df['returns'], -df['returns'])
    sharpe_ratio = calculate_sharpe_ratio(df['strategy_returns'])
    return sharpe_ratio

def calculate_sharpe_ratio(returns):
    mean_return = returns.mean()
    std_dev = returns.std()
    sharpe_ratio = mean_return / std_dev
    return sharpe_ratio
def monitor_performance(results):
    while True:
        updated_results = test_strategy(exchange, symbol_list, timeframes)
        for symbol, performance in updated_results.items():
            if performance < results[symbol]:
                logging.warning(f"Performance for {symbol} has decreased. Reviewing strategy...")
                # Review and adjust strategy
                new_nn, new_scaler = scale_and_train_nn(df)
                results[symbol] = backtest_strategy(exchange, symbol, start_date, end_date, new_nn, new_scaler)
        time.sleep(3600) # check performance every hour

def monitor_performance(exchange, symbol, nn, scaler, periods=100, interval=3600):
    while True:
        current_data = fetch_ohlcv_data(exchange, symbol, '1d', periods)
        current_data = calculate_indicators(current_data)
        predictions = nn.predict(scaler.transform(current_data))
        performance = evaluate_performance(current_data, predictions)
        if performance < desired_performance:
            updated_params = optimize_parameters(exchange, symbol, nn, scaler, periods)
            nn, scaler = scale_and_train_nn(updated_params)
            logging.info(f"Trading strategy performance improved to {performance} with new parameters {updated_params}")
        time.sleep(interval)




while True:
    # fetch latest market data
    df = fetch_ohlcv_data(exchange_id, symbol, timeframe, periods)
    # calculate indicators
    df = calculate_indicators(df)
    # check current market conditions
    if df['rsi'].iloc[-1] < 30 and df['ma5'].iloc[-1] > df['ma10'].iloc[-1] > df['ma20'].iloc[-1] and \
            df['close'].iloc[-1] < df['lower_band'].iloc[-1] and df['fib_retracement'].iloc[-1] < 50 and \
            df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
        # execute trade
        trading_strategy(exchange, symbol, df, nn, scaler, stop_loss=0.05)
    # wait for 5 minutes before checking market conditions again
    time.sleep(300)

