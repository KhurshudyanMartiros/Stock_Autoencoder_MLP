import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import talib

# Function to calculate ZigZag indicator using iloc for positional indexing
def zigzag(data, pct_change=5):
    zigzag = [0] * len(data)
    start = data.iloc[0]
    trend = None
    
    for i in range(1, len(data)):
        change = (data.iloc[i] - start) / start * 100
        
        if trend is None:
            if abs(change) > pct_change:
                trend = 'peak' if change > 0 else 'trough'
                zigzag[i] = data.iloc[i]
                start = data.iloc[i]
        elif trend == 'peak':
            if change < -pct_change:
                trend = 'trough'
                zigzag[i] = data.iloc[i]
                start = data.iloc[i]
            elif data.iloc[i] > zigzag[i-1]:
                zigzag[i-1] = 0
                zigzag[i] = data.iloc[i]
                start = data.iloc[i]
        elif trend == 'trough':
            if change > pct_change:
                trend = 'peak'
                zigzag[i] = data.iloc[i]
                start = data.iloc[i]
            elif data.iloc[i] < zigzag[i-1]:
                zigzag[i-1] = 0
                zigzag[i] = data.iloc[i]
                start = data.iloc[i]
    
    return pd.Series(zigzag, index=data.index)

# Fetch historical stock data for Apple
data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')

# Calculate 5-day, 10-day, and 25-day moving averages
data['MA5'] = data['Adj Close'].rolling(window=5).mean()
data['MA10'] = data['Adj Close'].rolling(window=10).mean()
data['MA25'] = data['Adj Close'].rolling(window=25).mean()

# Calculate MACD
ema12 = data['Adj Close'].ewm(span=12, adjust=False).mean()
ema26 = data['Adj Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema12 - ema26

# Calculate some common candlestick patterns
data['CDL_DOJI'] = talib.CDLDOJI(data['Open'], data['High'], data['Low'], data['Close'])
data['CDL_MORNING_STAR'] = talib.CDLMORNINGSTAR(data['Open'], data['High'], data['Low'], data['Close'], penetration=0.3)
data['CDL_EVENING_STAR'] = talib.CDLEVENINGSTAR(data['Open'], data['High'], data['Low'], data['Close'], penetration=0.3)
data['CDL_HAMMER'] = talib.CDLHAMMER(data['Open'], data['High'], data['Low'], data['Close'])
data['CDL_SHOOTING_STAR'] = talib.CDLSHOOTINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])

# Calculate ZigZag indicator
data['ZIGZAG'] = zigzag(data['Adj Close'])

# Drop rows with NaN values (at the beginning of the dataset where moving averages or candlestick patterns can't be computed)
data = data.dropna()

# Use the moving averages, MACD, candlestick patterns, ZigZag, and 'Adj Close' as features
features = data[['Adj Close', 'MA5', 'MA10', 'MA25', 'MACD', 'CDL_DOJI', 'CDL_MORNING_STAR', 'CDL_EVENING_STAR', 'CDL_HAMMER', 'CDL_SHOOTING_STAR', 'ZIGZAG']].values
targets = data['Adj Close'].values  # Next day's adjusted close price

# Normalize the features using MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Prepare the dataset: use the previous 5 days' features to predict the next day's adjusted close price
window_size = 15
X, y = [], []

for i in range(len(features) - window_size):
    X.append(features[i:i + window_size])
    y.append(targets[i + window_size])

X = np.array(X)
y = np.array(y)

# Reshape X to the shape (num_samples, num_columns)
num_columns = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], num_columns))

# Split the data into 70% training and 30% testing
split_index = int(len(X) * 0.7)

X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]

# Supervised Autoencoder MLP model
def create_ae_mlp(num_columns, hidden_units, dropout_rates, ls=1e-2, lr=1e-3):
    inp = tf.keras.layers.Input(shape=(num_columns,))
    x0 = tf.keras.layers.BatchNormalization()(inp)
    
    encoder = tf.keras.layers.GaussianNoise(dropout_rates[0])(x0)
    encoder = tf.keras.layers.Dense(hidden_units[0])(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Activation('relu')(encoder)
    
    decoder = tf.keras.layers.Dropout(dropout_rates[1])(encoder)
    decoder = tf.keras.layers.Dense(num_columns, name='decoder')(decoder)

    x_ae = tf.keras.layers.Dense(hidden_units[1])(decoder)
    x_ae = tf.keras.layers.BatchNormalization()(x_ae)
    x_ae = tf.keras.layers.Activation('relu')(x_ae)
    x_ae = tf.keras.layers.Dropout(dropout_rates[2])(x_ae)

    out_ae = tf.keras.layers.Dense(1, name='ae_action')(x_ae)
    
    x = tf.keras.layers.Concatenate()([x0, encoder])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rates[3])(x)
    
    for i in range(2, len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 2])(x)
        
    out = tf.keras.layers.Dense(1, name='action')(x)
    
    model = tf.keras.models.Model(inputs=inp, outputs=[decoder, out_ae, out])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss={'decoder': tf.keras.losses.MeanSquaredError(),
                        'ae_action': tf.keras.losses.MeanSquaredError(),
                        'action': tf.keras.losses.MeanSquaredError()},
                  metrics={'decoder': tf.keras.metrics.MeanAbsoluteError(name='MAE'),
                           'ae_action': tf.keras.metrics.MeanAbsoluteError(name='MAE'),
                           'action': tf.keras.metrics.MeanAbsoluteError(name='MAE')})
    
    return model

# Define model parameters
hidden_units = [512, 512, 256, 256, 128]
dropout_rates = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
lr = 1e-3

# Create the model
model = create_ae_mlp(num_columns, hidden_units, dropout_rates, lr=lr)

# Train the model
history = model.fit(X_train, {'decoder': X_train, 'ae_action': y_train, 'action': y_train},
                    epochs=250, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model
evaluation = model.evaluate(X_test, {'decoder': X_test, 'ae_action': y_test, 'action': y_test})

# Print the full evaluation results to understand the indices
print("Evaluation results:", evaluation)

# Extract evaluation metrics based on the length of the evaluation list
if len(evaluation) == 7:
    total_loss = evaluation[0]
    decoder_loss = evaluation[1]
    decoder_mae = evaluation[2]
    ae_action_loss = evaluation[3]
    ae_action_mae = evaluation[4]
    action_loss = evaluation[5]
    action_mae = evaluation[6]
elif len(evaluation) == 4:
    total_loss = evaluation[0]
    decoder_loss = evaluation[1]
    decoder_mae = evaluation[2]
    ae_action_loss = evaluation[3]
    ae_action_mae = None
    action_loss = None
    action_mae = None
else:
    print("Unexpected number of evaluation metrics:", len(evaluation))
    decoder_loss = None
    decoder_mae = None
    ae_action_loss = None
    ae_action_mae = None
    action_loss = None
    action_mae = None

if action_mae is not None:
    print(f"total_loss: {total_loss:.4f}, decoder_loss: {decoder_loss:.4f}, decoder_MAE: {decoder_mae:.4f}, ae_action_loss: {ae_action_loss:.4f}, ae_action_MAE: {ae_action_mae:.4f}, action_loss: {action_loss:.4f}, action_MAE: {action_mae:.4f}")
else:
    print(f"total_loss: {total_loss:.4f}, decoder_loss: {decoder_loss:.4f}, decoder_MAE: {decoder_mae:.4f}, ae_action_loss: {ae_action_loss:.4f}")

# Make predictions
y_pred = model.predict(X_test)[-1]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(data.index[split_index + window_size:], y_test, label='Actual')
plt.plot(data.index[split_index + window_size:], y_pred, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()

