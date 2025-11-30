import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. Load the Dataset
file_path = 'rain_forecasting.csv'
df = pd.read_csv(file_path)

# 2. Preprocessing
# Convert Date to datetime and sort (crucial for time series)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# Encode Binary Categorical Columns (Yes/No -> 1/0)
le = LabelEncoder()
df['RainToday'] = le.fit_transform(df['RainToday'])
df['RainTomorrow'] = le.fit_transform(df['RainTomorrow'])

# One-Hot Encode 'Location' (converts location names to separate binary columns)
df = pd.get_dummies(df, columns=['Location'])

# Select Features and Target
# Drop 'Date' as it's not a direct input feature after sorting
# Drop 'RainTomorrow' from inputs as it's the target
features = df.drop(['Date', 'RainTomorrow'], axis=1)
target = df['RainTomorrow']

# Scale the Features (LSTM works best with data scaled between 0 and 1)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# 3. Create Sequences for LSTM
# We use a sliding window approach: use past 'n_steps' days to predict the next day
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 14  # Use past 2 weeks to predict tomorrow
X, y = create_sequences(features_scaled, target, TIME_STEPS)

# 4. Train/Test Split
# Since it's a time series, we cannot shuffle. We split by time.
split_index = int(len(X) * 0.8)  # 80% for training
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

# 5. Build the LSTM Model
model = Sequential()
# First LSTM layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
# Second LSTM layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
# Output layer (1 unit for binary classification)
model.add(Dense(units=1, activation='sigmoid'))



model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 6. Train the Model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=True)],
    verbose=1
)

# 7. Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Plotting history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

model.save('rain_forecast_lstm.h5')