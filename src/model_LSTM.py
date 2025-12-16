from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def model_LSTM(window_size, epochs, batch_size, X_train, y_train):
    """
    Build and train an LSTM model with the given parameters.
        Inputs:
            window_size: The size of the input window for the LSTM.
            epochs: Number of epochs to train the model.
            batch_size: Size of each training batch.
            X_train: Training input data.
            y_train: Training target data.
        Outputs:
            model_seq: The trained LSTM model.
    """

    model_seq = Sequential()
    model_seq.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))    
    model_seq.add(Dense(1))
    model_seq.compile(optimizer='adam', loss='mean_squared_error')

    model_seq.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    return model_seq
