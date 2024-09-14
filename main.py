import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def autoencoder_anomaly_detection(data_stream, epochs=100, batch_size=32):
    # Scale the data
    scaler = MinMaxScaler()
    data_stream_scaled = scaler.fit_transform(np.array(data_stream).reshape(-1, 1))

    # Define the autoencoder model
    model = Sequential([
        Dense(64, input_dim=1, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder
    model.fit(data_stream_scaled, data_stream_scaled, epochs=epochs, batch_size=batch_size, verbose=0)

    # Predict the data and compute reconstruction error
    reconstructed = model.predict(data_stream_scaled)
    reconstruction_error = np.mean(np.square(data_stream_scaled - reconstructed), axis=1)

    # Set the threshold dynamically based on the 95th percentile of reconstruction errors
    threshold = np.percentile(reconstruction_error, 95)
    anomalies = reconstruction_error > threshold

    # Return results
    return reconstruction_error, anomalies

# Data stream simulation function
def generate_data_stream(n_points=1000, anomaly_chance=0.05):
    time = np.arange(n_points)
    seasonal_pattern = 10 * np.sin(2 * np.pi * time / 50)  # Seasonal component
    noise = np.random.normal(0, 2, n_points)               # Random noise
    data_stream = seasonal_pattern + noise
    
    # Introduce anomalies with higher magnitude
    anomalies = np.random.choice([0, 1], size=n_points, p=[1 - anomaly_chance, anomaly_chance])
    data_stream += anomalies * np.random.uniform(15, 50, n_points)  # Increase anomaly magnitude
    
    return data_stream, seasonal_pattern

# Visualization function
def real_time_visualization(data_stream, expected_data, anomaly_flags):
    plt.ion()  # Turn interactive mode on
    fig, ax = plt.subplots()
    for i in range(len(data_stream)):
        ax.clear()
        ax.plot(expected_data[:i+1], label='Expected Data', linestyle='--', color='blue')
        ax.plot(data_stream[:i+1], label='Data Stream', color='black')
        anomalies = np.where(np.array(anomaly_flags[:i+1]) == True)[0]
        ax.scatter(anomalies, data_stream[anomalies], color='red', label='Anomalies')
        ax.legend()
        plt.pause(0.01)
    plt.ioff()  # Turn off interactive mode
    plt.show()

# Test the functionality by generating and visualizing data
if __name__ == "__main__":
    data, expected_data = generate_data_stream(500)
    reconstruction_error, flags = autoencoder_anomaly_detection(data)
    real_time_visualization(data, expected_data, flags)
