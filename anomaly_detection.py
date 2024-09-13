import numpy as np
import matplotlib.pyplot as plt



# Data stream simulation function
def generate_data_stream(n_points=1000, anomaly_chance=0.01):
    time = np.arange(n_points)
    seasonal_pattern = 10 * np.sin(2 * np.pi * time / 50)  # Seasonal component
    noise = np.random.normal(0, 2, n_points)               # Random noise
    data_stream = seasonal_pattern + noise
    
    # Introduce anomalies with certain probability
    anomalies = np.random.choice([0, 1], size=n_points, p=[1 - anomaly_chance, anomaly_chance])
    data_stream += anomalies * np.random.uniform(20, 40, n_points)  # Spikes for anomalies
    
    return data_stream

# EWMA anomaly detection function
def moving_average(data_stream, window_size=5):
    return np.convolve(data_stream, np.ones(window_size) / window_size, mode='valid')

def ewma_anomaly_detection(data_stream, alpha=0.3, threshold_factor=3, window_size=5):
    smoothed_data = moving_average(data_stream, window_size)
    ewma = [smoothed_data[0]]
    residuals = []
    anomaly_flags = []

    for i in range(1, len(smoothed_data)):
        ewma.append(alpha * smoothed_data[i] + (1 - alpha) * ewma[-1])
        residual = abs(smoothed_data[i] - ewma[-1])
        residuals.append(residual)

    median_residual = np.median(residuals)
    mad = np.median(np.abs(residuals - median_residual))
    threshold = median_residual + threshold_factor * mad

    for residual in residuals:
        anomaly_flags.append(residual > threshold)

    return ewma, anomaly_flags


# Visualization function
def real_time_visualization(data_stream, anomaly_flags, ewma):
    plt.ion()  # Turn interactive mode on
    fig, ax = plt.subplots()
    for i in range(len(data_stream)):
        ax.clear()
        ax.plot(data_stream[:i+1], label='Data Stream')
        ax.plot(ewma[:i+1], label='EWMA')
        anomalies = np.where(np.array(anomaly_flags[:i+1]) == True)[0]
        ax.scatter(anomalies, data_stream[anomalies], color='red', label='Anomalies')
        ax.legend()
        plt.pause(0.01)
    plt.ioff()  # Turn off interactive mode
    plt.show()

# Test the functionality by generating and visualizing data
if __name__ == "__main__":
    data = generate_data_stream(500)
    ewma, flags = ewma_anomaly_detection(data)
    real_time_visualization(data, flags, ewma)
