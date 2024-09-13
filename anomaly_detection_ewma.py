import numpy as np
import matplotlib.pyplot as plt

# data stream generation with random anamolies
def generate_data_stream(n_points=1000, anomaly_chance=0.01):
    time = np.arange(n_points)
    seasonal_pattern = 10 * np.sin(2 * np.pi * time / 50)   # main data is based of a sine graph with trivial change as data wont actually always follow the expected curve
    noise = np.random.normal(0, 2, n_points)      # the obvious fluctuations         
    data_stream = seasonal_pattern + noise
    
    anomalies = np.random.choice([0, 1], size=n_points, p=[1 - anomaly_chance, anomaly_chance])
    data_stream += anomalies * np.random.uniform(20, 40, n_points)   # high spike anomalies added to data
    
    return data_stream, seasonal_pattern

def ewma_anomaly_detection(data_stream, alpha=0.1, threshold_factor=1):
    ewma = [data_stream[0]]
    residuals = []
    anomaly_flags = [False] * len(data_stream)

    for i in range(1, len(data_stream)):
        ewma.append(alpha * data_stream[i] + (1 - alpha) * ewma[-1])
        residual = abs(data_stream[i] - ewma[-1])
        residuals.append(residual)

    residuals = np.array(residuals)
    median_residual = np.median(residuals)
    mad = np.median(np.abs(residuals - median_residual))
    threshold = median_residual + threshold_factor * mad

    anomaly_flags[1:] = [residual > threshold for residual in residuals]

    return ewma, anomaly_flags

def real_time_visualization(data_stream, expected_data, anomaly_flags, ewma):
    plt.ion() 
    fig, ax = plt.subplots()
    for i in range(len(data_stream)):
        ax.clear()
        ax.plot(expected_data[:i+1], label='Expected Data', linestyle='--', color='blue')
        ax.plot(data_stream[:i+1], label='Data Stream', color='black')
        ax.plot(range(1, i + 1), ewma[:i], label='EWMA', color='green')
        anomalies = np.where(np.array(anomaly_flags[:i+1]) == True)[0]
        ax.scatter(anomalies, data_stream[anomalies], color='red', label='Anomalies')
        ax.legend()
        plt.pause(0.01)
    plt.ioff() 
    plt.show()

if __name__ == "__main__":
    data, expected_data = generate_data_stream(500)
    ewma, flags = ewma_anomaly_detection(data)
    real_time_visualization(data, expected_data, flags, ewma)
