import numpy as np
import matplotlib.pyplot as plt

def generate_exponential_data(n_points=350, anomaly_chance=0.01):
    time = np.arange(n_points)
    exponential_growth = np.exp(time / 100) 
    # noise must be proportional to streamming data in this case
    noise_magnitude = 0.5
    noise = np.random.normal(0, noise_magnitude * (1 + np.abs(exponential_growth / np.max(exponential_growth))), n_points)
    
    data_stream = exponential_growth + noise
    
    anomalies = np.random.choice([0, 1], size=n_points, p=[1 - anomaly_chance, anomaly_chance])
    anomaly_magnitude = np.random.uniform(10, 20, n_points)  # anomalies spikes
    
    data_stream += anomalies * anomaly_magnitude
    
    return data_stream, exponential_growth

#for exponential data, smoothing will be needed
def exponential_smoothing_anomaly_detection(data_stream, alpha=0.1, threshold=5):
    n = len(data_stream)
    smoothed_data = np.zeros(n)
    smoothed_data[0] = data_stream[0]  

    for i in range(1, n):
        smoothed_data[i] = alpha * data_stream[i] + (1 - alpha) * smoothed_data[i - 1]

    residuals = np.abs(data_stream - smoothed_data)
    anomaly_flags = residuals > threshold

    return smoothed_data, anomaly_flags

def real_time_visualization(data_stream, expected_data, anomaly_flags, smoothed_data):
    plt.ion() 
    fig, ax = plt.subplots()
    for i in range(len(data_stream)):
        ax.clear()
        ax.plot(expected_data[:i+1], label='Expected Data', linestyle='--', color='blue')
        ax.plot(data_stream[:i+1], label='Data Stream', color='black')
        ax.plot(smoothed_data[:i+1], label='Smoothed Data', color='green')
        anomalies = np.where(np.array(anomaly_flags[:i+1]) == True)[0]
        ax.scatter(anomalies, data_stream[anomalies], color='red', label='Anomalies')
        ax.legend()
        plt.pause(0.01)
    plt.ioff() 
    plt.show()

if __name__ == "__main__":
    data, expected_data = generate_exponential_data(350)
    smoothed_data, flags = exponential_smoothing_anomaly_detection(data)
    real_time_visualization(data, expected_data, flags, smoothed_data)
