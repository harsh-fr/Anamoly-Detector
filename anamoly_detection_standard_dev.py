import numpy as np
import matplotlib.pyplot as plt

# Streaming data generation
def generate_data_stream(n_points=1000, anomaly_chance=0.05):
    time = np.arange(n_points)
    seasonal_pattern = 10 * np.sin(2 * np.pi * time / 50)  
    noise = np.random.normal(0, 2, n_points)  
    data_stream = seasonal_pattern + noise
    
    # Add occasional anomalies
    anomalies = np.random.choice([0, 1], size=n_points, p=[1 - anomaly_chance, anomaly_chance])
    anomaly_magnitudes = np.random.uniform(-40, 40, n_points)  
    data_stream += anomalies * anomaly_magnitudes
    
    return data_stream, seasonal_pattern

def std_dev_anomaly_detection(data_stream, threshold_factor):
    residuals = []
    # Don't include anomalies in the calculation of standard deviations
    anomaly_flags = [False] * len(data_stream)
    
    for i in range(len(data_stream)):
        current_data = data_stream[:i+1]
        mean = np.mean(current_data)
        std_dev = np.std(current_data)
        
        residual = abs(data_stream[i] - mean)
        residuals.append(residual)
        threshold = threshold_factor * std_dev

        # Identifies anomaly based off threshold factor
        if residual > threshold:
            anomaly_flags[i] = True

    return residuals, anomaly_flags

# Plotting data
def real_time_visualization(data_stream, expected_data, anomaly_flags, residuals):
    plt.ion()  
    fig, ax = plt.subplots()
    for i in range(len(data_stream)):
        ax.clear()
        ax.plot(expected_data[:i+1], label='Expected Data', linestyle='--', color='blue')
        ax.plot(data_stream[:i+1], label='Data Stream', color='black')
        
        if i > 0:
            current_data = data_stream[:i+1]
            mean = np.mean(current_data)
            std_dev = np.std(current_data)
            
            ax.axhline(mean, color='green', linestyle='--', label='Mean')
            ax.fill_between(range(i + 1), mean - std_dev, mean + std_dev, color='green', alpha=0.2, label='±1 Std Dev')
        
        anomalies = np.where(np.array(anomaly_flags[:i+1]) == True)[0]
        ax.scatter(anomalies, data_stream[anomalies], color='red', label='Anomalies')
        
        ax.legend()
        plt.pause(0.01)  
    plt.ioff()  
    plt.show()

if __name__ == "__main__":
    data, expected_data = generate_data_stream(500)
    residuals, flags = std_dev_anomaly_detection(data, threshold_factor=2.5) 
    real_time_visualization(data, expected_data, flags, residuals)
