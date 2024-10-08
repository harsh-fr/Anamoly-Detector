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

# Z-score based anomaly detection excluding detected anomalies
def z_score_anomaly_detection(data_stream, threshold):
    anomaly_flags = [False] * len(data_stream)  # Initialize anomaly flags
    for i in range(len(data_stream)):
        current_data = data_stream[:i+1]
        mean = np.mean(current_data)
        std_dev = np.std(current_data)
        
        # Calculate Z-scores for current data
        z_scores = (current_data - mean) / std_dev
        # Detect anomalies based on Z-scores
        anomaly_flags[:i+1] = np.abs(z_scores) > threshold
    
    # Exclude detected anomalies from calculation
    valid_data = np.array([data_stream[i] for i in range(len(data_stream)) if not anomaly_flags[i]])
    if len(valid_data) == 0: 
        return np.zeros(len(data_stream)), anomaly_flags
    
    mean = np.mean(valid_data)
    std_dev = np.std(valid_data)
    
    z_scores = (data_stream - mean) / std_dev
    anomaly_flags = np.abs(z_scores) > threshold
    
    return z_scores, anomaly_flags

def real_time_visualization(data_stream, expected_data, anomaly_flags, z_scores):
    plt.ion()  
    fig, ax = plt.subplots()
    
    for i in range(len(data_stream)):
        ax.clear()
        
        ax.plot(expected_data[:i+1], label='Expected Data', linestyle='--', color='blue')
        ax.plot(data_stream[:i+1], label='Data Stream', color='black')
        
        ax.plot(range(i + 1), z_scores[:i + 1], label='Z-Score', color='green')
        anomalies = np.where(np.array(anomaly_flags[:i+1]) == True)[0]
        ax.scatter(anomalies, data_stream[anomalies], color='red', label='Anomalies')
        
        ax.legend()
        plt.pause(0.01) 

    plt.ioff()  
    plt.show()

if __name__ == "__main__":
    data, expected_data = generate_data_stream(500)
    z_scores, flags = z_score_anomaly_detection(data, threshold=2) 
    real_time_visualization(data, expected_data, flags, z_scores)
