import numpy as np
import matplotlib.pyplot as plt

# Data stream generation with both positive and negative anomalies
def generate_data_stream(n_points=1000, anomaly_chance=0.05):
    time = np.arange(n_points)
    seasonal_pattern = 10 * np.sin(2 * np.pi * time / 50)  # Simulate a sinusoidal pattern
    noise = np.random.normal(0, 2, n_points)  # Add noise
    data_stream = seasonal_pattern + noise
    
    anomalies = np.random.choice([0, 1], size=n_points, p=[1 - anomaly_chance, anomaly_chance])
    anomaly_magnitudes = np.random.uniform(-40, 40, n_points)  
    data_stream += anomalies * anomaly_magnitudes
    
    return data_stream, seasonal_pattern

# EWMA-based anomaly detection with improved thresholding
def ewma_anomaly_detection(data_stream, alpha, threshold_factor):
    ewma = [data_stream[0]]  # Initialize EWMA with the first value
    residuals = []
    anomaly_flags = [False] * len(data_stream)

    # Compute EWMA and residuals
    for i in range(1, len(data_stream)):
        ewma_value = alpha * data_stream[i] + (1 - alpha) * ewma[-1]
        ewma.append(ewma_value)
        residual = abs(data_stream[i] - ewma_value)
        residuals.append(residual)

    # Convert residuals to a NumPy array for easier processing
    residuals = np.array(residuals)

    # Calculate the Median Absolute Deviation (MAD) for robust thresholding
    median_residual = np.median(residuals)
    mad = np.median(np.abs(residuals - median_residual))
    
    # Improve the threshold calculation by making it adaptive to the data
    threshold = median_residual + threshold_factor * mad
    print(f"Anomaly detection threshold: {threshold:.2f}")

    # Detect anomalies based on the threshold
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

        # Mark anomalies with red dots
        anomalies = np.where(np.array(anomaly_flags[:i+1]) == True)[0]
        ax.scatter(anomalies, data_stream[anomalies], color='red', label='Anomalies')
        
        ax.legend()
        plt.pause(0.01)  

    plt.ioff()  
    plt.show()

if __name__ == "__main__":
    data, expected_data = generate_data_stream(500)
    ewma, flags = ewma_anomaly_detection(data, alpha=0.1, threshold_factor=3)  
    real_time_visualization(data, expected_data, flags, ewma)
