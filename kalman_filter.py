import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, process_noise=1e-2, measurement_noise=1e-1):
        self.state = np.array(initial_state, dtype=np.float32)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.covariance = np.eye(2, dtype=np.float32)
    
    def predict(self):
        return self.state
    
    def correct(self, measurement):
        kalman_gain = self.covariance / (self.covariance + self.measurement_noise)
        self.state = self.state + kalman_gain * (measurement - self.state)
        self.covariance = (1 - kalman_gain) * self.covariance + self.process_noise
        return self.state