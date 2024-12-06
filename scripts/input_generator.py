import numpy as np

# Constants
sampling_rate = 1000  # Hz
time = np.linspace(0, 1, sampling_rate)

def generate_random_signal(min_freq=85, max_freq=255, amplitude=1.0):
    """
    Generate a random sine wave signal within the given frequency range.
    """
    freq = np.random.uniform(min_freq, max_freq)
    label = "male" if freq <= 120 else "female"
    signal = amplitude * np.sin(2 * np.pi * freq * time)
    return signal, freq, label
