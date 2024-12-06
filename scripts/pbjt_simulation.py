import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.stats import kurtosis, skew
from input_generator import generate_random_signal

# Constants
d33 = 3.3e-12  # Piezoelectric constant (m/V)
d31 = 2.5e-12  # Piezoelectric constant (m/V)
L = 50e-6      # Length of the layer (meters)
t = 500e-9     # Thickness of the layer (meters)
C_CB = 1e-12   # Capacitance of collector-base junction (Farads)
C_EB = 1.2e-12 # Capacitance of emitter-base junction (Farads)
SCALING_FACTOR = 1e3  # Scaling factor for normalization

# PBJT Equations
def compute_total_charge(F33, F31):
    return d33 * F33 + (L / t) * d31 * F31

def compute_vce(Q_total):
    return Q_total / (1 / C_CB + 1 / C_EB)

def simulate_pbjt_response(input_signal, frequency):
    F33 = input_signal * frequency * 1e-3
    F31 = F33 / 2
    Q_total = compute_total_charge(F33, F31)
    V_CE = compute_vce(Q_total)
    return V_CE * SCALING_FACTOR  # Normalize to scale

# Dynamic Range Simulation
def dynamic_range_simulation():
    F33_values = np.linspace(0, 10, 100)  # Sound pressure range in N
    pbjt_output = [compute_vce(compute_total_charge(F33, F33 / 2)) for F33 in F33_values]

    plt.plot(F33_values, pbjt_output, label="PBJT Response")
    plt.title("Dynamic Range Simulation of PBJT")
    plt.xlabel("Sound Pressure Force (F33 in N)")
    plt.ylabel("Output Voltage (V_CE in V)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Sensitivity Calculation
def sensitivity_calculation():
    F33_values = np.linspace(0, 10, 100)
    pbjt_output = [compute_vce(compute_total_charge(F33, F33 / 2)) for F33 in F33_values]
    sensitivity = np.gradient(pbjt_output, F33_values)

    plt.plot(F33_values, sensitivity, label="PBJT Sensitivity")
    plt.title("Sensitivity of PBJT vs Sound Pressure Force")
    plt.xlabel("Sound Pressure Force (F33 in N)")
    plt.ylabel("Sensitivity (dV_CE / dF33 in V/N)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Sensitivity Calculation
def sensitivity_calculation():
    F33_values = np.linspace(0, 10, 100)
    pbjt_output = [compute_vce(compute_total_charge(F33, F33 / 2)) for F33 in F33_values]
    sensitivity = np.gradient(pbjt_output, F33_values)

    plt.plot(F33_values, sensitivity, label="PBJT Sensitivity")
    plt.title("Sensitivity of PBJT vs Sound Pressure Force")
    plt.xlabel("Sound Pressure Force (F33 in N)")
    plt.ylabel("Sensitivity (dV_CE / dF33 in V/N)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Signal Transformation Test
def signal_transformation_test():
    time = np.linspace(0, 1, 1000)  # 1-second simulation
    signal, freq, label = generate_random_signal()
    pbjt_response = simulate_pbjt_response(signal, freq)

    # Plotting original signal
    plt.figure()
    plt.plot(time, signal, label=f"Original Signal ({label})")
    plt.title(f"Original Signal ({label})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting PBJT-transformed signal
    plt.figure()
    plt.plot(time, pbjt_response, label=f"PBJT Response ({label})")
    plt.title(f"PBJT Response ({label})")
    plt.xlabel("Time (s)")
    plt.ylabel("V_CE (Scaled)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Adding comparison visualization
    normalized_original = signal / np.max(np.abs(signal))
    normalized_pbjt = pbjt_response / np.max(np.abs(pbjt_response))

    plt.figure(figsize=(12, 6))
    plt.plot(time, normalized_original, label="Normalized Original Signal", color='blue', alpha=0.7, linewidth=2)
    plt.plot(time, normalized_pbjt, label="Normalized PBJT Response", color='red', alpha=0.7, linewidth=2)

    plt.title(f"Comparison of Original Signal and PBJT Response ({label})")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.grid(True)

    # Highlight max points
    plt.scatter([time[np.argmax(normalized_original)]], [np.max(normalized_original)], color='blue', label='Original Max', zorder=5)
    plt.scatter([time[np.argmax(normalized_pbjt)]], [np.max(normalized_pbjt)], color='red', label='PBJT Max', zorder=5)

    plt.show()

# Generate AI Dataset (if needed)
def generate_ai_dataset(num_samples=1000):
    dataset = []
    num_male = num_samples // 2
    num_female = num_samples - num_male

    # Generate male signals
    for _ in range(num_male):
        signal, freq, label = generate_random_signal(min_freq=85, max_freq=120)
        output = simulate_pbjt_response(signal, freq)
        features = extract_features_from_signal(output, label)
        features["label"] = label
        dataset.append(features)

    # Generate female signals
    for _ in range(num_female):
        signal, freq, label = generate_random_signal(min_freq=121, max_freq=255)
        output = simulate_pbjt_response(signal, freq)
        features = extract_features_from_signal(output, label)
        features["label"] = label
        dataset.append(features)

    return pd.DataFrame(dataset)

# Feature Extraction (for AI datasets)
def extract_features_from_signal(signal, label):
    fft_vals = np.abs(fft(signal))
    features = {
        "meanfreq": np.mean(fft_vals),
        "sd": np.std(fft_vals),
        "median": np.median(fft_vals),
        "Q25": np.percentile(fft_vals, 25),
        "Q75": np.percentile(fft_vals, 75),
        "IQR": np.percentile(fft_vals, 75) - np.percentile(fft_vals, 25),
        "sp.ent": -np.sum(fft_vals * np.log(fft_vals + 1e-12)),
        "sfm": np.exp(np.mean(np.log(fft_vals + 1e-12))) / np.mean(fft_vals),
        "centroid": np.sum(fft_vals * np.arange(len(fft_vals))) / np.sum(fft_vals),
        "kurt": kurtosis(fft_vals),
        "skew": skew(fft_vals),
        "label": label,
    }
    return features

# Main Execution
if __name__ == "__main__":
    print("Select a feature to run:")
    print("1. Generate Dataset")
    print("2. Dynamic Range Simulation")
    print("3. Sensitivity Calculation")
    print("4. Signal Transformation Test")

    choice = input("Enter the number of your choice: ")

    if choice == "1":
        dataset = generate_ai_dataset(num_samples=1000)
        dataset.to_csv("datasets/pbjt_test_dataset.csv", index=False)
        print("Dataset saved to 'datasets/pbjt_test_dataset.csv'")
    elif choice == "2":
        dynamic_range_simulation()
    elif choice == "3":
        sensitivity_calculation()
    elif choice == "4":
        signal_transformation_test()
    else:
        print("Invalid choice. Exiting.")

