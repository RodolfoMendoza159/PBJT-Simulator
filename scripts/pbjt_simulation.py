import numpy as np
import pandas as pd
from scipy.fftpack import fft
from scipy.stats import kurtosis, skew
from input_generator import generate_random_signal


# PBJT Constants (refined from the paper)
d33 = 3.3e-12 #Zinc Oxide (ZnO)
d31 = 2.5e-12 #Poly(3-hexylthiophene) (P3HT)
#d33 = 5.0e-12 # Barium Titanate (BaTiO₃)
#d31 = 3.5e-12 #Polyaniline (PANI)
L = 50e-6
t = 500e-9
C_CB = 1e-12
C_EB = 1.2e-12
SCALING_FACTOR = 1e3

# Equations
def compute_total_charge(F33, F31):
    return d33 * F33 + (L / t) * d31 * F31

def compute_vce(Q_total):
    return Q_total / (1 / C_CB + 1 / C_EB)

def simulate_pbjt_response(signal, frequency):
    F33 = signal * frequency * 1e-3
    F31 = F33 / 2
    Q_total = compute_total_charge(F33, F31)
    V_CE = compute_vce(Q_total)
    return V_CE * SCALING_FACTOR  # Normalize

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

def generate_dataset(num_samples=50000):
    """
    Generate a large dataset for AI training with equal male and female samples.
    """
    dataset = []
    num_male = num_samples // 2
    num_female = num_samples - num_male

    # Generate male signals
    for _ in range(num_male):
        signal, freq, label = generate_random_signal(min_freq=85, max_freq=120)  # Male frequency range
        pbjt_response = simulate_pbjt_response(signal, freq)
        features = extract_features_from_signal(pbjt_response, label)  # Pass label to extract features
        features["label"] = label  # Assign label to features
        dataset.append(features)

    # Generate female signals
    for _ in range(num_female):
        signal, freq, label = generate_random_signal(min_freq=121, max_freq=255)  # Female frequency range
        pbjt_response = simulate_pbjt_response(signal, freq)
        features = extract_features_from_signal(pbjt_response, label)  # Pass label to extract features
        features["label"] = label  # Assign label to features
        dataset.append(features)

    return pd.DataFrame(dataset)

if __name__ == "__main__":
    # Generate a dataset with 50,000 examples
    dataset = generate_dataset(num_samples=50000)
    print("Dataset generated with 50,000 examples.")
    dataset.to_csv("pbjt_large_dataset.csv", index=False)
    print("Dataset saved to 'pbjt_large_dataset.csv'")
