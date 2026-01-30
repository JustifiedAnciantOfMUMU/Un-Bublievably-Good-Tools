import os, math, csv
import numpy as np
from scipy.signal import butter, filtfilt, freqz
from scipy.io import wavfile
import matplotlib.pyplot as plt


def analyze_wav(filepath, lowcut=750.0, highcut=3000.0, order=4):
    sample_rate, data = wavfile.read(filepath)
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    ## x = freqz(b,a)

    # plt.figure()
    # plt.plot(np.abs(x[1]))
    # plt.show()

    if data.ndim > 1:
        data_to_filter = data[:, 0]
    else:
        data_to_filter = data

    
    filtered_data = filtfilt(b, a, data_to_filter)
    plt.figure()
    plt.plot(filtered_data)
    plt.show()

    peak_amplitude = np.max(np.abs(filtered_data))
    amp_in_volts = (peak_amplitude / 32768.0) * 2.598
    reciever_pa = amp_in_volts / (10**(hydrophone_sensitivity_db/20) * (1e06))
    revciever_db = 20 * math.log10(reciever_pa / 1e-06)  # Convert to dB re 1uPa

    fft_result = np.fft.rfft(filtered_data)
    fft_freqs = np.fft.rfftfreq(len(filtered_data), d=1/sample_rate)
    max_freq_index = np.argmax(np.abs(fft_result))
    max_freq = fft_freqs[max_freq_index]

    return max_freq, reciever_pa



hydrophone_sensitivity_db = -164.7   # Sensitivity of the hydrophone in dB re 1uPa
hydrophone_sensitivity_v = 10 ** (hydrophone_sensitivity_db / 20)  # Convert dB to voltage ratio

dataset_path = 'C:\\Users\\jkf1g22\\OneDrive - University of Southampton\\Desktop\\Effect of Varying Depth\\bubbles'

# Get list of file names in current directory
file_names = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]

# Group files by first 2 characters
grouped_files = {}
for file in file_names:
    key = file[:2]
    key = key.replace('m', '')

    if key not in grouped_files:
        grouped_files[key] = []

    bub_freq, bub_SPL = analyze_wav(os.path.join(dataset_path, file))

    grouped_files[key].append({"index":file[4:-4], "frequency": bub_freq, "SPL": bub_SPL})  # Append filename without first 4 chars and .wav extension

  # Print the grouped files dictionary to verify the result
csv_filepath = os.path.join(dataset_path, 'all_depths.csv')
with open(csv_filepath, 'w', newline='') as csvfile:
    fieldnames = ['Depth', 'index', 'frequency', 'SPL']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for group, files in grouped_files.items():
        for file_data in files:
            row = {'Depth': group, **file_data}
            writer.writerow(row)