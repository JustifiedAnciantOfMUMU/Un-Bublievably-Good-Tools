import os, wave
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def simulate_bubble(freq, decay_rate = 100, time = 0.1, sr = 96000, amplitude = 500):
    
    t = np.linspace(0, time, int(sr * time))  # Time array
    y = amplitude * np.exp(-decay_rate * t) * np.sin(2 * np.pi * freq * t)  # Exponentially decaying sinusoid
    return t, y

def plot(data):
    plt.figure(3)
    plt.clf()
    plt.plot(data)
    plt.ylabel('Amplitude')
    plt.xlabel('Samples')
    plt.show()


##Example usage
background_dir = r'C:\Users\jkf1g22\OneDrive - University of Southampton\Documents\_PostGraduateResearch\UKAN 25\background_clips'
out_dir = r'C:\Users\jkf1g22\OneDrive - University of Southampton\Documents\_PostGraduateResearch\UKAN 25\pos_samples'
files = os.listdir(background_dir)

x = 0
for file in files:
    # Randomly select a number between 1 and 3 for each file
    num_bubs = random.randint(1, 1)
    bub_freq = random.randint(1000, 8000)
    # Generate the exponentially decaying sinusoid
    master_bub = np.zeros(96000)
    for i in range(num_bubs):
        bub_time = 1 - (random.randint(0, 98) / 100)
        start = np.zeros(int(bub_time * 96000))
        bubble = simulate_bubble(bub_freq)
        bubble = np.append(start, bubble[1])
        if bubble.size < 96000:
            len = 96000 - bubble.size
            bubble = np.append(bubble, np.zeros(len))
        else:
            bubble = bubble[:96000]
        #plot(bubble)
        master_bub = bubble.astype(np.int16) + master_bub.astype(np.int16)
        #plot(master_bub)

    #open audio
    sr, audio_data = wavfile.read(os.path.join(background_dir, file))

    channels = audio_data[0].size
    if channels > 1:
        # Extract the first channel
        first_channel_data = audio_data[:, 0]
    else:
        # If the audio is already mono, use the audio data as is
        first_channel_data = audio_data

    #sum
    out_wav = first_channel_data.astype(np.int16) + master_bub.astype(np.int16)

    plot(out_wav)
    #save audio
    out_name = 'pos_' + str(x) + '.wav'
    wavfile.write(os.path.join(out_dir, out_name), 96000, out_wav)
    x += 1

    

    

print('Done!')