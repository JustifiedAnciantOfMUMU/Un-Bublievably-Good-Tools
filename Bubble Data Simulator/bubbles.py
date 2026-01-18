import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.io import wavfile


# Bubble radius distribution
r0_mean = 1e-3
r0_std = 1e-3

# Paul's model parameters
rho = 1000.
k = 1.4 # From Freya
h = 200
p0 = 100e3 + rho*9.81*h
sigma = 72e-3 # N/m # Freya

alpha = 100 # Envelope decay
an = 1 # Bubble signal amplitude

# Sampling paramters
tmax = 10.
fs = 22.05e3
dt = 1/fs
t = np.arange(0,tmax,dt)


# White, Gaussian-distributed noise
SNR = 0. # dB
n = np.random.randn(len(t)) * 10**(-SNR/20)
y = n


# Generate bubble signal
nrand = 300
r_dist = np.random.lognormal(0,0.5,nrand) / 3 * 1e-2


# Plot distribution
plt.figure(1)
plt.clf()
plt.hist(r_dist, bins=np.arange(0,1.5e-2,0.05e-2))
plt.xlabel('Bubble radius, m')
plt.ylabel('Histogram')
plt.show()


for n in range(nrand):
    
    r0 = r_dist[n] #np.random.randn()*r0_std + r0_mean
    
    f0 = 1/(2*np.pi*r0) * \
np.sqrt( 2*k*p0/rho * (1 + 2*sigma/(p0*r0) - 2*sigma/(p0*r0)) )
    print(f0)
    
    tn = np.random.rand() * tmax
    
    s = an * np.exp(-alpha*np.abs(t-tn)) * np.sin( 2*np.pi*f0 * (t-tn) ) * (t>tn)
    
    # Combined signal and noise
    y = y + s
    
    plt.plot(t,s)

plt.xlabel('Time, s')
plt.ylabel('Signal Amplitude, s')


# Spectrogram
# twin = 0.05
# [fspec, tspec, spec] = scipy.signal.spectrogram(y,int(fs),nperseg=round(twin/dt))
# plt.figure(3)
# plt.clf()
# plt.pcolormesh(tspec, fspec/1e3, 20*np.log10(spec/np.median(spec)),vmin=-30,vmax=30)
# plt.ylabel('Frequency, kHz')
# plt.xlabel('Time, s')
# plt.colorbar(label='dB re Median')
# plt.show()

wavfile.write(r"C:\Users\jkf1g22\OneDrive - University of Southampton\Documents\_PostGraduateResearch\_Tools\JoeFone\GasFluxScripts_JoeFone-ed\AcousticGasFlux\test\test.wav", int(fs), y)

print("done")