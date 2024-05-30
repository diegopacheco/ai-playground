import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
import matplotlib.pyplot as plt

# Load the audio file
sample_rate, data = wavfile.read('audio_file.wav')

# Perform the Fourier Transform
transformed_data = fft(data)

# Compute the frequencies
frequencies = np.linspace(0.0, sample_rate/2, len(transformed_data)//2)

# Plot the frequencies
plt.plot(frequencies, 2.0/len(transformed_data) * np.abs(transformed_data[0:len(transformed_data)//2]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()