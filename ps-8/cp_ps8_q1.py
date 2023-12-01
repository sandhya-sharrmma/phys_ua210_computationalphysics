"""
Name: cp_ps8_q1.py
Author: Sandhya Sharma
Date: November 30, 2023
Description: Performing a discrete Fourier transform on a piano and trumpet signal. 

"""

import numpy as np 
import matplotlib.pyplot as plt
from pylab import *
from numpy.fft import rfft,irfft
from scipy.fft import fft, fftfreq

#Loading data
file1_path = "/Users/sandhyasharma/Desktop/piano.txt"
file2_path = "/Users/sandhyasharma/Desktop/trumpet.txt"

piano_signal = np.loadtxt(file1_path)
trumpet_signal = np.loadtxt(file2_path)

#Plotting the signals
x = np.linspace(0, 1, len(piano_signal))
plt.scatter(x, piano_signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (arbs)')
plt.title('Piano Waveform')
plt.show()

plt.scatter(x, trumpet_signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (arbs)')
plt.title('Trumpet Waveform')
plt.show()

#calculating the fourier transform for piano signal
N = len(piano_signal)
T = 1.0 /44100
piano_signal_fft = fft(piano_signal)
xf = fftfreq(N, T)
xf = fftshift(xf)
piano_signal_fft = fftshift(piano_signal_fft)
plt.plot(xf, 1.0/N * np.abs(piano_signal_fft))
plt.xlim(0, 10000)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (arbs)')
plt.title('Fourier Transform of Piano Signal')
plt.grid()
plt.show()

#calculating the fourier transform for trumpet signal
N = len(trumpet_signal)
trumpet_signal_fft = fft(trumpet_signal)
xf = fftfreq(N, T) 
xf = fftshift(xf)
trumpet_signal_fft = fftshift(trumpet_signal_fft)
plt.plot(xf, 1.0/N * np.abs(trumpet_signal_fft))
plt.xlim(0, 10000)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (arbs)')
plt.title('Fourier Transform of Trumpet Signal')
plt.grid()
plt.show()

#finding the dominant frequency
dominant_piano_freq = xf[np.argmax(np.abs(piano_signal_fft))]
print()
print('Dominant frequency of piano (Hz):', np.abs(dominant_piano_freq))

dominant_trumpet_freq = xf[np.argmax(np.abs(trumpet_signal_fft))]
print('Dominant frequency of trumpet (Hz):', np.abs(dominant_trumpet_freq))
print()





