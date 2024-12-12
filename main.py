import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# Load dataset
dataset_path = "airline7.csv"
data = pd.read_csv(dataset_path)
data['Date'] = pd.to_datetime(data['Date'])

data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Fourier Transform of passenger
time_series = data['Number'].values
n = len(time_series)
frequencies = np.fft.fftfreq(n)
fft_values = fft(time_series)

# revenue fractions for autumn months : X
data_autumn = data[data['Month'].isin([9, 10, 11])]
data_total_revenue = data['Revenue'].sum()
revenue_fraction_autumn = (data_autumn['Revenue'].sum() / data_total_revenue) * 100

# passenger fractions for autumn months : Y
passenger_fraction_autumn = (data_autumn['Number'].sum() / data['Number'].sum()) * 100

# Power Spectrum
power_spectrum = np.abs(fft_values[:n // 2])**2
non_zero_frequencies = frequencies[:n // 2][frequencies[:n // 2] != 0]
power_spectrum_non_zero = power_spectrum[frequencies[:n // 2] != 0]

# Plots
plt.figure(figsize=(10, 6))
plt.bar(range(1, 13), data.groupby('Month')['Number'].mean(), color='skyblue', label='Monthly Average Passengers')
plt.xlabel('Month')
plt.ylabel('Passengers (in thousands)')
plt.title('Monthly Passenger Analysis\nStudent ID: 23078277')
plt.legend()
plt.savefig('fig1.png')

plt.figure(figsize=(10, 6))
plt.plot(non_zero_frequencies, power_spectrum_non_zero, label='Power Spectrum')
plt.xlabel('Frequency (1/day)')
plt.ylabel('Power')
plt.title('Power Spectrum\nStudent ID: 23078277')
plt.legend()
plt.savefig('fig2.png')

print(f"Value X (Revenue Fraction in Autumn): {revenue_fraction_autumn:.2f}%")
print(f"Value Y (Passenger Fraction in Autumn): {passenger_fraction_autumn:.2f}%")
