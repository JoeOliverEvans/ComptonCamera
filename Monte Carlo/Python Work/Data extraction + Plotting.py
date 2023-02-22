import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import statistics

"""
Go to the Excel file in Folders, right click and select properties(at the bottom)
then copy everything after "Location: " and paste into file_path = ''. You will
need to write the name for the file (Experimental Data 1, in my case) manually.
You can't have the Excel file open whilst running this code otherwise it won't 
work.
"""
file_path = 'C:/Users/dylan/OneDrive/Documents/Experimental Data 1.xlsx'
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' could not be found.")
    df = None

if df is not None:
    print(df)
    
energy = df["ENERGY"]

average = sum(energy)/len(energy)
stdev = statistics.stdev(energy)
print("Average energy is ", average)
print("Standard Deviation is ", stdev)


num_bins = 500
x_min = 0
x_max = 1000
x_axis = np.arange(x_min, x_max, 0.01)

# Generate histogram data with a Gaussian distribution
mu = average
sigma = stdev
data = norm.pdf(mu, sigma, 10000)

# Plot the histogram
plt.plot(x_axis, norm.pdf(x_axis, mu, sigma))

# Add labels and title
plt.xlabel('Energy (keV)')
plt.ylabel('Counts')
plt.title('XXX Spectrum')
plt.show()

