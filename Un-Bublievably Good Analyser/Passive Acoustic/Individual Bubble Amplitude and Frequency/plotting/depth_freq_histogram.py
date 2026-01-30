import csv
import os, math
import matplotlib.pyplot as plt
import numpy as np


# Path to the CSV file produced by the previous script
csv_filepath = 'C:\\Users\\jkf1g22\\OneDrive - University of Southampton\\Desktop\\Effect of Varying Depth\\bubbles\\grouped_files.csv'

# Read the CSV file
data = []
with open(csv_filepath, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(row)


depths = list(set(row['Depth'] for row in data))


# Group radii by depth
radii_by_depth = {}
for depth in depths:
    radii = [float(row['frequency']) for row in data if row['Depth'] == depth]
    radii_by_depth[depth] = radii


# Define bins once
bins = np.linspace(500, 3500, 15)

# Plot histogram for each depth
plt.figure(figsize=(10, 6))
for depth in sorted(radii_by_depth.keys(), key=float):
    counts, bin_edges = np.histogram(radii_by_depth[depth], bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_centers, counts, linewidth=2, marker='o', markersize=4, label=f'Depth {depth}m')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Probability Density')
plt.title('Bubble Frequency Distribution at Various Depths')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(data)  # Print the data to verify it has been read correctly

