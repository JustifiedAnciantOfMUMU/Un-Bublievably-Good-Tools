import csv
import os, math
import matplotlib.pyplot as plt
import numpy as np



g     = 9.8           
rho   = 1000        # Density of surrounding  - liquid Kg/m3
k     = 1.289       # Polytropic index of gas
sigma = 0.072         # Surface Tension

def frequency_to_radius(d, f):
    p = rho * g * d
    return (1/(2 * math.pi * f)) * math.sqrt((3*k*p) / rho)


center_radii = 0.0019
radii_deviation = 0.00015


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
power_by_depth = {}
for depth in depths:
    radii = [float(row['SPL']) for row in data if row['Depth'] == depth and abs(float(frequency_to_radius(float(depth), float(row['frequency']))) - center_radii) <= radii_deviation]
    power_by_depth[depth] = radii


# Plotting the data
plt.figure(figsize=(10, 6))
# Sort depths for ordered plotting
sorted_depths = sorted(power_by_depth.keys(), key=float)
for depth in sorted_depths:
    radii = power_by_depth[depth]
    plt.scatter([depth] * len(radii), radii, label=f'Depth: {depth}')

plt.xlabel('Depth (m)')
plt.ylabel('Power (SPL)')
plt.title('Power by Depth for bubbles with radii of {:.2f} cm +- {:.3f} cm'.format(center_radii * 100, radii_deviation * 100))
plt.legend()
plt.grid()
plt.show()



print(data)  # Print the data to verify it has been read correctly