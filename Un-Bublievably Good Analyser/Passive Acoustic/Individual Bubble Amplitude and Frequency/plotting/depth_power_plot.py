import csv
import os
import matplotlib.pyplot as plt

# Path to the CSV file produced by the previous script
csv_filepath = 'C:\\Users\\jkf1g22\\OneDrive - University of Southampton\\Desktop\\Effect of Varying Depth\\bubbles\\all_depths.csv'

# Read the CSV file
data = []
with open(csv_filepath, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(row)


depths = list(set(row['Depth'] for row in data))

mean_spl_by_depth = {}
median_spl_by_depth = {}

plt.figure(figsize=(10, 6))
for depth in depths:
    spl_values = [float(row['SPL']) for row in data if row['Depth'] == depth and row["index"][-2:] != '01']
    mean_spl_by_depth[depth] = sum(spl_values) / len(spl_values) if spl_values else 0
    median_spl_by_depth[depth] = sorted(spl_values)[len(spl_values)//2] if spl_values else 0

print(mean_spl_by_depth)

depths_sorted = sorted(median_spl_by_depth.keys(), key=float)
spl_values = [median_spl_by_depth[depth] for depth in depths_sorted]
errors = []
for depth in depths_sorted:
    depth_spl_values = [float(row['SPL']) for row in data if row['Depth'] == depth and row["index"][-2:] != '01']
    sorted_values = sorted(depth_spl_values)
    p95_index = int(len(sorted_values) * 0.95)
    p5_index = int(len(sorted_values) * 0.05)
    high_error = sorted_values[p95_index] - median_spl_by_depth[depth] if depth_spl_values else 0
    low_error = median_spl_by_depth[depth] - sorted_values[p5_index] if depth_spl_values else 0
    errors.append((low_error, high_error))

plt.errorbar([int(d) for d in depths_sorted], spl_values, yerr=list(zip(*errors)), fmt='o', linestyle='')

plt.xlabel('Depth')
plt.ylabel('Median SPL')
plt.title('Median SPL by Depth')
plt.grid(True)
plt.show()

print(data)  # Print the data to verify it has been read correctly

