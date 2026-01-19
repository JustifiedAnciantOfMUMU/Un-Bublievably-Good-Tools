import csv
import os
import matplotlib.pyplot as plt

# Path to the CSV file produced by the previous script
csv_filepath = 'C:\\Users\\jkf1g22\\OneDrive - University of Southampton\\Desktop\\Effect of Varying Depth\\bubbles\\grouped_files.csv'

# Read the CSV file
data = []
with open(csv_filepath, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(row)


depths = list(set(row['Depth'] for row in data))

mean_spl_by_depth = {}
median_spl_by_depth = {}
for depth in depths:
    spl_values = [float(row['frequency']) for row in data if row['Depth'] == depth and row["index"][-2:] != '01']
    mean_spl_by_depth[depth] = sum(spl_values) / len(spl_values) if spl_values else 0
    median_spl_by_depth[depth] = sorted(spl_values)[len(spl_values)//2] if spl_values else 0

print(mean_spl_by_depth)

depths_sorted = sorted(median_spl_by_depth.keys(), key=float)
spl_values = [median_spl_by_depth[depth] for depth in depths_sorted]

plt.figure(figsize=(10, 6))
plt.plot(depths_sorted, spl_values, marker='o')
plt.xlabel('Depth')
plt.ylabel('Mean Frequency')
plt.title('Mean Frequency by Depth')
plt.grid(True)
plt.show()

print(data)  # Print the data to verify it has been read correctly

