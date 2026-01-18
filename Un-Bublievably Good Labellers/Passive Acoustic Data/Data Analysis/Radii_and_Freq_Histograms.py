import json,math
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt


def frequency_to_radius(d, f):
    p = 1000 * 9.8 * d
    return (1/(2 * math.pi * f)) * math.sqrt((3 * 1.289 * p) / 1000)


def open_file_dialog():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[("JSON Files", "*.json")]
    )
    
    return file_path

if __name__ == "__main__":
    selected_file = open_file_dialog()
    if selected_file:
        print(f"Selected file: {selected_file}")
    else:
        print("No file selected")


    try:
        with open(selected_file, 'r') as json_file:
            data = json.load(json_file)
        print("JSON file loaded successfully")
    except json.JSONDecodeError:
        print("Error: File is not valid JSON")
    except Exception as e:
        print(f"Error loading file: {e}")

    freqs = []
    radii = []
    for index in data:
        freqs.append(index['Freq'])
        radii.append(frequency_to_radius(10.41, index['Freq']))

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ax1.hist(freqs, bins=60, edgecolor='black')
    # ax1.set_xlabel('Frequency (Hz)')
    # ax1.set_ylabel('Count')
    # ax1.set_title('Frequency Histogram')
    
    # ax2.hist(radii, bins=30, edgecolor='black')
    # ax2.set_xlabel('Radius (m)')
    # ax2.set_ylabel('Count')
    # ax2.set_title('Radius Histogram')
    ax1.hist(freqs, bins=60, edgecolor='black', density=True)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Frequency Probability Distribution')
    
    ax2.hist(radii, bins=30, edgecolor='black', density=True)
    ax2.set_xlabel('Radius (m)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Radius Probability Distribution')
    plt.tight_layout()
    plt.show()

    print("Script execution completed.")