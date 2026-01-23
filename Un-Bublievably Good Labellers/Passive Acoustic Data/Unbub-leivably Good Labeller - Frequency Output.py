import tkinter as tk
from tkinter import ttk
import numpy as np
import scipy
from scipy.io import wavfile
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import json
from tkinter import filedialog

CHUNK_DURATION = 2.0  # seconds

class SpectrogramGUI:
    def __init__(self, root, wav_path):
        self.root = root
        self.root.title("Spectrogram Viewer")
        self.root.geometry("1400x800")

        # Load WAV
        self.sr, self.data = wavfile.read(wav_path)
        if self.data.ndim > 1:
            self.data = self.data.mean(axis=1)  # convert to mono

        self.chunk_size = int(CHUNK_DURATION * self.sr)
        self.num_chunks = len(self.data) // self.chunk_size
        self.current_chunk = 0
        self.overlap = 0.5  # 75% overlap
        self.chunk_markers = []  # Store markers for current chunk
        self.markers_dict = []  # Store markers for all chunks

        self.upper_freq_limit = 3500  # Hz
        self.lower_freq_limit = 750   # Hz

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Buttons
        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=10)

        # Configure button style for larger buttons
        style = ttk.Style()
        style.configure('Large.TButton', font=('Arial', 12), padding=10)

        ttk.Button(btn_frame, text="◀ Prev", command=self.prev_chunk, style='Large.TButton').grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Next ▶", command=self.next_chunk, style='Large.TButton').grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Undo", command=self.undo_marker, style='Large.TButton').grid(row=0, column=2, padx=5)
        ttk.Button(btn_frame, text="Export", command=self.export_markers, style='Large.TButton').grid(row=0, column=3, padx=5)

        # Connect click event
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        # Draw first chunk
        self.draw_chunk()

    def get_chunk(self, idx):
        start = idx * self.chunk_size
        end = start + self.chunk_size
        return self.data[start:end]

    def draw_chunk(self):
        self.ax.clear()
        self.chunk_markers = []  # Clear markers when drawing new chunk
        # Move markers for the current chunk
        

        chunk = self.get_chunk(self.current_chunk)

        fft_size = int (self.sr / 10)
        num_padding_samples = (self.sr / 10) - fft_size
        overlap_segment = int(fft_size * self.overlap)

        fft_size = int (self.sr/100)
        num_padding_samples = (self.sr/100) - fft_size
        overlap_segment = int(fft_size * self.overlap)
    
        f, t, Sxx = scipy.signal.spectrogram(
            chunk,
            fs=self.sr,
            nperseg=fft_size,
            nfft=fft_size + num_padding_samples,
            noverlap=overlap_segment,
            scaling='density',
            mode='psd',
            window='hann'
        )

        freq_dif = f[3] - f[2]
        lower_limit = int(self.lower_freq_limit / freq_dif)
        upper_limit = int(self.upper_freq_limit / freq_dif)
        

        self.ax.set_title(f"Chunk {self.current_chunk+1}/{self.num_chunks}")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")
        Sxx[Sxx < 0] = 0
        Sxx_db = 10 * np.log10((np.abs(Sxx[lower_limit:upper_limit]) ** 2) + (2e-6) ** 2)
        im = self.ax.pcolormesh(t, f[lower_limit:upper_limit], Sxx_db, shading='auto', cmap='viridis')
        self.fig.tight_layout()


        for marker in self.markers_dict:
            if marker["Chunk"] == self.current_chunk:
                marker, = self.ax.plot(marker["Time"], marker["Freq"], marker="o", color="cyan", markersize=6)
                self.chunk_markers.append(marker)


        self.canvas.draw()

    def prev_chunk(self):
        for marker in self.chunk_markers:
            self.markers_dict.append({"Chunk": self.current_chunk, "Time": marker.get_xdata()[0], "Freq": marker.get_ydata()[0]})

        if self.current_chunk > 0:
            self.current_chunk -= 1
            self.draw_chunk()

    def next_chunk(self):
        for marker in self.chunk_markers:
            self.markers_dict.append({"Chunk": self.current_chunk, "Time": marker.get_xdata()[0], "Freq": marker.get_ydata()[0]})

        if self.current_chunk < self.num_chunks - 1:
            self.current_chunk += 1
            self.draw_chunk()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Draw a small marker
        marker, = self.ax.plot(event.xdata, event.ydata, marker="o", color="cyan", markersize=6)
        self.chunk_markers.append(marker)
        self.canvas.draw()

        #print(f"Clicked at time={event.xdata:.2f}s, freq={event.ydata:.1f}Hz")

    def undo_marker(self):
        if self.chunk_markers:
            marker = self.chunk_markers.pop()
            marker.remove()
            self.canvas.draw()








############### change to do function ###############
    def export_markers(self):

        # Create a progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Exporting...")
        progress_window.geometry("600x200")
        progress_window.resizable(False, False)

        progress_label = ttk.Label(progress_window, text="Processing markers...")
        progress_label.pack(pady=10)

        progress_bar = ttk.Progressbar(progress_window, length=550, mode='determinate', maximum=self.num_chunks)
        progress_bar.pack(pady=10)

        progress_window.update()


        # Save current chunk markers before exporting
        for marker in self.chunk_markers:
            self.markers_dict.append({"Chunk": self.current_chunk, "Time": marker.get_xdata()[0], "Freq": marker.get_ydata()[0]})

        output = []

        for i in range(self.num_chunks):
            progress_bar['value'] = i + 1
            progress_window.update()
            chunk = self.get_chunk(i)

            fft_size = int (self.sr / 10)
            num_padding_samples = (self.sr/ 10) - fft_size
            overlap_segment = int(fft_size * self.overlap)

            freqs, t, Sxx = scipy.signal.spectrogram(
                chunk,
                fs=self.sr,
                nperseg=fft_size,
                nfft=fft_size + num_padding_samples,
                noverlap=overlap_segment,
                scaling='density',
                mode='psd',
                window='hann'
            )

            times = []
            for marker in self.markers_dict:
                if marker["Chunk"] == i:
                    times.append(marker["Time"])
            t_dif = t[3] - t[2] 

            freq_dif = freqs[3] - freqs[2]
            lower_limit_index = int(self.lower_freq_limit / freq_dif)
            upper_limit_index = int(self.upper_freq_limit / freq_dif)

            for time in times:
                time_index = int(time / t_dif)- 1
                try:
                    max_freq_at_time = freqs[np.argmax(Sxx[lower_limit_index:upper_limit_index, time_index]) + lower_limit_index] + (0.5 * freq_dif)
                except:
                    print(f"Error processing chunk {i} at time {time}")

                output.append({
                    "Chunk": i,
                    "Time": time,
                    "Freq": max_freq_at_time
                })

        
        # Export to JSON file
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if filename:
            with open(filename, "w") as f:
                json.dump(output, f, indent=2)
        
        print(f"Exported {len(self.markers_dict)} markers to {filename}")
        progress_window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1400x800")
    wav_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
    if not wav_path:
        root.destroy()
        exit()
    app = SpectrogramGUI(root, wav_path)
    root.mainloop()
