import os
import obspy
import seisbench
import seisbench.models as sbm
import numpy as np
import tkinter as tk
from tkinter import filedialog
from obspy import read
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class SeismicGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Seismic Data Picking Viewer")
        self.eqt_model = sbm.EQTransformer.from_pretrained("original")

        # Frame for buttons
        self.frame = tk.Frame(root)
        self.frame.pack(side=tk.TOP, fill=tk.X)

        # Load Button
        self.load_button = tk.Button(self.frame, text="Load SAC File", command=self.load_sac_file)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Read Button
        self.read_button = tk.Button(self.frame, text="Read Data", command=self.plot_data)
        self.read_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Save Button
        self.save_button = tk.Button(self.frame, text="Save Point", command=self.save_point)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Reset Button
        self.reset_button = tk.Button(self.frame, text="Reset Zoom", command=self.reset_zoom)
        self.reset_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Increase Amplitude Button
        self.increase_amplitude_button = tk.Button(self.frame, text="+", command=self.increase_amplitude)
        self.increase_amplitude_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Decrease Amplitude Button
        self.decrease_amplitude_button = tk.Button(self.frame, text="-", command=self.decrease_amplitude)
        self.decrease_amplitude_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Reset Amplitude Button
        self.reset_amplitude_button = tk.Button(self.frame, text="↶", command=self.reset_amplitude)
        self.reset_amplitude_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Text box for bandpass filter parameters
        self.filter_frame = tk.Frame(root)
        self.filter_frame.pack(side=tk.TOP, fill=tk.X)
        self.low_freq_label = tk.Label(self.filter_frame, text="Low Frequency (Hz):")
        self.low_freq_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.low_freq_entry = tk.Entry(self.filter_frame, width=10)
        self.low_freq_entry.pack(side=tk.LEFT, padx=5, pady=5)

        self.high_freq_label = tk.Label(self.filter_frame, text="High Frequency (Hz):")
        self.high_freq_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.high_freq_entry = tk.Entry(self.filter_frame, width=10)
        self.high_freq_entry.pack(side=tk.LEFT, padx=5, pady=5)

        self.apply_filter_button = tk.Button(self.filter_frame, text="Apply Filter", command=self.apply_filter)
        self.apply_filter_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Figure and Axes for plotting
        self.fig, self.axs = plt.subplots(3, 1, sharex=True, figsize=(8, 6), gridspec_kw={'hspace': 0})
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Variables to hold data
        self.stream = None
        self.filtered_stream = None  # Store filtered version for plotting
        self.selected_sample = None
        self.selected_sample_s = None
        self.initial_xlim = []
        self.dragging = False
        self.previous_mouse_x = None
        self.vertical_lines = [None, None, None]  # Store vertical line references for each plot
        self.red_lines = [None, None, None]  # Store red line references for each plot
        self.amplitude_factor = 1.0  # Amplitude scaling factor

        # Connect matplotlib events
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press_drag)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        

    def load_sac_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("All files", "*.*"), ("SAC files", "*.SAC")])
        if not self.file_path:
            return

        # Automatically load DPN, DPE, DPZ files
        self.base_path = os.path.splitext(self.file_path)[0][:-3]  # Remove .DPN/.DPE/.DPZ
        self.dpz_path = self.base_path + "DPZ.SAC"
        self.dpn_path = self.base_path + "DPN.SAC"
        self.dpe_path = self.base_path + "DPE.SAC"

        try:
            self.stream = obspy.Stream()
            if os.path.exists(self.dpz_path):
                self.stream += read(self.dpz_path)
            if os.path.exists(self.dpn_path):
                self.stream += read(self.dpn_path)
            if os.path.exists(self.dpe_path):
                self.stream += read(self.dpe_path)
        except FileNotFoundError:
            tk.messagebox.showerror("Error", "Could not find all components (DPN, DPE, DPZ)")
            return

        for ax in self.axs:
            ax.clear()
        self.canvas.draw()

        self.selected_sample = None
        self.selected_sample_s = None
        self.vertical_lines = [None, None, None]
        self.red_lines = [None, None, None]
        self.dragging = False
        self.previous_mouse_x = None
        self.initial_xlim = []
        self.amplitude_factor = 1.0
        
        self.filtered_stream = self.stream.copy()  # Keep a copy of the original stream for filtering
        
        eqt_preds = self.eqt_model.annotate(self.stream)
        if 'user7' in self.stream[0].stats.sac:
            self.p_label = self.stream[0].stats.sac['user7']
        else:
            p_idx = np.argmax(eqt_preds[1])
            self.p_label = eqt_preds[1].stats.starttime - self.stream[0].stats.starttime + (p_idx / 100)
        if 'user8' in self.stream[0].stats.sac:
            self.s_label = self.stream[0].stats.sac['user8']
        else:
            s_idx = np.argmax(eqt_preds[2])
            self.s_label = eqt_preds[1].stats.starttime - self.stream[0].stats.starttime + (s_idx / 100)
        if 'a' in self.stream[0].stats.sac:
            self.selected_sample = self.stream[0].stats.sac['a']
        if 't0' in self.stream[0].stats.sac:
            self.selected_sample_s = self.stream[0].stats.sac['t0']

        # Convert indices to actual times
        self.components = ['DPZ', 'DPN', 'DPE']
        print(self.stream)

    def plot_data(self):
        if self.filtered_stream is None and self.stream is None:
            return

        for ax in self.axs:
            ax.clear()

        stream_to_plot = self.filtered_stream if self.filtered_stream is not None else self.stream

        for i, tr in enumerate(stream_to_plot[:3]):
            self.axs[i].plot(tr.times(), tr.data * self.amplitude_factor, color='black', label=self.components[i])
            self.axs[i].axvline(self.p_label, color='green', linestyle='--')
            self.axs[i].axvline(self.s_label, color='green', linestyle='--')
            if self.selected_sample is not None:
                self.vertical_lines[i] = self.axs[i].axvline(self.selected_sample, color='blue', linestyle='--')
            if self.selected_sample_s is not None:
                self.red_lines[i] = self.axs[i].axvline(self.selected_sample_s, color='red', linestyle='--')
            self.axs[i].legend(loc='upper right')
            if self.amplitude_factor == 1.0:
                self.initial_ylim = (np.min(tr.data) * 1.1, np.max(tr.data) * 1.1)
            self.axs[i].set_ylim(self.initial_ylim)
            self.initial_xlim.append(self.axs[i].get_xlim())

        self.axs[2].set_xlabel("Time (s)")
        self.fig.tight_layout()
        self.canvas.draw()

    def apply_filter(self):
        if self.stream is None:
            tk.messagebox.showwarning("Warning", "No data loaded.")
            return

        try:
            low_freq = float(self.low_freq_entry.get())
            high_freq = float(self.high_freq_entry.get())
        except ValueError:
            tk.messagebox.showerror("Error", "Invalid filter parameters.")
            return

        if low_freq >= high_freq:
            tk.messagebox.showerror("Error", "Low frequency must be less than high frequency.")
            return

        self.filtered_stream = self.stream.copy()  # Copy original stream before applying filter
        self.filtered_stream.filter('bandpass', freqmin=low_freq, freqmax=high_freq)
        self.plot_data()

    def update_amplitude(self):
        stream_to_plot = self.filtered_stream if self.filtered_stream is not None else self.stream

        for i, tr in enumerate(stream_to_plot[:3]):
            self.axs[i].clear()
            self.axs[i].plot(tr.times(), tr.data * self.amplitude_factor, color='black')
            self.axs[i].set_ylim(self.initial_ylim)

        self.canvas.draw()

        if stream_to_plot is None:
            return

        for ax in self.axs:
            ax.clear()

        for i, tr in enumerate(stream_to_plot[:3]):
            self.axs[i].plot(tr.times(), tr.data * self.amplitude_factor, color='black', label=self.components[i])
            self.axs[i].axvline(self.p_label, color='green', linestyle='--')
            self.axs[i].axvline(self.s_label, color='green', linestyle='--')
            self.axs[i].legend(loc='upper right')
            if self.selected_sample is not None:
                self.vertical_lines[i] = self.axs[i].axvline(self.selected_sample, color='blue', linestyle='--')
            if self.selected_sample_s is not None:
                self.red_lines[i] = self.axs[i].axvline(self.selected_sample_s, color='red', linestyle='--')
            if self.amplitude_factor == 1.0:
                self.initial_ylim = (np.min(tr.data) * 1.1, np.max(tr.data) * 1.1)
            self.axs[i].set_ylim(self.initial_ylim)  # Fix y-axis limits after initial plot * 1.1, np.max(tr.data * self.amplitude_factor) * 1.1)  # Dynamically adjust y-axis limits
            self.initial_xlim.append(self.axs[i].get_xlim())

        self.axs[2].set_xlabel("Time (s)")
        self.fig.tight_layout()
        self.canvas.draw()

    def increase_amplitude(self):
        self.amplitude_factor *= 1.2
        self.update_amplitude()

    def decrease_amplitude(self):
        self.amplitude_factor /= 1.2
        self.update_amplitude()

    def reset_amplitude(self):
        self.amplitude_factor = 1.0
        self.update_amplitude()

    def on_key_press(self, event):
        if event.key == 'p' and (self.filtered_stream is not None or self.stream is not None):
            # Mark the point when 'P' key is pressed
            for i, ax in enumerate(self.axs):
                if self.vertical_lines[i] is not None:
                    self.vertical_lines[i].remove()
                self.vertical_lines[i] = ax.axvline(event.xdata, color='blue', linestyle='--')
            self.selected_sample = event.xdata
            self.canvas.draw()

            # Print selected point (for debugging)
            print(f"Marked sample at time: {event.xdata}")

        elif event.key == 's' and (self.filtered_stream is not None or self.stream is not None):
            # Mark another point when 'S' key is pressed
            for i, ax in enumerate(self.axs):
                if self.red_lines[i] is not None:
                    self.red_lines[i].remove()
                self.red_lines[i] = ax.axvline(event.xdata, color='red', linestyle='--')
            self.selected_sample_s = event.xdata
            self.canvas.draw()

            # Print selected point (for debugging)
            print(f"Marked sample with 'S' at time: {event.xdata}")

        elif event.key == 'q':
            # Zoom out using 'q' key
            self.on_scroll(type('event', (object,), {'step': -1, 'xdata': None}))
        elif event.key == 'w':
            # Zoom in using 'w' key
            self.on_scroll(type('event', (object,), {'step': 1, 'xdata': None}))

    def on_scroll(self, event):
        base_scale = 1.1
        if event.step > 0:  # Zoom in
            scale_factor = base_scale
        else:  # Zoom out
            scale_factor = 1 / base_scale

        for ax in self.axs:
            xlim = ax.get_xlim()
            x_range = (xlim[1] - xlim[0]) * scale_factor
            center = event.xdata if event.xdata is not None else (xlim[0] + xlim[1]) / 2
            ax.set_xlim([center - x_range / 2, center + x_range / 2])

        self.canvas.draw()

    def reset_zoom(self):
        if not self.initial_xlim:
            return

        for i, ax in enumerate(self.axs):
            ax.set_xlim(self.initial_xlim[i])

        self.canvas.draw()

    def on_press_drag(self, event):
        if event.button == 1 and event.inaxes in self.axs:  # Left mouse button pressed
            self.dragging = True
            self.previous_mouse_x = event.xdata

    def on_release(self, event):
        if event.button == 1:  # Left mouse button released
            self.dragging = False
            self.previous_mouse_x = None

    def on_motion(self, event):
        if self.dragging and event.inaxes in self.axs:
            dx = (event.xdata - self.previous_mouse_x) * 0.15  # Reduce sensitivity by a factor of 10
            for ax in self.axs:
                xlim = ax.get_xlim()
                ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            self.previous_mouse_x = event.xdata
            self.canvas.draw()

    def save_point(self):
        if self.stream is None:
            tk.messagebox.showwarning("Warning", "No data loaded.")
            return
        
        for trace in self.stream:
            # Update header with selected samples
            if self.selected_sample is not None:
                trace.stats.sac['a'] = float(self.selected_sample)
            if self.selected_sample_s is not None:
                trace.stats.sac['t0'] = float(self.selected_sample_s)
            if self.p_label is not None:
                trace.stats.sac['user7'] = float(self.p_label)
            if self.s_label is not None:
                trace.stats.sac['user8'] = float(self.s_label)
            
        # Write to new SAC file
        self.stream[0].write(self.dpz_path, format='SAC')
        self.stream[1].write(self.dpn_path, format='SAC')
        self.stream[2].write(self.dpe_path, format='SAC')
        tk.messagebox.showinfo("Info", "File saved successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SeismicGUI(root)
    root.mainloop()
