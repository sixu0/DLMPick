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
        self.root.title("Seismic Data Viewer")
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

        # Figure and Axes for plotting
        self.fig, self.axs = plt.subplots(3, 1, sharex=True, figsize=(8, 6), gridspec_kw={'hspace': 0})
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Variables to hold data
        self.stream = None
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
        base_path = os.path.splitext(self.file_path)[0][:-3]  # Remove .DPN/.DPE/.DPZ
        dpn_path = base_path + "DPZ.SAC"
        dpe_path = base_path + "DPN.SAC"
        dpz_path = base_path + "DPE.SAC"

        try:
            self.stream = obspy.Stream()
            if os.path.exists(dpn_path):
                self.stream += read(dpn_path)
            if os.path.exists(dpe_path):
                self.stream += read(dpe_path)
            if os.path.exists(dpz_path):
                self.stream += read(dpz_path)
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

        print(self.stream)

    def plot_data(self):
        if self.stream is None:
            return

        for ax in self.axs:
            ax.clear()

        eqt_preds = self.eqt_model.annotate(self.stream)
        p_label = np.argmax(eqt_preds[1]) / 100
        s_label = np.argmax(eqt_preds[2]) / 100

        components = ['DPZ', 'DPN', 'DPE']
        for i, tr in enumerate(self.stream[:3]):
            self.axs[i].plot(tr.times(), tr.data * self.amplitude_factor, color='black', label=components[i])
            self.axs[i].axvline(p_label, color='green', linestyle='--')
            self.axs[i].axvline(s_label, color='green', linestyle='--')
            self.axs[i].legend(loc='upper right')
            if self.amplitude_factor == 1.0:
                self.initial_ylim = (np.min(tr.data) * 1.1, np.max(tr.data) * 1.1)
            self.axs[i].set_ylim(self.initial_ylim)

        self.axs[2].set_xlabel("Time (s)")
        self.fig.tight_layout()
        self.canvas.draw()

    def update_amplitude(self):
        for i, tr in enumerate(self.stream[:3]):
            self.axs[i].clear()
            self.axs[i].plot(tr.times(), tr.data * self.amplitude_factor, color='black')
            self.axs[i].set_ylim(self.initial_ylim)

        self.canvas.draw()
        if self.stream is None:
            return

        for ax in self.axs:
            ax.clear()

        eqt_preds = self.eqt_model.annotate(self.stream)
        p_label = np.argmax(eqt_preds[1]) / 100
        s_label = np.argmax(eqt_preds[2]) / 100

        components = ['DPZ', 'DPN', 'DPE']
        for i, tr in enumerate(self.stream[:3]):
            self.axs[i].plot(tr.times(), tr.data * self.amplitude_factor, color='black', label=components[i])
            self.axs[i].axvline(p_label, color='green', linestyle='--')
            self.axs[i].axvline(s_label, color='green', linestyle='--')
            self.axs[i].legend(loc='upper right')
            if self.selected_sample is not None:
                self.axs[i].axvline(self.selected_sample, color='blue', linestyle='--')
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
          # Maintain fixed y-axis limits
        self.update_amplitude()

    def decrease_amplitude(self):
        self.amplitude_factor /= 1.2
          # Maintain fixed y-axis limits
        self.plot_data()
        self.update_amplitude()

    def on_key_press(self, event):
        if event.key == 'p' and self.stream is not None:
            # Mark the point when 'P' key is pressed
            for i, ax in enumerate(self.axs):
                if self.vertical_lines[i] is not None:
                    self.vertical_lines[i].remove()
                self.vertical_lines[i] = ax.axvline(event.xdata, color='blue', linestyle='--')
            self.selected_sample = event.xdata
            self.canvas.draw()

            # Print selected point (for debugging)
            print(f"Marked sample at time: {event.xdata}")

        elif event.key == 's' and self.stream is not None:
            # Mark another point when 'S' key is pressed
            for i, ax in enumerate(self.axs):
                if self.red_lines[i] is not None:
                    self.red_lines[i].remove()
                self.red_lines[i] = ax.axvline(event.xdata, color='red', linestyle='--')
            self.selected_sample_s = event.xdata
            self.canvas.draw()

            # Print selected point (for debugging)
            print(f"Marked sample with 'S' at time: {event.xdata}")

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

        dpz_trace = self.stream.select(component="Z")[0]
        base_path = os.path.splitext(self.file_path)[0][:-3]
        default_name = os.path.basename(base_path) + "pick.SAC"  # Only file name
        save_path = filedialog.asksaveasfilename(defaultextension=".SAC", initialfile=default_name, filetypes=[("SAC files", "*.SAC")])
        if not save_path:
            return

        # Update header with selected samples
        if self.selected_sample is not None:
            dpz_trace.stats.sac['user1'] = float(self.selected_sample)
        if self.selected_sample_s is not None:
            dpz_trace.stats.sac['user2'] = float(self.selected_sample_s)

        # Write to new SAC file
        dpz_trace.write(save_path, format='SAC')
        tk.messagebox.showinfo("Info", "File saved successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SeismicGUI(root)
    root.mainloop()
