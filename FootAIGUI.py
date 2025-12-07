import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from analyze.analyze import Analyze
import os
import subprocess
import platform

class FootAIGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("FootAI - Match Analysis")
        self.window.geometry("1200x800")

        self.analyzer = None
        self.selected_video_file = None
        self.config_file = "config.yaml"  # Domyślny config

        self._create_widgets()

    def _create_widgets(self):
        # Nagłówek
        header = tk.Label(self.window, text="FootAI - Video Analysis Tool",
                         font=("Arial", 16, "bold"))
        header.pack(pady=20)

        # Frame do wyboru config
        config_frame = tk.Frame(self.window)
        config_frame.pack(pady=5, padx=20, fill=tk.X)

        tk.Label(config_frame, text="Config:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.config_label = tk.Label(config_frame, text=self.config_file, fg="green")
        self.config_label.pack(side=tk.LEFT, padx=10)

        config_btn = tk.Button(config_frame, text="Change the config",
                              command=self._select_config)
        config_btn.pack(side=tk.RIGHT)

        # Frame do wyboru pliku video
        file_frame = tk.Frame(self.window)
        file_frame.pack(pady=10, padx=20, fill=tk.X)

        self.file_label = tk.Label(file_frame, text="Video not selected",
                                   fg="gray", anchor="w")
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        select_btn = tk.Button(file_frame, text="Choose Video",
                              command=self._select_video)
        select_btn.pack(side=tk.RIGHT, padx=5)

        # Przycisk analizy
        self.analyze_btn = tk.Button(self.window, text="Start the analysis",
                                     command=self._start_analysis,
                                     state=tk.DISABLED, bg="#4CAF50",
                                     fg="white", font=("Arial", 12))
        self.analyze_btn.pack(pady=20)

        # Pasek postępu
        self.progress = ttk.Progressbar(self.window, mode='indeterminate')
        self.progress.pack(pady=10, padx=20, fill=tk.X)

        # Obszar logów
        log_frame = tk.Frame(self.window)
        log_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        tk.Label(log_frame, text="Logs:", font=("Arial", 10, "bold")).pack(anchor="w")

        self.log_text = tk.Text(log_frame, height=10, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(self.log_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)

    def _select_config(self):
        file_path = filedialog.askopenfilename(
            title="Choose Configuration File",
            filetypes=[
                ("YAML files", "*.yaml *.yml"),
                ("All files", "*.*")
            ],
            initialfile="config.yaml"
        )

        if file_path:
            self.config_file = file_path
            self.config_label.config(text=file_path)
            self._log(f"Chosen config: {file_path}")

    def _select_video(self):
        file_path = filedialog.askopenfilename(
            title="Choose Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.selected_video_file = file_path
            self.file_label.config(text=file_path, fg="black")
            self.analyze_btn.config(state=tk.NORMAL)
            self._log(f"Chosen vidoe: {file_path}")

    def _start_analysis(self):
        if not self.selected_video_file:
            messagebox.showwarning("No file", "Choose a video file first!")
            return

        self.analyze_btn.config(state=tk.DISABLED)
        self.progress.start()

        # Uruchom analizę w osobnym wątku
        thread = threading.Thread(target=self._run_analysis)
        thread.daemon = True
        thread.start()

    def _run_analysis(self):
        try:
            self.analyzer = Analyze(self.config_file)

            self._log(f"Starting Analysis: {self.selected_video_file}")
            self.analyzer.run(self.selected_video_file)

            self._log("Analysis Completed!")
            self._display_log_file()
            messagebox.showinfo("Success", "Analysis completed!")
            self._open_outputs_folder()
        except Exception as e:
            self._log(f"Error: {str(e)}")
            messagebox.showerror("Error", f"There was an error:\n{str(e)}")

        finally:
            self.progress.stop()
            self.analyze_btn.config(state=tk.NORMAL)
    def _display_log_file(self):

        try:
            with open("logs/footai.log", "r") as log_file:
                logs = log_file.read()
                self._log("=== Logs file ===")
                self._log(logs)

        except FileNotFoundError:
            self._log("Log file not found.")
    def _log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _open_outputs_folder(self):
        """Otwórz folder outputs w eksploratorze plików"""
        outputs_path = os.path.abspath("outputs")

        if not os.path.exists(outputs_path):
            self._log("outputs folder does not exist.")
            return

        try:
            system = platform.system()

            if system == "Windows":
                os.startfile(outputs_path)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", outputs_path])
            else:  # Linux
                subprocess.run(["xdg-open", outputs_path])

            self._log(f"Folder opened: {outputs_path}")
        except Exception as e:
            self._log(f"Cannot open the folder: {str(e)}")
    def run(self):
        self.window.mainloop()


