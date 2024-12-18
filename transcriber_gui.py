import tkinter as tk
from tkinter import filedialog, Text, messagebox
from tkinter import ttk  # For progress bar
import whisper
import threading
import os
import numpy as np
from tkinter import font

# Load the Whisper model
def load_model():
    try:
        global model
        model = whisper.load_model("large-v2")  # Use "large-v2" for the best model
        print("Whisper model loaded successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load Whisper model: {e}")

# Function to transcribe audio with progress
def transcribe_audio_with_progress(file_path, text_widget, progress_bar, progress_label, stop_event):
    try:
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File not found: {file_path}")
            return

        print(f"Transcribing file: {file_path}")
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, "Transcribing...\n")
        text_widget.update_idletasks()

        # Load the audio and calculate total duration
        audio = whisper.load_audio(file_path)
        duration = audio.shape[-1] / whisper.audio.SAMPLE_RATE  # Duration in seconds

        # Split audio into chunks of 30 seconds
        chunk_size = 30  # 30 seconds
        num_chunks = int(np.ceil(duration / chunk_size))
        transcription = ""

        for i in range(num_chunks):
            if stop_event.is_set():  # Stop transcription if the event is set
                progress_label.config(text="Transcription canceled.")
                return

            start_time = i * chunk_size
            end_time = min((i + 1) * chunk_size, duration)

            # Extract the current chunk
            chunk = audio[int(start_time * whisper.audio.SAMPLE_RATE):int(end_time * whisper.audio.SAMPLE_RATE)]
            chunk = whisper.pad_or_trim(chunk)  # Pad or trim to fit the model input size

            # Transcribe the current chunk
            result = model.transcribe(chunk, language="he")  # Hebrew transcription
            transcription += result["text"] + " "

            # Update progress bar and label
            progress = int((i + 1) / num_chunks * 100)
            progress_bar["value"] = progress
            progress_label.config(text=f"Progress: {progress}%")
            progress_label.update_idletasks()

        # Display the final transcription
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, transcription.strip(), "rtl")  # Apply RTL tag

        # Reset progress bar and label
        progress_bar["value"] = 0
        progress_label.config(text="Transcription complete.")
    except Exception as e:
        progress_bar["value"] = 0
        progress_label.config(text="An error occurred.")
        messagebox.showerror("Error", f"An error occurred: {e}")
        print(f"Error details: {e}")

# Function to select an audio file
def open_file(text_widget, progress_bar, progress_label, stop_event):
    file_path = filedialog.askopenfilename(
        title="Select an Audio File",
        filetypes=[("Audio Files", "*.mp3 *.wav *.m4a *.flac")]
    )
    if file_path:
        stop_event.clear()  # Reset the stop event
        threading.Thread(
            target=transcribe_audio_with_progress, 
            args=(file_path, text_widget, progress_bar, progress_label, stop_event)
        ).start()

# Function to stop the transcription
def stop_transcription(stop_event):
    stop_event.set()

# Create the GUI
def create_gui():
    root = tk.Tk()
    root.title("Audio Transcriber")
    root.geometry("1100x800")  # Make the window bigger

    # Create stop_event to handle stopping transcription
    stop_event = threading.Event()

    # Frame for buttons and text box
    frame = tk.Frame(root)
    frame.pack(padx=20, pady=20, expand=True, fill=tk.BOTH)

    # Button to select an audio file
    select_button = tk.Button(frame, text="Select Audio File", command=lambda: open_file(text_box, progress_bar, progress_label, stop_event))
    select_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

    # Button to stop transcription
    stop_button = tk.Button(frame, text="Stop Transcription", command=lambda: stop_transcription(stop_event))
    stop_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

    # Custom font for Hebrew text
    custom_font = font.Font(family="Helvetica", size=14, weight="normal")

    # Text widget to display the transcription
    text_box = Text(frame, wrap=tk.WORD, width=80, height=20, font=custom_font)
    text_box.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

    # Configure text widget for RTL display
    text_box.tag_configure("rtl", justify="right")  # Align text to the right for RTL

    # Progress bar widget
    progress_bar = ttk.Progressbar(frame, mode="determinate", maximum=100)
    progress_bar.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")

    # Label to display progress percentage
    progress_label = tk.Label(frame, text="Progress: 0%", font=custom_font)
    progress_label.grid(row=3, column=0, columnspan=2, pady=5)

    # Make the frame resize proportionally
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(1, weight=1)

    root.mainloop()

if __name__ == "__main__":
    print("Loading Whisper model...")
    load_model()
    print("Starting the GUI...")
    create_gui()