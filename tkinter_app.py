import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import arabic_reshaper
import onnxruntime
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
import os
import winsound
import pyttsx3
from difflib import SequenceMatcher
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import threading
import bidi.algorithm as bidialg
from loding import load_all_surahs

class QuranicApp:
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained("tarteel-ai/whisper-tiny-ar-quran")
        self.model = WhisperForConditionalGeneration.from_pretrained("tarteel-ai/whisper-tiny-ar-quran")
        self.tts_engine = pyttsx3.init()
        self.quran_text = load_all_surahs()
        df = pd.read_csv("Quran_Reciters_Classification/files_paths.csv")
        self.class_to_idx = {cls: idx for idx, cls in enumerate(df['Class'].unique())}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def transcribe_audio(self, file_path):
        y, _ = librosa.load(file_path, sr=16000)
        input = self.processor(y, sampling_rate=16000, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)
        predicted_ids = self.model.generate(input['input_features'], attention_mask=input['attention_mask'])
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    def find_ayah_arabic(self, transcription, threshold=0.6):
        matches = []
        for surah, ayahs in self.quran_text.items():
            for ayah_num, ayah in enumerate(ayahs, start=1):
                similarity = SequenceMatcher(None, transcription, ayah).ratio()
                if similarity >= threshold:
                    matches.append((surah, ayah_num, similarity))
        return sorted(matches, key=lambda x: x[2], reverse=True) if matches else []

    def preprocess_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        fixed_length = 201
        
        if mel_spec_db.shape[1] < fixed_length:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, fixed_length - mel_spec_db.shape[1])))
        else:
            mel_spec_db = mel_spec_db[:, :fixed_length]
        return mel_spec_db[np.newaxis, np.newaxis, :, :]

    def predict_reciter(self, file_path):
        mel_spec = self.preprocess_audio(file_path)
        session = onnxruntime.InferenceSession("Quran_Reciters_Classification/reciter_model.onnx")
        prediction = session.run(None, {session.get_inputs()[0].name: mel_spec.astype(np.float32)})
        return np.argmax(prediction[0], axis=1)[0]

    def speak_text(self, text):
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

class QuranUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Quranic App")
        self.root.geometry("1200x800")
        self.app = QuranicApp()
        self.setup_ui()

    def setup_ui(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # Create tabs
        self.verse_tab = ttk.Frame(notebook)
        self.reciter_tab = ttk.Frame(notebook)
        notebook.add(self.verse_tab, text='Verse Identifier')
        notebook.add(self.reciter_tab, text='Reciter Classifier')

        self.setup_verse_tab()
        self.setup_reciter_tab()

    def setup_verse_tab(self):
        # Recording controls
        self.record_button = ttk.Button(self.verse_tab, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(pady=10)

        # Results display
        self.verse_result = tk.Text(self.verse_tab, height=10, width=70)
        self.verse_result.pack(pady=10)
        
        self.is_recording = False
        self.audio_data = []

    def setup_reciter_tab(self):
        # File selection
        ttk.Button(self.reciter_tab, text="Select Audio File", command=self.select_file).pack(pady=10)
        ttk.Button(self.reciter_tab, text="Predict Reciter", command=self.predict_reciter).pack(pady=10)
        
        self.reciter_result = tk.Text(self.reciter_tab, height=10, width=70)
        self.reciter_result.pack(pady=10)
        
        self.selected_file = None

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.audio_data = []
        self.record_button.config(text="Stop Recording")
        self.verse_result.delete(1.0, tk.END)
        self.verse_result.insert(tk.END, "Recording...")
        
        sd.default.samplerate = 16000
        sd.default.channels = 1
        self.stream = sd.InputStream(callback=self.audio_callback)
        self.stream.start()

    def stop_recording(self):
        self.is_recording = False
        self.record_button.config(text="Start Recording")
        self.stream.stop()
        self.process_recording()

    def audio_callback(self, indata, frames, time, status):
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def process_recording(self):
        if not self.audio_data:
            return

        temp_file = "temp_recording.wav"
        try:
            audio_array = np.concatenate(self.audio_data, axis=0)
            sf.write(temp_file, audio_array, 16000)
            
            transcription = self.app.transcribe_audio(temp_file)
            matches = self.app.find_ayah_arabic(transcription)
            
            self.verse_result.delete(1.0, tk.END)
            if matches:
                surah, ayah_num, score = matches[0]
                result = f"Surah: {surah}\nAyah: {ayah_num}\nConfidence: {score:.2f}\n\n"
                result += f"Transcription:\n{bidialg.get_display(arabic_reshaper.reshape(transcription))}\n\n"
                result += f"Matching Ayah:\n{self.app.quran_text[surah][ayah_num - 1]}"
                self.verse_result.insert(tk.END, result)
                self.app.speak_text(f"Found match in Surah {surah}, Ayah {ayah_num}")
            else:
                self.verse_result.insert(tk.END, "No matching verse found")
                self.app.speak_text("No matching verse found")
                
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def select_file(self):
        self.selected_file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if self.selected_file:
            self.reciter_result.delete(1.0, tk.END)
            self.reciter_result.insert(tk.END, f"Selected file: {os.path.basename(self.selected_file)}")

    def predict_reciter(self):
        if not self.selected_file:
            messagebox.showerror("Error", "Please select an audio file first")
            return

        try:
            reciter_idx = self.app.predict_reciter(self.selected_file)
            reciter_name = self.app.idx_to_class[reciter_idx].replace('_', ' ')
            transcription = self.app.transcribe_audio(self.selected_file)
            
            result = f"Reciter: {reciter_name}\n\nTranscription:\n"
            result += bidialg.get_display(arabic_reshaper.reshape(transcription))
            
            self.reciter_result.delete(1.0, tk.END)
            self.reciter_result.insert(tk.END, result)
            
            self.app.speak_text(f"The reciter is {reciter_name}")
            threading.Thread(target=lambda: winsound.PlaySound(self.selected_file, winsound.SND_FILENAME)).start()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = QuranUI()
    app.run()