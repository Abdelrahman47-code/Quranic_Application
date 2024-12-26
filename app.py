import tkinter as tk  # For creating the GUI application.
from tkinter import filedialog, messagebox, ttk  # For file dialog, message boxes, and advanced GUI widgets.
from tkinter import font as tkFont  # For managing and using custom fonts in the GUI.
import sounddevice as sd  # For recording and handling audio playback.
from transformers import WhisperProcessor, WhisperForConditionalGeneration  # Hugging Face models for audio transcription.
from difflib import SequenceMatcher  # For comparing strings and calculating similarity ratios.
import json  # For handling JSON files (e.g., Quran Surah data).
import pyttsx3  # For text-to-speech functionality.
import os  # For interacting with the operating system (e.g., file paths).
from playsound import playsound  # For playing audio files.
import torchaudio  # For loading and processing audio files.
import torch  # PyTorch library for deep learning.
import numpy as np  # For numerical operations.
import pandas as pd  # For handling structured data (e.g., dataframes for predictions).
import threading  # For managing concurrent execution (e.g., audio playback without freezing the GUI).

# Initialize the model and processor
processor = WhisperProcessor.from_pretrained("tarteel-ai/whisper-tiny-ar-quran")
model = WhisperForConditionalGeneration.from_pretrained("tarteel-ai/whisper-tiny-ar-quran")

# Mapping of Surah names to JSON files
base_path = "Quranic_Verses_Identifier/surahs_json_files/"
surah_files = {
    "Al-Fatiha (الفاتحة)": base_path + "001_al-fatiha.json",
    "Al-Baqarah (البقرة)": base_path + "002_al-baqarah.json",
    "Aal-E-Imran (آل عمران)": base_path + "003_aal-e-imran.json",
    "An-Nisa (النساء)": base_path + "004_an-nisa.json",
    "Al-Maidah (المائدة)": base_path + "005_al-maidah.json",
    "Al-An'am (الأنعام)": base_path + "006_al-anam.json",
    "Al-A'raf (الأعراف)": base_path + "007_al-a'raf.json",
    "Al-Anfal (الأنفال)": base_path + "008_al-anfal.json",
    "At-Tawbah (التوبة)": base_path + "009_at-tawbah.json",
    "Yunus (يونس)": base_path + "010_yunus.json",
    "Hud (هود)": base_path + "011_hud.json",
    "Yusuf (يوسف)": base_path + "012_yusuf.json",
    "Ar-Ra'd (الرعد)": base_path + "013_ar-rad.json",
    "Ibrahim (إبراهيم)": base_path + "014_ibrahim.json",
    "Al-Hijr (الحجر)": base_path + "015_al-hijr.json",
    "An-Nahl (النحل)": base_path + "016_an-nahl.json",
    "Al-Isra (الإسراء)": base_path + "017_al-isra.json",
    "Al-Kahf (الكهف)": base_path + "018_al-kahf.json",
    "Maryam (مريم)": base_path + "019_maryam.json",
    "Ta-Ha (طه)": base_path + "020_ta-ha.json",
    "Al-Anbiya (الأنبياء)": base_path + "021_al-anbiya.json",
    "Al-Hajj (الحج)": base_path + "022_al-hajj.json",
    "Al-Mu'minun (المؤمنون)": base_path + "023_al-mu'minun.json",
    "An-Nur (النور)": base_path + "024_an-nur.json",
    "Al-Furqan (الفرقان)": base_path + "025_al-furqan.json",
    "Ash-Shu'ara (الشعراء)": base_path + "026_ash-shu'ara.json",
    "An-Naml (النمل)": base_path + "027_an-naml.json",
    "Al-Qasas (القصص)": base_path + "028_al-qasas.json",
    "Al-Ankabut (العنكبوت)": base_path + "029_al-ankabut.json",
    "Ar-Rum (الروم)": base_path + "030_ar-rum.json",
    "Luqman (لقمان)": base_path + "031_luqman.json",
    "As-Sajdah (السجدة)": base_path + "032_as-sajdah.json",
    "Al-Ahzab (الأحزاب)": base_path + "033_al-ahzab.json",
    "Saba (سبأ)": base_path + "034_saba.json",
    "Fatir (فاطر)": base_path + "035_fatir.json",
    "Ya-Sin (يس)": base_path + "036_ya-sin.json",
    "As-Saffat (الصافات)": base_path + "037_as-saffat.json",
    "Sad (ص)": base_path + "038_sad.json",
    "Az-Zumar (الزمر)": base_path + "039_az-zumar.json",
    "Ghafir (غافر)": base_path + "040_ghafir.json",
    "Fussilat (فصلت)": base_path + "041_fussilat.json",
    "Ash-Shura (الشورى)": base_path + "042_ash-shura.json",
    "Az-Zukhruf (الزخرف)": base_path + "043_az-zukhruf.json",
    "Ad-Dukhkhan (الدخان)": base_path + "044_ad-dukhkhan.json",
    "Al-Jathiya (الجاثية)": base_path + "045_al-jathiya.json",
    "Al-Ahqaf (الأحقاف)": base_path + "046_al-ahqaf.json",
    "Muhammad (محمد)": base_path + "047_muhammad.json",
    "Al-Fath (الفتح)": base_path + "048_al-fath.json",
    "Al-Hujurat (الحجرات)": base_path + "049_al-hujurat.json",
    "Qaf (ق)": base_path + "050_qaf.json",
    "Adh-Dhariyat (الذاريات)": base_path + "051_adh-dhariyat.json",
    "At-Tur (الطور)": base_path + "052_at-tur.json",
    "An-Najm (النجم)": base_path + "053_an-najm.json",
    "Al-Qamar (القمر)": base_path + "054_al-qamar.json",
    "Ar-Rahman (الرحمن)": base_path + "055_ar-rahman.json",
    "Al-Waqi'ah (الواقعة)": base_path + "056_al-waqi'ah.json",
    "Al-Hadid (الحديد)": base_path + "057_al-hadid.json",
    "Al-Mujadila (المجادلة)": base_path + "058_al-mujadila.json",
    "Al-Hashr (الحشر)": base_path + "059_al-hashr.json",
    "Al-Mumtahina (الممتحنة)": base_path + "060_al-mumtahina.json",
    "As-Saff (الصف)": base_path + "061_as-saff.json",
    "Al-Jumu'ah (الجمعة)": base_path + "062_al-jumu'ah.json",
    "Al-Munafiqoon (المنافقون)": base_path + "063_al-munafiqoon.json",
    "At-Taghabun (التغابن)": base_path + "064_at-taghabun.json",
    "At-Talaq (الطلاق)": base_path + "065_at-talaq.json",
    "At-Tahrim (التحريم)": base_path + "066_at-tahrim.json",
    "Al-Mulk (الملك)": base_path + "067_al-mulk.json",
    "Al-Qalam (القلم)": base_path + "068_al-qalam.json",
    "Al-Haqqah (الحاقة)": base_path + "069_al-haqqah.json",
    "Al-Ma'arij (المعارج)": base_path + "070_al-ma'arij.json",
    "Nooh (نوح)": base_path + "071_nooh.json",
    "Al-Jinn (الجن)": base_path + "072_al-jinn.json",
    "Al-Muzzammil (المزمل)": base_path + "073_al-muzzammil.json",
    "Al-Muddathir (المدثر)": base_path + "074_al-muddathir.json",
    "Al-Qiyamah (القيامة)": base_path + "075_al-qiyamah.json",
    "Al-Insan (الإنسان)": base_path + "076_al-insan.json",
    "Al-Mursalat (المرسلات)": base_path + "077_al-mursalat.json",
    "An-Naba (النبأ)": base_path + "078_an-naba.json",
    "An-Nazi'at (النازعات)": base_path + "079_an-nazi'at.json",
    "Abasa (عبس)": base_path + "080_abasa.json",
    "At-Takwir (التكوير)": base_path + "081_at-takwir.json",
    "Al-Infitar (الإنفطار)": base_path + "082_al-infitar.json",
    "Al-Mutaffifin (المطففين)": base_path + "083_al-mutaffifin.json",
    "Al-Inshiqaq (الإنشقاق)": base_path + "084_al-inshiqaq.json",
    "Al-Buruj (البروج)": base_path + "085_al-buruj.json",
    "At-Tariq (الطارق)": base_path + "086_at-tariq.json",
    "Al-A'la (الأعلى)": base_path + "087_al-a'la.json",
    "Al-Ghashiyah (الغاشية)": base_path + "088_al-ghashiyah.json",
    "Al-Fajr (الفجر)": base_path + "089_al-fajr.json",
    "Al-Balad (البلد)": base_path + "090_al-balad.json",
    "Ash-Shams (الشمس)": base_path + "091_ash-shams.json",
    "Al-Lail (الليل)": base_path + "092_al-layl.json",
    "Ad-Duha (الضحى)": base_path + "093_ad-duha.json",
    "Ash-Sharh (الشرح)": base_path + "094_ash-sharh.json",
    "At-Tin (التين)": base_path + "095_at-tin.json",
    "Al-Alaq (العلق)": base_path + "096_al-'alaq.json",
    "Al-Qadr (القدر)": base_path + "097_al-qadr.json",
    "Al-Bayyina (البينة)": base_path + "098_al-bayyina.json",
    "Az-Zalzalah (الزلزلة)": base_path + "099_az-zalzalah.json",
    "Al-Adiyat (العاديات)": base_path + "100_al-adiyat.json",
    "Al-Qari'ah (القارعة)": base_path + "101_al-qari'ah.json",
    "At-Takathur (التكاثر)": base_path + "102_at-takathur.json",
    "Al-Asr (العصر)": base_path + "103_al-asr.json",
    "Al-Humazah (الهمزة)": base_path + "104_al-humazah.json",
    "Al-Fil (الفيل)": base_path + "105_al-fil.json",
    "Quraish (قريش)": base_path + "106_quraish.json",
    "Al-Ma'un (الماعون)": base_path + "107_al-ma'un.json",
    "Al-Kawthar (الكوثر)": base_path + "108_al-kawthar.json",
    "Al-Kafirun (الكافرون)": base_path + "109_al-kafirun.json",
    "An-Nasr (النصر)": base_path + "110_an-nasr.json",
    "Al-Masad (المسد)": base_path + "111_al-masad.json",
    "Al-Ikhlas (الإخلاص)": base_path + "112_al-ikhlas.json",
    "Al-Falaq (الفلق)": base_path + "113_al-falaq.json",
    "An-Nas (الناس)": base_path + "114_an-nas.json",
}

# Function to load all surahs with their verses
def load_all_surahs():
    quran_text_arabic = {}
    for surah_name, json_file in surah_files.items():
        with open(json_file, "r", encoding="utf-8") as file:
            quran_data = json.load(file)
            if 'ayahs' in quran_data and isinstance(quran_data['ayahs'], list):
                verses = [ayah['text'] for ayah in quran_data['ayahs']]
                quran_text_arabic[surah_name] = verses
            else:
                print(f"Unexpected structure in {json_file}")
    return quran_text_arabic

quran_text_arabic = load_all_surahs()

# Load saved model for reciter classification
class ReciterCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(ReciterCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(32 * 16 * 50, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the base dataset path
base_path = "Quran_Reciters_Classification/audios"
csv_path = "Quran_Reciters_Classification/files_paths.csv"

# Load the CSV file and keep only rows where files exist
df = pd.read_csv(csv_path)
df['FilePath'] = df['FilePath'].apply(lambda x: os.path.join(base_path, '/'.join(x.split('/')[2:])))
df['Exists'] = df['FilePath'].apply(os.path.exists)
missing_files = df[~df['Exists']]
df = df[df['Exists']].drop(columns=['Exists'])

# Create a mapping of class names to indices
class_to_idx = {cls: idx for idx, cls in enumerate(df['Class'].unique())}
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

# Instantiate the model
num_classes = len(class_to_idx)
reciter_model = ReciterCNN(num_classes)
reciter_model.load_state_dict(torch.load("Quran_Reciters_Classification/model.pth"))
reciter_model.eval()


# Main application class
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("🔊 Quranic Verses Identification & Reciter Classification 🎤")
        self.root.state("zoomed")

        # Constants
        self.BG_COLOR = "#ff5733"
        self.BUTTON_COLOR = "#007ACC"
        self.BUTTON_ACTIVE_COLOR = "#005A8B"

        # Configure Fonts
        self.title_font = tkFont.Font(family="Helvetica", size=24, weight="bold")
        self.button_font = tkFont.Font(family="Helvetica", size=14, weight="bold")
        self.label_font = tkFont.Font(family="Helvetica", size=12, weight="bold")

        # Configure Button Style
        self.style = ttk.Style()
        self.style.configure(
            "Rounded.TButton",
            font=self.button_font,
            padding=10,
            background=self.BUTTON_COLOR,
            relief="flat",
        )
        self.style.map(
            "Rounded.TButton",
            background=[("active", self.BUTTON_ACTIVE_COLOR), ("!active", self.BUTTON_COLOR)],
        )

        # Set up the main menu and pages
        self.main_frame = tk.Frame(root, bg=self.BG_COLOR)
        self.main_frame.pack(fill="both", expand=True)
        self.create_main_menu()
        self.ayah_page = self.create_ayah_page(self.main_frame)
        self.reciter_page = self.create_reciter_page(self.main_frame)
        
        self.tts_engine = pyttsx3.init()
        self.selected_file = None

    def create_main_menu(self):
        """Create the main menu with navigation buttons."""
        button_frame = tk.Frame(self.main_frame, bg=self.BG_COLOR)
        button_frame.pack(pady=20, anchor='center')

        title_label = tk.Label(
            button_frame,
            text="🔊 Quranic Verses Identification & Reciter Classification 🎤",
            font=self.title_font,
            bg=self.BG_COLOR
        )
        title_label.pack(pady=20)

        ayah_button = ttk.Button(button_frame, text="Verses Identifier", command=self.show_ayah_page, style="Rounded.TButton")
        ayah_button.pack(side="left", padx=50)

        reciter_button = ttk.Button(button_frame, text="Reciter Classifier", command=self.show_reciter_page, style="Rounded.TButton")
        reciter_button.pack(side="right", padx=50)

        separator = ttk.Separator(button_frame, orient="horizontal")
        separator.pack(fill="x", pady=20)
        separator = ttk.Separator(button_frame, orient="horizontal")
        separator.pack(fill="x", pady=20)

    def show_ayah_page(self):
        """Show Ayah Identifier Page."""
        self.reciter_page.pack_forget()
        self.ayah_page.pack(fill="both", expand=True)

    def show_reciter_page(self):
        """Show Reciter Classification Page."""
        self.ayah_page.pack_forget()
        self.reciter_page.pack(fill="both", expand=True)
        
    def create_ayah_page(self, parent):
        """Create and return the Ayah Identifier Page."""
        ayah_frame = tk.Frame(parent, bg=self.BG_COLOR)

        # Add the Required Buttons and Labels
        self.start_label = self.add_label(ayah_frame, "Click 'Start Recording' to begin.", self.label_font)
        self.start_button = self.create_button(ayah_frame, "🎙 Start Recording", self.start_recording)
        self.stop_button = self.create_button(ayah_frame, "🛑 Stop Recording", self.stop_recording, state=tk.DISABLED)
        self.transcription_label = self.add_label(ayah_frame, "", self.label_font)
        self.ayah_label = self.add_label(ayah_frame, "", self.label_font)    
        ayah_frame.pack(fill="both", expand=True)

        return ayah_frame

    def create_reciter_page(self, parent):
        """Create and return the Reciter Classification Page."""
        reciter_frame = tk.Frame(parent, bg=self.BG_COLOR)

        # Add the Required Buttons and Labels
        self.browse_label = self.add_label(reciter_frame, "Click 'Browse Audio File' to begin.", self.label_font)
        self.create_button(reciter_frame, "Browse Audio File", self.browse_file)
        self.file_label = self.add_label(reciter_frame, "No file selected", self.label_font)
        self.reciter_prediction_label = self.add_label(reciter_frame, "", self.label_font)
        self.create_button(reciter_frame, "Predict", self.predict_file)
        self.result_label = self.add_label(reciter_frame, "", self.label_font)

        return reciter_frame

    def create_button(self, parent, text, command, state=tk.NORMAL):
        """Create and return a styled button."""
        button = ttk.Button(parent, text=text, command=command, style="Rounded.TButton", state=state)
        button.pack(pady=10)
        return button

    def add_label(self, parent, text, font):
        """Create and return a label."""
        label = tk.Label(parent, text=text, font=font, bg=self.BG_COLOR)
        label.pack(pady=10)
        return label

    def add_title(self, parent, text):
        """Add a title label to the parent."""
        tk.Label(parent, text=text, font=self.title_font).pack(pady=20)


    # ----- Verses Identifier -----
    def calculate_similarity(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

    def find_ayah_arabic(self, transcription, threshold=0.6):
        matches = []
        for surah, ayahs in quran_text_arabic.items():
            for ayah_num, ayah in enumerate(ayahs, start=1):
                similarity_score = self.calculate_similarity(transcription, ayah)
                if similarity_score >= threshold:
                    matches.append((surah, ayah_num, similarity_score))
        if matches:
            return sorted(matches, key=lambda x: x[2], reverse=True)  # Return matches sorted by similarity score
        return []

    def start_recording(self):
        self.is_recording = True
        self.audio_data = []
        self.start_button['state'] = tk.DISABLED
        self.stop_button['state'] = tk.NORMAL
        sd.default.samplerate = 16000
        sd.default.channels = 1
        self.stream = sd.InputStream(callback=self.audio_callback)
        self.stream.start()
        self.start_label.config(text="")
        self.start_label.update_idletasks()
        self.transcription_label.config(text="")
        self.transcription_label.update_idletasks()
        self.ayah_label.config(text="Recording...")
        self.ayah_label.update_idletasks()

    def audio_callback(self, indata, frames, time, status):
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def stop_recording(self):
        self.is_recording = False
        self.start_button['state'] = tk.NORMAL
        self.stop_button['state'] = tk.DISABLED
        self.stream.stop()

        # Display the "Recording stopped" message
        self.ayah_label.config(text="Recording stopped ... Result will appear")
        self.ayah_label.update_idletasks()

        if self.audio_data:
            audio_array = np.concatenate(self.audio_data, axis=0)
            audio_tensor = torch.from_numpy(audio_array.flatten())

            # Transcribe the audio
            transcription = self.transcribe_record(audio_tensor)
            self.transcription_label.config(text=f"Transcript\n{transcription}")

            # Match transcription with Quranic text
            best_match = self.find_ayah_arabic(transcription)
            if best_match:
                surah_name, ayah_num, similarity_score = best_match[0]
                identified_ayah = quran_text_arabic[surah_name][ayah_num - 1]  # 0-based index

                self.ayah_label.config(text=f": {identified_ayah}")
                match_text = f"Surah {surah_name}, Ayah {ayah_num}, Similarity: {similarity_score:.2f}.\n\n\nThe matching Ayah is:\n{identified_ayah}"
                self.ayah_label.config(text=match_text)
                self.speak_text(match_text)
            else:
                self.ayah_label.config(text="No matching Ayah found.")
                self.speak_text("No matching Ayah found.")
            
        self.start_label.config(text="Start Recording Again")
        self.start_label.update_idletasks()
    def transcribe_record(self, audio_chunk, sample_rate=16000):
        inputs = processor(audio_chunk.numpy(), sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription

    def update_mode(self):
        self.is_recording = False
        self.audio_data = []
        self.transcriptions = []
        self.current_ayah = 1
        self.transcription_label.config(text="")
        self.ayah_label.config(text="")
        self.ayah_label.config(text="Press 'Start Recording' to begin.")
        
    def speak_text(self, text):
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()


    # ----- Reciteer Classification -----
    def preprocess_audio_fixed_length(self, file_path, target_sample_rate=16000, n_mels=64, fixed_length=201):
        """Preprocess audio file into mel spectrogram for classification."""
        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
        mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=target_sample_rate, n_mels=n_mels)
        mel_spec = mel_transform(waveform)
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        if mel_spec_db.size(-1) < fixed_length:
            mel_spec_db = torch.nn.functional.pad(mel_spec_db, (0, fixed_length - mel_spec_db.size(-1)))
        else:
            mel_spec_db = mel_spec_db[:, :, :fixed_length]
        return mel_spec_db

    def predict_reciter(self, file_path):
        self.result_label.config(text="Prediction started ... Result will appear")
        self.result_label.update_idletasks()

        """Predict reciter from audio file."""
        mel_spec = self.preprocess_audio_fixed_length(file_path)
        mel_spec = mel_spec.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = reciter_model(mel_spec)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()

    def transcribe_audio(self, file_path):
        """Transcribe audio using Whisper model."""
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        input_features = processor(waveform.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt").input_features
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription

    def play_audio(self, file_path):
        """Play the audio file in a separate thread."""
        def play():
            try:
                playsound(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Could not play audio: {str(e)}")

        threading.Thread(target=play, daemon=True).start()

    def browse_file(self):
        self.result_label.config(text="")
        self.result_label.update_idletasks()
    
        """Browse for an audio file."""
        self.selected_file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if self.selected_file:
            self.browse_label.config(text="")
            self.browse_label.update_idletasks()
            self.file_label.config(text=f"Selected File: {os.path.basename(self.selected_file)}")
            self.reciter_prediction_label.config(text="Click 'Predict' to know the reciter.")
            self.reciter_prediction_label.update_idletasks()

    def predict_file(self):
        self.reciter_prediction_label.config(text="")
        self.reciter_prediction_label.update_idletasks()
    
        """Predict reciter and transcribe audio file."""
        if not self.selected_file:
            messagebox.showerror("Error", "Please select an audio file first.")
            return

        try:
            # Predict reciter
            reciter_idx = self.predict_reciter(self.selected_file)
            reciter_name = idx_to_class.get(reciter_idx, "Unknown Reciter").replace('_', ' ')
            transcription = self.transcribe_audio(self.selected_file)

            self.reciter_prediction_label.config(text="")
            self.result_label.config(text=f"Reciter\n{reciter_name}\n\nTranscription\n{transcription}")
            self.speak_text(f"The reciter is {reciter_name}. The transcription is as follows: {transcription}")

            self.play_audio(self.selected_file)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

        self.browse_label.config(text="Browse a new file")
        self.browse_label.update_idletasks()

# Initialize the Tkinter window
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()