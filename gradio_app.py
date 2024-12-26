import gradio as gr  # For creating a web interface
import torch  # For deep learning model operations
from transformers import WhisperProcessor, WhisperForConditionalGeneration  # For Quranic verse transcription
import pyttsx3  # For text-to-speech functionality
import os  # For file path operations
import json  # For working with JSON data
import torchaudio  # For audio processing
import pandas as pd  # For handling CSV data
from difflib import SequenceMatcher  # For string similarity calculation
import threading  # For running background tasks
from playsound import playsound  # For playing audio files

# Initialize the Whisper processor and model for Quranic verse transcription
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

# Function to load all Surahs and their verses from JSON files
def load_all_surahs():
    quran_text_arabic = {}
    for surah_name, json_file in surah_files.items():
        with open(json_file, "r", encoding="utf-8") as file:
            quran_data = json.load(file)
            if 'ayahs' in quran_data and isinstance(quran_data['ayahs'], list):
                verses = [ayah['text'] for ayah in quran_data['ayahs']]
                quran_text_arabic[surah_name] = verses
            else:
                print(f"Unexpected structure in {json_file}")  # Handle unexpected JSON formats
    return quran_text_arabic

# Load the Quranic verses into memory
quran_text_arabic = load_all_surahs()

# Define the CNN model for reciter classification
class ReciterCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(ReciterCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # First convolutional layer
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Second convolutional layer
        self.fc1 = torch.nn.Linear(32 * 16 * 50, 128)  # Fully connected layer 1
        self.fc2 = torch.nn.Linear(128, num_classes)  # Fully connected layer 2 for classification

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load and process the CSV file for reciter classification
base_path = "Quran_Reciters_Classification/audios"
csv_path = "Quran_Reciters_Classification/files_paths.csv"
df = pd.read_csv(csv_path)
df['FilePath'] = df['FilePath'].apply(lambda x: os.path.join(base_path, '/'.join(x.split('/')[2:])))
df['Exists'] = df['FilePath'].apply(os.path.exists)  # Check if files exist
df = df[df['Exists']].drop(columns=['Exists'])  # Filter out non-existent files
class_to_idx = {cls: idx for idx, cls in enumerate(df['Class'].unique())}  # Map classes to indices
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}  # Map indices back to classes

# Instantiate and load the pre-trained reciter classification model
num_classes = len(class_to_idx)
reciter_model = ReciterCNN(num_classes)
reciter_model.load_state_dict(torch.load("Quran_Reciters_Classification/model.pth"))
reciter_model.eval()  # Set the model to evaluation mode

# Function to calculate similarity between two strings
def calculate_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Function to find the most similar Ayah to the given transcription
def find_ayah_arabic(transcription, threshold=0.6):
    matches = []
    for surah, ayahs in quran_text_arabic.items():
        for ayah_num, ayah in enumerate(ayahs, start=1):
            similarity_score = calculate_similarity(transcription, ayah)
            if similarity_score >= threshold:
                matches.append((surah, ayah_num, similarity_score))
    if matches:
        return sorted(matches, key=lambda x: x[2], reverse=True)  # Sort matches by similarity
    return []

# Function to preprocess audio for classification
def preprocess_audio_fixed_length(file_path, target_sample_rate=16000, n_mels=64, fixed_length=201):
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.size(0) > 1:  # Convert stereo to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != target_sample_rate:  # Resample if needed
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=target_sample_rate, n_mels=n_mels)
    mel_spec = mel_transform(waveform)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    if mel_spec_db.size(-1) < fixed_length:
        mel_spec_db = torch.nn.functional.pad(mel_spec_db, (0, fixed_length - mel_spec_db.size(-1)))
    else:
        mel_spec_db = mel_spec_db[:, :, :fixed_length]  # Truncate to fixed length
    return mel_spec_db

# Predict the reciter from the given audio file
def predict_reciter(file_path):
    mel_spec = preprocess_audio_fixed_length(file_path)
    mel_spec = mel_spec.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = reciter_model(mel_spec)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Transcribe audio to text using the Whisper model
def transcribe_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:  # Resample if needed
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    input_features = processor(waveform.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# Play an audio file in a separate thread
def play_audio(file_path):
    def play():
        try:
            playsound(file_path)
        except Exception as e:
            print(f"Could not play audio: {str(e)}")

    threading.Thread(target=play, daemon=True).start()

# Speak the given text using text-to-speech
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Predict the reciter and transcribe the audio
def browse_and_predict(file):
    if not file:
        return "Please select an audio file."
    
    reciter_idx = predict_reciter(file.name)
    reciter_name = idx_to_class.get(reciter_idx, "Unknown Reciter").replace('_', ' ')
    transcription = transcribe_audio(file.name)
    
    result = f"Reciter: {reciter_name}\nTranscription: {transcription}"
    speak_text(reciter_name)

    return result

# Process the audio file and identify the Quranic Ayah
def process_audio(file):
    if not file:
        return "Please upload an audio file.", "No Ayah identified."
    
    transcription = transcribe_audio(file.name)
    matches = find_ayah_arabic(transcription)
    if matches:
        surah_name, ayah_num, similarity_score = matches[0]
        identified_ayah = quran_text_arabic[surah_name][ayah_num - 1]  # 0-based index
        result = f"Surah: {surah_name}, Ayah: {ayah_num}, Similarity: {similarity_score:.2f}\n\n{identified_ayah}"
        speak_text(result)
    else:
        result = "No matching Ayah found."
    
    return transcription, result

# Function to reset outputs when a new file is uploaded
def reset_outputs(file):
    if file:
        return "", "", ""
    return None, None, None

# Gradio interface for the application
with gr.Blocks() as demo:
    gr.Markdown("# 🔊 Quranic Verses Identification & Reciter Classification 🎤")

    with gr.Row():
        file_input = gr.File(label="Upload an Audio File")

        with gr.Column():
            play_button = gr.Button("Play Audio")
            get_transcript_button = gr.Button("Verse Identification")
            get_reciter_button = gr.Button("Reciter Classification")

    transcription_output = gr.Textbox(label="Transcription")
    ayah_output = gr.Textbox(label="Identified Ayah")
    reciter_output = gr.Textbox(label="Result")

    # Reset textboxes when a new file is uploaded
    file_input.change(reset_outputs, inputs=file_input, outputs=[transcription_output, ayah_output, reciter_output])

    play_button.click(lambda file: play_audio(file.name) if file else None, inputs=file_input)
    get_transcript_button.click(process_audio, inputs=file_input, outputs=[transcription_output, ayah_output])
    get_reciter_button.click(browse_and_predict, inputs=file_input, outputs=reciter_output)

demo.launch() # Launch the Gradio app