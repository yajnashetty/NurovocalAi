from flask import Flask, request, render_template, jsonify
import os
import numpy as np
import librosa
import soundfile as sf
import joblib
import parselmouth
from parselmouth.praat import call
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load saved models and objects
interpreter = tflite.Interpreter(model_path='crnn_model.tflite')
interpreter.allocate_tensors()
rf_model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('audio_scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')
rf_feature_cols = joblib.load('rf_feature_columns.joblib')

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/')
def index():
    return render_template('index.html')  # Serves the recording page

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an audio file is uploaded
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded file
    in_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(in_path)

    # If MP3, convert to WAV (e.g., using pydub or ffmpeg via subprocess)
    if file.filename.lower().endswith('.mp3'):
        wav_path = os.path.splitext(in_path)[0] + '.wav'
        from pydub import AudioSegment
        sound = AudioSegment.from_mp3(in_path)
        sound.export(wav_path, format='wav')
    else:
        # If already WAV (or other), ensure WAV format
        wav_path = os.path.splitext(in_path)[0] + '.wav'
        if not file.filename.lower().endswith('.wav'):
            data, sr = librosa.load(in_path, sr=44100)
            sf.write(wav_path, data, sr)
        else:
            wav_path = in_path

    # --- Feature extraction for Random Forest ---
    y, sr = librosa.load(wav_path, sr=44100)
    sound_obj = parselmouth.Sound(wav_path)
    features = {}
    try:
        # Jitter and Shimmer
        pp = call(sound_obj, "To PointProcess (periodic, cc)", 75, 500)
        features['jitter_local'] = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features['shimmer_local'] = call([sound_obj, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        # HNR
        harmonicity = call(sound_obj, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        features['hnr'] = call(harmonicity, "Get mean", 0, 0)
        # Pitch and Intensity stats
        pitch = sound_obj.to_pitch()
        intensity = sound_obj.to_intensity()
        features['mean_f0'] = call(pitch, "Get mean", 0, 0, "Hertz")
        features['std_f0'] = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        features['mean_intensity'] = call(intensity, "Get mean", 0, 0, "energy")
    except Exception:
        # Fallback if parselmouth fails
        features.update({'jitter_local':0, 'shimmer_local':0, 'hnr':0,
                          'mean_f0':0, 'std_f0':0, 'mean_intensity':0})
    # MFCC stats and spectral features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    # Fill missing features if any
    for col in rf_feature_cols:
        features.setdefault(col, 0)
    X_rf = scaler.transform([np.array([features[col] for col in rf_feature_cols])])
    rf_probs = rf_model.predict_proba(X_rf)[0]

    # --- Spectrogram for CRNN ---
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Pad or truncate
    max_len = 250
    if mel_spec_db.shape[1] > max_len:
        mel_spec_db = mel_spec_db[:, :max_len]
    else:
        pad_width = max_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0,0),(0,pad_width)), mode='constant')
    X_crnn = np.expand_dims(mel_spec_db, axis=(0, 3))  # add batch and channel dims

    # Run TFLite prediction
    interpreter.set_tensor(input_details[0]['index'], X_crnn.astype(np.float32))
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    crnn_probs = tflite_output[0]

    # Ensemble averaging
    ensemble_probs = (rf_probs + crnn_probs) / 2.0
    pred_idx = int(np.argmax(ensemble_probs))
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(ensemble_probs[pred_idx])

    # Prepare JSON response
    class_labels = label_encoder.classes_.tolist()
    probs_dict = {class_labels[i]: float(ensemble_probs[i]) for i in range(len(class_labels))}
    return jsonify({
        'label': pred_label,
        'confidence': confidence,
        'probs': probs_dict
    })
if __name__ == '__main__':
    app.run(debug=True)