"""
AI Music Mixer — Backend Python
================================
API REST que recibe dos archivos de audio y produce una mezcla real de ~60 segundos.

Algoritmo:
  1. Librosa analiza BPM y tono de cada canción
  2. Cadenas de Markov sintéticas deciden la curva de crossfade
  3. Equal-power crossfade en PCM con NumPy/SciPy
  4. Devuelve MP3 descargable (sin Spleeter ni TensorFlow para evitar RAM excesiva)

Deploy gratuito: Railway / Render / Fly.io
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import librosa
import numpy as np
import soundfile as sf
import io, os, time, tempfile, logging
from scipy.signal import resample_poly
from math import gcd

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# ── Markov transition matrix para curva de crossfade ────────────────────────
# Estados: 0=silencio, 1=bajo, 2=medio, 3=alto, 4=lleno
# La cadena decide cómo evoluciona el volumen del track-in durante el crossfade
MARKOV = np.array([
    [0.05, 0.60, 0.25, 0.08, 0.02],   # desde silencio
    [0.02, 0.10, 0.55, 0.28, 0.05],   # desde bajo
    [0.01, 0.05, 0.15, 0.55, 0.24],   # desde medio
    [0.00, 0.02, 0.10, 0.35, 0.53],   # desde alto
    [0.00, 0.01, 0.05, 0.20, 0.74],   # desde lleno
], dtype=np.float32)

STATE_VOLUMES = np.array([0.0, 0.20, 0.45, 0.72, 1.0], dtype=np.float32)

def markov_volume_curve(n_steps: int, seed: int = 42) -> np.ndarray:
    """Genera curva de volumen suave usando cadena de Markov."""
    rng = np.random.RandomState(seed)
    state = 0
    raw = []
    for _ in range(n_steps):
        raw.append(STATE_VOLUMES[state])
        state = rng.choice(5, p=MARKOV[state])
    curve = np.array(raw, dtype=np.float32)
    # Suavizar con ventana Hanning para eliminar saltos bruscos
    window = np.hanning(max(n_steps // 8, 5))
    window /= window.sum()
    smooth = np.convolve(curve, window, mode='same')
    # Forzar que empiece en 0 y termine en 1
    smooth = smooth - smooth[0]
    if smooth[-1] != 0:
        smooth /= smooth[-1]
    return smooth.clip(0, 1)

def load_audio(file_bytes: bytes, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """Carga audio desde bytes, mono, resampleado."""
    with tempfile.NamedTemporaryFile(suffix='.audio', delete=False) as f:
        f.write(file_bytes)
        tmp = f.name
    try:
        y, sr = librosa.load(tmp, sr=target_sr, mono=True)
    finally:
        os.unlink(tmp)
    return y, sr

def analyze_bpm_key(y: np.ndarray, sr: int) -> dict:
    """Analiza BPM y tonalidad usando Librosa."""
    # BPM
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo)

    # Tonalidad (chroma)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_idx = int(np.argmax(np.mean(chroma, axis=1)))
    KEYS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    key = KEYS[key_idx]

    # Energía RMS promedio
    rms = float(np.mean(librosa.feature.rms(y=y)))

    return {"bpm": round(bpm, 1), "key": key, "rms": round(rms, 5)}

def normalize(y: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Normaliza audio a nivel target en dB."""
    rms = np.sqrt(np.mean(y**2))
    if rms < 1e-8:
        return y
    target_rms = 10 ** (target_db / 20.0)
    return y * (target_rms / rms)

def time_stretch_to_bpm(y: np.ndarray, src_bpm: float, tgt_bpm: float) -> np.ndarray:
    """Time-stretch ligero para alinear BPMs (máx ±10%)."""
    ratio = tgt_bpm / src_bpm
    ratio = max(0.90, min(1.10, ratio))  # clampeado
    if abs(ratio - 1.0) < 0.01:
        return y
    return librosa.effects.time_stretch(y, rate=ratio)

def mix_tracks(y1: np.ndarray, y2: np.ndarray, sr: int,
               crossfade_sec: float = 30.0,
               total_mix_sec: float = 60.0,
               bpm1: float = 120.0, bpm2: float = 120.0) -> np.ndarray:
    """
    Mezcla real de ~60 segundos:
      - Toma los últimos `crossfade_sec` del track1
      - Toma los primeros `crossfade_sec` del track2
      - Equal-power crossfade con curva Markov en la zona de empalme
      - Ajuste de BPM ligero
    """
    cf_samples = int(crossfade_sec * sr)
    total_samples = int(total_mix_sec * sr)

    # ── Segmentos a usar ────────────────────────────────────────────────────
    # Pre-crossfade: intro del track1 (hasta total_mix_sec - crossfade_sec)
    pre_sec = total_mix_sec - crossfade_sec
    pre_samples = int(pre_sec * sr)

    seg1 = y1[:pre_samples] if len(y1) >= pre_samples else \
           np.pad(y1, (0, pre_samples - len(y1)))

    # Cola del track1 para el crossfade
    tail1_raw = y1[-cf_samples:] if len(y1) >= cf_samples else \
                np.pad(y1, (cf_samples - len(y1), 0))

    # Head del track2 — alineado en BPM
    y2_stretched = time_stretch_to_bpm(y2, bpm2, bpm1)
    head2_raw = y2_stretched[:cf_samples] if len(y2_stretched) >= cf_samples else \
                np.pad(y2_stretched, (0, cf_samples - len(y2_stretched)))

    # ── Curva de volumen Markov ─────────────────────────────────────────────
    n_steps = cf_samples
    fade_in  = markov_volume_curve(n_steps, seed=int(bpm2))          # 0→1 suave
    fade_out = 1.0 - markov_volume_curve(n_steps, seed=int(bpm1))    # 1→0 suave

    # Equal-power encima de la curva Markov
    t = np.linspace(0, np.pi / 2, n_steps, dtype=np.float32)
    ep_out = np.cos(t) * fade_out
    ep_in  = np.sin(t) * fade_in

    crossfade_zone = (tail1_raw * ep_out + head2_raw * ep_in)

    # ── Concatenar ──────────────────────────────────────────────────────────
    mix = np.concatenate([seg1, crossfade_zone])

    # Limitar a total_mix_sec
    mix = mix[:total_samples]
    if len(mix) < total_samples:
        mix = np.pad(mix, (0, total_samples - len(mix)))

    return normalize(mix, target_db=-1.0)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "AI Music Mixer API", "version": "1.0"})

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analiza BPM y key de un track sin mezclar."""
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    data = request.files['file'].read()
    try:
        y, sr = load_audio(data)
        info = analyze_bpm_key(y, sr)
        info['duration_sec'] = round(len(y) / sr, 1)
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/mix', methods=['POST'])
def mix():
    """
    POST /mix
    form-data:
      track1: <audio file>
      track2: <audio file>
      crossfade_sec: float (5-60, default 30)
      ai_mode: bool (default true)
    Devuelve: audio/mpeg (MP3)
    """
    if 'track1' not in request.files or 'track2' not in request.files:
        return jsonify({"error": "Need track1 and track2"}), 400

    crossfade_sec = float(request.form.get('crossfade_sec', 30))
    crossfade_sec = max(5.0, min(60.0, crossfade_sec))
    ai_mode = request.form.get('ai_mode', 'true').lower() == 'true'

    t0 = time.time()
    try:
        data1 = request.files['track1'].read()
        data2 = request.files['track2'].read()

        app.logger.info(f"Loading tracks — sizes: {len(data1)//1024}KB, {len(data2)//1024}KB")
        y1, sr = load_audio(data1)
        y2, _  = load_audio(data2, target_sr=sr)

        app.logger.info(f"Analyzing BPM/key")
        info1 = analyze_bpm_key(y1, sr)
        info2 = analyze_bpm_key(y2, sr)
        app.logger.info(f"Track1: {info1} | Track2: {info2}")

        total_mix_sec = 62.0  # ~62s para tener margen
        mix = mix_tracks(y1, y2, sr,
                         crossfade_sec=crossfade_sec,
                         total_mix_sec=total_mix_sec,
                         bpm1=info1['bpm'], bpm2=info2['bpm'])

        # Exportar como WAV en memoria, luego convertir con pydub a MP3
        wav_buf = io.BytesIO()
        sf.write(wav_buf, mix, sr, format='WAV', subtype='PCM_16')
        wav_buf.seek(0)

        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_wav(wav_buf)
            mp3_buf = io.BytesIO()
            seg.export(mp3_buf, format='mp3', bitrate='192k',
                       tags={'title': 'AI Mix', 'artist': 'AI Music Mixer',
                             'comment': f'BPM1={info1["bpm"]} KEY1={info1["key"]} BPM2={info2["bpm"]} KEY2={info2["key"]}'})
            mp3_buf.seek(0)
            elapsed = round(time.time() - t0, 1)
            app.logger.info(f"Mix done in {elapsed}s")
            resp = send_file(mp3_buf, mimetype='audio/mpeg',
                             as_attachment=True, download_name='ai_mix.mp3')
            resp.headers['X-Mix-BPM1'] = str(info1['bpm'])
            resp.headers['X-Mix-Key1'] = info1['key']
            resp.headers['X-Mix-BPM2'] = str(info2['bpm'])
            resp.headers['X-Mix-Key2'] = info2['key']
            resp.headers['X-Mix-Time'] = str(elapsed)
            return resp
        except Exception:
            # Fallback: devolver WAV si pydub no está disponible
            wav_buf.seek(0)
            return send_file(wav_buf, mimetype='audio/wav',
                             as_attachment=True, download_name='ai_mix.wav')

    except Exception as e:
        app.logger.error(f"Mix error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
