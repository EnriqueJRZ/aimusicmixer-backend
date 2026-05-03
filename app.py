"""
AI Music Mixer — Backend Python
Mezcla real de ~60s con Librosa + Cadenas de Markov
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import io, os, time, tempfile, logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# ── Importar librerías pesadas lazy (no al arrancar) ─────────────────────────
_librosa = None
_sf = None

def get_librosa():
    global _librosa
    if _librosa is None:
        import librosa as _lib
        _librosa = _lib
    return _librosa

def get_sf():
    global _sf
    if _sf is None:
        import soundfile as sf_lib
        _sf = sf_lib
    return _sf

# ── Markov matrix ────────────────────────────────────────────────────────────
MARKOV = np.array([
    [0.05, 0.60, 0.25, 0.08, 0.02],
    [0.02, 0.10, 0.55, 0.28, 0.05],
    [0.01, 0.05, 0.15, 0.55, 0.24],
    [0.00, 0.02, 0.10, 0.35, 0.53],
    [0.00, 0.01, 0.05, 0.20, 0.74],
], dtype=np.float32)

STATE_VOLS = np.array([0.0, 0.20, 0.45, 0.72, 1.0], dtype=np.float32)

def markov_curve(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    state = 0
    raw = []
    for _ in range(n):
        raw.append(STATE_VOLS[state])
        state = rng.choice(5, p=MARKOV[state])
    curve = np.array(raw, dtype=np.float32)
    win = np.hanning(max(n // 8, 5))
    win /= win.sum()
    smooth = np.convolve(curve, win, mode='same')
    smooth -= smooth[0]
    if smooth[-1] > 0:
        smooth /= smooth[-1]
    return smooth.clip(0, 1)

def load_audio(file_bytes: bytes, sr: int = 44100):
    lib = get_librosa()
    with tempfile.NamedTemporaryFile(suffix='.tmp', delete=False) as f:
        f.write(file_bytes)
        tmp = f.name
    try:
        y, sr_out = lib.load(tmp, sr=sr, mono=True)
    finally:
        try: os.unlink(tmp)
        except: pass
    return y, sr_out

def analyze(y, sr):
    lib = get_librosa()
    tempo, _ = lib.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo)
    chroma = lib.feature.chroma_cqt(y=y, sr=sr)
    key_idx = int(np.argmax(np.mean(chroma, axis=1)))
    keys = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    return bpm, keys[key_idx]

def normalize(y, db=-1.0):
    rms = np.sqrt(np.mean(y**2))
    if rms < 1e-8: return y
    target = 10 ** (db / 20.0)
    return y * (target / rms)

def do_mix(y1, y2, sr, cf_sec=30.0, bpm1=120.0, bpm2=120.0):
    lib = get_librosa()
    total_sec = 62.0
    cf = int(cf_sec * sr)
    pre = int((total_sec - cf_sec) * sr)

    # Segmento principal del track 1
    seg1 = y1[:pre] if len(y1) >= pre else np.pad(y1, (0, pre - len(y1)))

    # Cola del track 1
    tail1 = y1[-cf:] if len(y1) >= cf else np.pad(y1, (cf - len(y1), 0))

    # Head del track 2 con time-stretch para alinear BPM
    ratio = np.clip(bpm1 / max(bpm2, 1.0), 0.90, 1.10)
    y2s = lib.effects.time_stretch(y2, rate=ratio) if abs(ratio - 1.0) > 0.01 else y2
    head2 = y2s[:cf] if len(y2s) >= cf else np.pad(y2s, (0, cf - len(y2s)))

    # Curvas de crossfade
    t = np.linspace(0, np.pi / 2, cf, dtype=np.float32)
    mk_in  = markov_curve(cf, seed=int(bpm2))
    mk_out = 1.0 - markov_curve(cf, seed=int(bpm1))
    fade_out = np.cos(t) * mk_out
    fade_in  = np.sin(t) * mk_in

    zone = tail1 * fade_out + head2 * fade_in
    mix = np.concatenate([seg1, zone])
    mix = mix[:int(total_sec * sr)]
    return normalize(mix)

@app.route('/health')
def health():
    return jsonify({"status": "ok", "service": "AI Music Mixer API", "version": "1.0"})

@app.route('/analyze', methods=['POST'])
def analyze_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    try:
        y, sr = load_audio(request.files['file'].read())
        bpm, key = analyze(y, sr)
        return jsonify({"bpm": round(bpm, 1), "key": key,
                        "duration_sec": round(len(y) / sr, 1)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/mix', methods=['POST'])
def mix_route():
    if 'track1' not in request.files or 'track2' not in request.files:
        return jsonify({"error": "Need track1 and track2"}), 400

    cf_sec = float(request.form.get('crossfade_sec', 30))
    cf_sec = max(5.0, min(60.0, cf_sec))

    try:
        t0 = time.time()
        app.logger.info("Loading tracks...")
        y1, sr = load_audio(request.files['track1'].read())
        y2, _  = load_audio(request.files['track2'].read(), sr=sr)

        app.logger.info("Analyzing BPM...")
        bpm1, key1 = analyze(y1, sr)
        bpm2, key2 = analyze(y2, sr)
        app.logger.info(f"A:{bpm1}BPM {key1}  B:{bpm2}BPM {key2}")

        app.logger.info("Mixing...")
        mix = do_mix(y1, y2, sr, cf_sec=cf_sec, bpm1=bpm1, bpm2=bpm2)

        # Exportar WAV en memoria
        sf = get_sf()
        buf = io.BytesIO()
        sf.write(buf, mix, sr, format='WAV', subtype='PCM_16')
        buf.seek(0)

        # Convertir a MP3 con pydub si está disponible
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_wav(buf)
            out = io.BytesIO()
            seg.export(out, format='mp3', bitrate='192k')
            out.seek(0)
            mime = 'audio/mpeg'
            fname = 'ai_mix.mp3'
            buf = out
        except Exception:
            buf.seek(0)
            mime = 'audio/wav'
            fname = 'ai_mix.wav'

        elapsed = round(time.time() - t0, 1)
        app.logger.info(f"Done in {elapsed}s")

        resp = send_file(buf, mimetype=mime, as_attachment=True,
                         download_name=fname)
        resp.headers['X-Mix-BPM1'] = str(round(bpm1, 1))
        resp.headers['X-Mix-BPM2'] = str(round(bpm2, 1))
        resp.headers['X-Mix-Key1'] = key1
        resp.headers['X-Mix-Key2'] = key2
        resp.headers['X-Mix-Time'] = str(elapsed)
        return resp

    except Exception as e:
        app.logger.error(f"Mix error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
