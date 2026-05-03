# AI Music Mixer — Python Backend

## Deploy gratis en Railway (recomendado)
1. Crea cuenta en railway.app
2. New Project → Deploy from GitHub (sube esta carpeta)
3. Copia la URL pública (ej: https://aimusicmixer.up.railway.app)
4. En Android: cambia API_BASE_URL en AIMixApiClient.kt

## Deploy gratis en Render
1. render.com → New Web Service → GitHub
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `gunicorn app:app --workers 2 --timeout 120`

## Endpoints
- GET  /health       → status ok
- POST /analyze      → {bpm, key, duration_sec}  (form: file=<audio>)
- POST /mix          → MP3 descargable            (form: track1, track2, crossfade_sec, ai_mode)
