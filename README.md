## Sugarcane Disease Detection (Flask + TensorFlow)

This repo contains:
- **Model training / utilities** in `src/`
- A **Flask web app** in `src/webapp/` to upload a leaf image and get a prediction + remedies

### Run locally

1) Create and activate a virtual env, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

2) Start the webapp:

```bash
python src/webapp/app.py
```

Then open `http://127.0.0.1:5000`.

### Notes about deployment (Vercel)

Vercel’s Python serverless environment often **cannot deploy TensorFlow** due to package size / build constraints.
If Vercel fails, use **Render / Railway / Fly.io** for the Flask app instead (they handle heavy Python deps much better).

