# Agentic RAG: split deployment (frontend on Vercel, backend in Docker)

```
agentic-rag-split/
├── backend/    FastAPI + LangGraph + OCR  ->  deploy to Render (Docker) or Cloud Run
│   ├── server.py  Dockerfile  requirements.txt  .env.example
│   └── knowledge/   (handbook is indexed from here at startup)
└── frontend/   static page  ->  deploy to Vercel
    ├── index.html
    └── config.js    <- the ONLY file you edit: put your backend URL here
```

## 1. Deploy the backend (Render, Docker runtime)
1. Push this whole folder to a GitHub repo.
2. Render dashboard -> New -> Web Service -> pick the repo.
3. Settings: **Language = Docker**, **Root Directory = backend**.
4. Environment variables: `GROQ_API_KEY` = your key, `ALLOWED_ORIGINS` = `*` for now.
5. Health check path: `/api/health`. Deploy and wait for the build (the Dockerfile installs Tesseract, so the first build is slow).
6. Open `https://YOUR-SERVICE.onrender.com/api/health` -> should show `{"ok":true}`.
   Open `.../api/stats` -> should show chunks, `"ocr": true`.

## 2. Point the frontend at it
Edit `frontend/config.js`:
```js
window.API_BASE = "https://YOUR-SERVICE.onrender.com";
```
Commit and push.

## 3. Deploy the frontend (Vercel)
1. Vercel -> Add New -> Project -> import the same repo.
2. **Root Directory = frontend**   (this is what avoids the 500 MB Python error: Vercel then sees only static files)
3. Framework Preset = Other. Leave Build Command and Output Directory empty. Deploy.

## 4. Lock down CORS
Back on Render, change `ALLOWED_ORIGINS` to your Vercel URL (e.g. `https://my-app.vercel.app`, no trailing slash) and let it redeploy.
Several URLs are allowed, comma-separated.

## Check it works
- Page shows "online · N chunks" in the top-right (it may say "waking server…" for up to a minute on a free instance).
- Upload dialog says "OCR is on".
- Ask: "How many privilege leaves do employees get each year?"

## Troubleshooting
| Symptom | Cause / fix |
|---|---|
| "server offline" forever | Wrong `API_BASE` in config.js, or backend crashed: check Render logs |
| Browser console: "blocked by CORS policy" | `ALLOWED_ORIGINS` doesn't match your Vercel URL exactly (https, no trailing slash) |
| "OCR is off" | Render service isn't using Docker, or the build skipped the tesseract step |
| Vercel still shows the Lambda size error | Root Directory isn't set to `frontend` |
| First load very slow | Free instances sleep when idle; the server also re-indexes the handbook on every start |
