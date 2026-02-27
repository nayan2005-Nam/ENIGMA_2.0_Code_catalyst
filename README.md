# Interactive ML Sandbox

This is a simple Flask-based application that lets users explore machine learning
algorithms interactively. It includes a chat-powered AI tutor (Gemini/GenAI) for
asking questions about learning rate, neural networks, decision boundaries, etc.

## Configuration

The app reads configuration from environment variables. The only one you need is:

```
GEMINI_API_KEY=<your Google Cloud API key>
```

Create a file named `.env` in the project root (it is ignored by git) and add:

```
GEMINI_API_KEY=ya29.your-real-key-here
```

Alternatively you can set the variable in your shell before running the server:

- **PowerShell**: `$env:GEMINI_API_KEY="ya29.your-real-key"`
- **bash (WSL/git-bash)**: `export GEMINI_API_KEY="ya29.your-real-key"`

Once the key is present, the backend will forward chat queries to the Gemini
model; if it’s missing, a simple local fallback gives brief answers. You can
verify that the server runs in "online mode" by watching the log output on
startup – it will say either "Gemini API key found; chat will use online mode."
or "No GEMINI_API_KEY provided; using offline fallback responses."

## Running

### Development mode (with automatic reload & debug toolbar):

```bash
set FLASK_ENV=development
python app.py
```

Then open http://127.0.0.1:5000 in your browser (login with admin/password).

### Production mode (safe for deployment):

By default (without `FLASK_ENV=development`), debug mode is off:

```bash
python app.py
```

For actual production deployments, use a WSGI server like Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

This avoids the Flask development server warning and properly handles concurrent requests.

## Notes

- The provided key above (`AIzaSyC6JAB1dlZEEgcnEIGq7LpgMloq-OdfzBs`) will already
  be loaded if you keep it in `.env`, but it’s meant as an example – treat real
  keys as secrets and rotate them when necessary.
- Do **not** check `.env` into source control. The `.gitignore` file excludes it.

## Dependencies

See `requirements.txt` for Python packages. Use a virtual environment or conda
workspace as configured previously.
