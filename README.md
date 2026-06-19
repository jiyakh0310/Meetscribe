# MeetScribe

AI-powered Minutes of Meeting generator using Sarvam transcription and Gemini
meeting analysis.

## Streamlit Community Cloud Deployment

Use `app/main.py` as the Streamlit entry point.

Required secrets:

```toml
SARVAM_API_KEY = "your-sarvam-key"
GEMINI_API_KEY = "your-gemini-key"
```

Audio preprocessing supports `.aac`, `.m4a`, `.mp3`, `.mp4`, and `.wav`.
The app uses `imageio-ffmpeg` to provide a cloud-compatible FFmpeg binary from
Python dependencies, so no manual FFmpeg installation is required.

`packages.txt` also includes `ffmpeg` as a Streamlit Cloud apt dependency. This
gives the deployment a managed system FFmpeg fallback while keeping setup fully
defined in the repository.

Deployment files:

- `requirements.txt` installs Python dependencies.
- `packages.txt` installs apt packages on Streamlit Community Cloud.
- `app/main.py` is the app entry point.

After pushing changes, reboot the Streamlit Cloud app so it reinstalls both
Python and apt dependencies.
