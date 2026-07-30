# MeetScribe

MeetScribe is a Streamlit-based Minutes of Meeting application. Audio
transcription is handled through Sarvam, and production Minutes of Meeting
generation now uses the local ML pipeline in `ml_mom/`:

Transcript -> parser -> preprocessing -> feature extraction -> sentence
embeddings -> clustering -> ANN prediction -> rule-based MoM generator ->
`MeetingAnalysisResult`.

The legacy Agentic/Gemini implementation is retained in the repository only for
manual review and rollback context. It is not called by the production
Streamlit entry point.

## Streamlit Community Cloud Deployment

Use `app/main.py` as the Streamlit entry point.

Required secrets:

```toml
SARVAM_API_KEY = "your-sarvam-key"
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

Runtime model artifacts:

- `datasets/models/best_model.pt`
- `datasets/models/label_mapping.json`

Training and annotation utilities are kept separate from production report
generation under `ml_mom/annotation_*.py`, `ml_mom/merge_annotations.py`,
`ml_mom/training_dataset.py`, and `ml_mom/train_ann.py`.

## Optional local Gemma refinement

After deterministic formatting, MeetScribe can polish the Minutes of Meeting
with a Gemma model already installed in local Ollama. No transcript or ML
internals are sent to the model. If Ollama or Gemma is unavailable, the
deterministic output is used automatically.

```dotenv
USE_LOCAL_GEMMA=true
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_GEMMA_MODEL=gemma3:4b
OLLAMA_TIMEOUT_SECONDS=45
```

Set `USE_LOCAL_GEMMA=false` to bypass the rewriter. MeetScribe selects the
configured installed Gemma model, or another installed model whose name
contains `gemma`; it never downloads a model.

After pushing changes, reboot the Streamlit Cloud app so it reinstalls both
Python and apt dependencies.
