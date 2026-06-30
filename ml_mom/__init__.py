"""Future Machine Learning package for Minutes of Meeting generation.

Purpose:
    Contain the complete ML-based MoM pipeline while remaining isolated from
    the existing Streamlit UI and export modules during this skeleton phase.

Responsibilities:
    - Define importable pipeline placeholders.
    - Reserve module boundaries for parsing, preprocessing, embeddings,
      clustering, classification, and template generation.

Inputs:
    Future modules will receive reviewed transcript text and meeting metadata.

Outputs:
    Future modules will produce structured MoM sections.

Future Implementation Notes:
    No real ML algorithms are implemented here yet. Model loading, training,
    inference, and persistence will be added in later phases.
"""

from ml_mom.pipeline import generate_minutes_placeholder

__all__ = ["generate_minutes_placeholder"]
