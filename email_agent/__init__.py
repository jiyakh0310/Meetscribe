"""Future email-agent package for generated email content only.

Purpose:
    Reserve a separate agentic boundary for email body and subject generation.

Responsibilities:
    - Keep email-generation intelligence separate from ML-based MoM generation.
    - Avoid changing the existing SMTP sender during this skeleton phase.

Inputs:
    Future email requests containing recipients, meeting metadata, and MoM
    context.

Outputs:
    Future email drafts containing subject and body text.

Future Implementation Notes:
    Agentic AI may later be used here only for email content. It must not be
    imported into the ML MoM pipeline.
"""

from email_agent.email_schema import EmailDraft, EmailGenerationRequest

__all__ = ["EmailDraft", "EmailGenerationRequest"]
