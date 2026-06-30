"""Future agentic email generation boundary.

Purpose:
    Reserve the only allowed agentic-AI area in the redesigned architecture:
    email subject and body generation.

Responsibilities:
    - Generate professional email draft content in a later phase.
    - Keep generated content separate from actual SMTP sending.
    - Ensure MoM generation remains fully ML-based and non-agentic.

Inputs:
    ``EmailGenerationRequest`` with recipients, meeting title, and optional MoM
    context.

Outputs:
    ``EmailDraft`` containing subject and body placeholders.

Future Implementation Notes:
    This module must pass future email drafts to the existing SMTP sender only
    after user-facing integration is explicitly implemented.
"""

from email_agent.email_schema import EmailDraft, EmailGenerationRequest


class EmailAgent:
    """Placeholder boundary for future professional email generation."""

    def generate_draft(self, request: EmailGenerationRequest) -> EmailDraft:
        """Generate a placeholder email draft without calling an AI provider.

        Args:
            request: Email generation request with recipients and meeting
                context.

        Returns:
            Placeholder email draft.
        """

        # Keep the request part of the interface now so the future agent can be
        # added without changing callers, but do not invoke AI in this phase.
        _ = request

        # TODO:
        # Add agentic email subject and body generation later.
        return EmailDraft(subject="", body="")
