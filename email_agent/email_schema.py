"""Schemas for the future email generation agent.

Purpose:
    Define data contracts for generating professional email content while
    keeping SMTP sending unchanged.

Responsibilities:
    - Represent input required to draft an email.
    - Represent generated subject and body content.
    - Avoid coupling email generation to the SMTP backend.

Inputs:
    Recipient information, meeting title, and optional MoM context.

Outputs:
    Email draft data that can later be passed to the existing sender.

Future Implementation Notes:
    These schemas should remain provider-neutral if an email agent is added.
"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class EmailGenerationRequest:
    """Request for future professional email content generation.

    Attributes:
        recipients: Email recipient addresses or display names.
        meeting_title: Meeting title used to personalize the email.
        mom_summary: Optional Minutes of Meeting context.
        tone: Desired email tone.
        metadata: Optional extra context for future email personalization.
    """

    recipients: list[str] = field(default_factory=list)
    meeting_title: str | None = None
    mom_summary: str | None = None
    tone: str = "professional"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class EmailDraft:
    """Generated email draft content.

    Attributes:
        subject: Email subject line.
        body: Email body text.
    """

    subject: str
    body: str
