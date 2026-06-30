"""Fallback messages for the future ML Minutes of Meeting pipeline.

Purpose:
    Centralize friendly validation text used when transcript structure is not
    suitable for reliable MoM generation.

Responsibilities:
    - Provide consistent user-facing messages.
    - Avoid scattered hard-coded validation text across pipeline modules.

Inputs:
    Optional validation context from parser or preprocessing stages.

Outputs:
    Human-readable messages safe for display in the UI.

Future Implementation Notes:
    Messages can later be localized or mapped to UI-specific severity levels.
"""


def speaker_information_missing_message() -> str:
    """Return the standard message for missing speaker information.

    Returns:
        Friendly guidance explaining how users can add speaker labels.
    """

    return (
        "Speaker information could not be identified in this transcript. "
        'Please add speaker labels such as "Speaker 1:", "Rahul:", or '
        '"Priya [00:10]" before generating Minutes of Meeting.'
    )
