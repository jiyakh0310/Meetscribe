"""SMTP email delivery for generated MeetScribe reports."""

from __future__ import annotations

import os
import re
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
EMAIL_PATTERN = re.compile(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$", re.IGNORECASE)
SMTP_ENV_VARS = ("SMTP_SERVER", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD")


class EmailValidationError(ValueError):
    """Raised when user-provided email fields are invalid."""


class SMTPConfigurationError(ValueError):
    """Raised when SMTP configuration is missing or invalid."""


class EmailDeliveryError(RuntimeError):
    """Raised when SMTP delivery fails."""


@dataclass(frozen=True, slots=True)
class SMTPConfig:
    server: str
    port: int
    username: str
    password: str


def send_report_email(
    *,
    recipient: str,
    cc: str = "",
    subject: str,
    message: str,
    attachment_path: str | Path,
) -> None:
    """Validate inputs and send a generated report PDF via SMTP."""

    recipients = _parse_email_list(recipient, required=True)
    cc_recipients = _parse_email_list(cc, required=False)
    subject = subject.strip()
    message = message.strip()
    attachment = Path(attachment_path)

    if not subject:
        raise EmailValidationError("Email subject cannot be empty.")
    if not message:
        raise EmailValidationError("Email message cannot be empty.")
    if not attachment.is_file():
        raise EmailDeliveryError("The Minutes of Meeting PDF could not be found.")

    config = load_smtp_config()
    email = EmailMessage()
    email["From"] = config.username
    email["To"] = ", ".join(recipients)
    if cc_recipients:
        email["Cc"] = ", ".join(cc_recipients)
    email["Subject"] = subject
    email.set_content(message)
    email.add_attachment(
        attachment.read_bytes(),
        maintype="application",
        subtype="pdf",
        filename=attachment.name,
    )

    try:
        smtp_factory = smtplib.SMTP_SSL if config.port == 465 else smtplib.SMTP
        with smtp_factory(config.server, config.port, timeout=30) as smtp:
            smtp.ehlo()
            if config.port not in {25, 465}:
                smtp.starttls()
                smtp.ehlo()
            smtp.login(config.username, config.password)
            smtp.send_message(email, to_addrs=[*recipients, *cc_recipients])
    except smtplib.SMTPException as exc:
        raise EmailDeliveryError("SMTP delivery failed. Please check the SMTP settings and try again.") from exc
    except OSError as exc:
        raise EmailDeliveryError("Could not connect to the SMTP server. Please check the SMTP settings.") from exc


def load_smtp_config() -> SMTPConfig:
    """Load SMTP settings from environment variables or the project .env file."""

    values = {name: _read_config_value(name) for name in SMTP_ENV_VARS}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise SMTPConfigurationError(
            "Email is not configured. Please set SMTP_SERVER, SMTP_PORT, "
            "SMTP_USERNAME, and SMTP_PASSWORD."
        )

    try:
        port = int(values["SMTP_PORT"] or "")
    except ValueError as exc:
        raise SMTPConfigurationError("SMTP_PORT must be a valid number.") from exc

    return SMTPConfig(
        server=str(values["SMTP_SERVER"]),
        port=port,
        username=str(values["SMTP_USERNAME"]),
        password=str(values["SMTP_PASSWORD"]),
    )


def _parse_email_list(value: str, *, required: bool) -> list[str]:
    addresses = [
        item.strip()
        for item in re.split(r"[,;]", value or "")
        if item.strip()
    ]
    if required and not addresses:
        raise EmailValidationError("Recipient email is required.")

    invalid = [address for address in addresses if not EMAIL_PATTERN.match(address)]
    if invalid:
        raise EmailValidationError(f"Invalid email address: {invalid[0]}")
    return addresses


def _read_config_value(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    if value:
        return value
    dotenv_value = _read_dotenv_value(name)
    return dotenv_value or None


def _read_dotenv_value(name: str) -> str | None:
    if not ENV_FILE.is_file():
        return None

    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() != name:
            continue
        value = value.strip().strip('"').strip("'")
        return value or None
    return None
