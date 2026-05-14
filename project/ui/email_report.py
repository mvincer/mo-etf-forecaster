"""Send analysis summaries via Gmail SMTP (app password)."""

from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_html_email(
    *,
    to_addr: str,
    subject: str,
    html_body: str,
    text_body: str,
    smtp_user: str | None = None,
    smtp_password: str | None = None,
) -> None:
    password = smtp_password or os.environ.get("GMAIL_APP_PASSWORD")
    user = smtp_user or os.environ.get("GMAIL_USER")
    if not password or not user:
        raise RuntimeError(
            "Missing Gmail credentials. Set GMAIL_USER and GMAIL_APP_PASSWORD in the environment "
            "(Gmail: Google Account → Security → 2-Step Verification → App passwords)."
        )
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(user, [to_addr], msg.as_string())
