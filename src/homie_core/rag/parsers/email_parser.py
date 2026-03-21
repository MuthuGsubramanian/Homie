from __future__ import annotations

import email
import re
from email import policy
from pathlib import Path

from homie_core.rag.parsers import ParsedDocument, TextBlock, register_parser


@register_parser("email")
def parse_email(path: Path) -> ParsedDocument:
    raw = path.read_bytes()
    msg = email.message_from_bytes(raw, policy=policy.default)
    subject = msg.get("subject", "")
    sender = msg.get("from", "")
    to = msg.get("to", "")
    date = msg.get("date", "")
    # Extract body
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_content()
                break
            elif part.get_content_type() == "text/html" and not body:
                html = part.get_content()
                try:
                    from bs4 import BeautifulSoup
                    body = BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)
                except ImportError:
                    body = re.sub(r"<[^>]+>", " ", html)
    else:
        body = (
            msg.get_content()
            if hasattr(msg, "get_content")
            else str(msg.get_payload(decode=True) or "")
        )
    header_text = f"From: {sender}\nTo: {to}\nDate: {date}\nSubject: {subject}"
    blocks = [
        TextBlock(content=header_text, block_type="heading", level=1),
        TextBlock(content=body.strip(), block_type="paragraph"),
    ]
    attachments = [part.get_filename() for part in msg.walk() if part.get_filename()]
    return ParsedDocument(
        text_blocks=blocks,
        metadata={
            "format": "email",
            "subject": subject,
            "from": sender,
            "to": to,
            "date": date,
            "attachments": attachments,
        },
        source_path=str(path),
    )
