"""
Email Notifier Module for Ultimate Trading Bot v2.2.

This module provides email notification capabilities including:
- SMTP email sending
- HTML and plain text formatting
- Attachment support
- Template rendering
- Batch email handling
"""

import asyncio
import logging
import smtplib
import ssl
from dataclasses import dataclass, field
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from .base_notifier import (
    BaseNotifier,
    NotificationChannel,
    NotificationMessage,
    NotificationType,
    NotificationPriority,
    NotifierConfig,
    DeliveryResult,
)


logger = logging.getLogger(__name__)


@dataclass
class EmailConfig(NotifierConfig):
    """Email notifier configuration."""

    # SMTP settings
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    use_tls: bool = True
    use_ssl: bool = False

    # Sender settings
    from_email: str = ""
    from_name: str = "Trading Bot"
    reply_to: str | None = None

    # Default recipients
    default_recipients: list[str] = field(default_factory=list)

    # Content settings
    include_html: bool = True
    include_plain: bool = True
    footer_text: str = "Sent by Ultimate Trading Bot v2.2"

    # Tracking
    track_opens: bool = False
    track_clicks: bool = False


class EmailTemplate:
    """Email template for formatting messages."""

    # HTML template
    HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: {header_color};
            color: white;
            padding: 20px;
            border-radius: 8px 8px 0 0;
            margin-bottom: 0;
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
        }}
        .content {{
            background: #f9f9f9;
            padding: 20px;
            border: 1px solid #ddd;
            border-top: none;
        }}
        .priority-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .priority-low {{ background: #e3f2fd; color: #1976d2; }}
        .priority-normal {{ background: #e8f5e9; color: #388e3c; }}
        .priority-high {{ background: #fff3e0; color: #f57c00; }}
        .priority-urgent {{ background: #fce4ec; color: #c2185b; }}
        .priority-critical {{ background: #ffebee; color: #c62828; }}
        .message-body {{
            margin: 15px 0;
            white-space: pre-wrap;
        }}
        .metadata {{
            background: #fff;
            padding: 15px;
            border-radius: 4px;
            margin-top: 15px;
        }}
        .metadata table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .metadata td {{
            padding: 8px;
            border-bottom: 1px solid #eee;
        }}
        .metadata td:first-child {{
            font-weight: bold;
            width: 120px;
        }}
        .footer {{
            text-align: center;
            padding: 15px;
            color: #666;
            font-size: 12px;
            border-top: 1px solid #ddd;
        }}
        .trade-info {{
            background: #fff;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid {accent_color};
        }}
        .pnl-positive {{ color: #388e3c; }}
        .pnl-negative {{ color: #c62828; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
    </div>
    <div class="content">
        <span class="priority-badge priority-{priority}">{priority_label}</span>
        <div class="message-body">
{body}
        </div>
        {metadata_section}
    </div>
    <div class="footer">
        {footer}
        <br>
        <small>{timestamp}</small>
    </div>
</body>
</html>
"""

    # Color mapping for message types
    TYPE_COLORS = {
        NotificationType.INFO: "#2196f3",
        NotificationType.SUCCESS: "#4caf50",
        NotificationType.WARNING: "#ff9800",
        NotificationType.ERROR: "#f44336",
        NotificationType.ALERT: "#e91e63",
        NotificationType.TRADE: "#673ab7",
        NotificationType.ORDER: "#3f51b5",
        NotificationType.SIGNAL: "#009688",
        NotificationType.SYSTEM: "#607d8b",
        NotificationType.PERFORMANCE: "#795548",
    }

    @classmethod
    def render_html(
        cls,
        message: NotificationMessage,
        footer_text: str = "",
    ) -> str:
        """Render message as HTML."""
        header_color = cls.TYPE_COLORS.get(message.message_type, "#2196f3")
        accent_color = header_color

        # Format body with markdown-like conversions
        body = message.html_body or message.body
        body = body.replace("**", "<strong>").replace("*", "<em>")

        # Build metadata section
        metadata_section = ""
        if message.metadata:
            rows = []
            for key, value in message.metadata.items():
                if key not in ["severity"]:  # Skip internal keys
                    formatted_key = key.replace("_", " ").title()
                    rows.append(f"<tr><td>{formatted_key}</td><td>{value}</td></tr>")

            if rows:
                metadata_section = f"""
        <div class="metadata">
            <table>
                {"".join(rows)}
            </table>
        </div>
"""

        return cls.HTML_TEMPLATE.format(
            title=message.title,
            body=body,
            header_color=header_color,
            accent_color=accent_color,
            priority=message.priority.value,
            priority_label=message.priority.value.upper(),
            metadata_section=metadata_section,
            footer=footer_text,
            timestamp=message.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

    @classmethod
    def render_plain(cls, message: NotificationMessage) -> str:
        """Render message as plain text."""
        lines = [
            f"{'=' * 50}",
            f"{message.title}",
            f"{'=' * 50}",
            f"Priority: {message.priority.value.upper()}",
            f"Type: {message.message_type.value.upper()}",
            f"Time: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            message.body,
        ]

        if message.metadata:
            lines.append("")
            lines.append("-" * 30)
            for key, value in message.metadata.items():
                formatted_key = key.replace("_", " ").title()
                lines.append(f"{formatted_key}: {value}")

        lines.append("")
        lines.append("-" * 50)

        return "\n".join(lines)


class EmailNotifier(BaseNotifier):
    """
    Email notification channel.

    Sends notifications via SMTP email.
    """

    def __init__(self, config: EmailConfig | None = None) -> None:
        """Initialize email notifier."""
        self._email_config = config or EmailConfig()
        super().__init__(self._email_config)
        self._connection: smtplib.SMTP | smtplib.SMTP_SSL | None = None
        self._template = EmailTemplate()

        logger.info("EmailNotifier initialized")

    @property
    def channel(self) -> NotificationChannel:
        """Get notification channel type."""
        return NotificationChannel.EMAIL

    async def validate_config(self) -> bool:
        """Validate email configuration."""
        if not self._email_config.smtp_host:
            logger.error("SMTP host not configured")
            return False

        if not self._email_config.smtp_user:
            logger.error("SMTP user not configured")
            return False

        if not self._email_config.from_email:
            logger.error("From email not configured")
            return False

        # Test connection
        try:
            await self._test_connection()
            return True
        except Exception as e:
            logger.error(f"Email configuration validation failed: {e}")
            return False

    async def _test_connection(self) -> None:
        """Test SMTP connection."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._connect_smtp)

    def _connect_smtp(self) -> smtplib.SMTP | smtplib.SMTP_SSL:
        """Connect to SMTP server."""
        if self._email_config.use_ssl:
            context = ssl.create_default_context()
            server = smtplib.SMTP_SSL(
                self._email_config.smtp_host,
                self._email_config.smtp_port,
                context=context,
            )
        else:
            server = smtplib.SMTP(
                self._email_config.smtp_host,
                self._email_config.smtp_port,
            )

            if self._email_config.use_tls:
                context = ssl.create_default_context()
                server.starttls(context=context)

        server.login(
            self._email_config.smtp_user,
            self._email_config.smtp_password,
        )

        return server

    async def _send(self, message: NotificationMessage) -> DeliveryResult:
        """Send email notification."""
        try:
            # Build email message
            email_msg = self._build_email(message)

            # Get recipients
            recipients = message.recipients or self._email_config.default_recipients
            if not recipients:
                return DeliveryResult(
                    success=False,
                    message_id=message.message_id or "",
                    channel=self.channel,
                    error="No recipients specified",
                )

            # Send email
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._send_email,
                email_msg,
                recipients,
            )

            logger.info(f"Email sent successfully to {len(recipients)} recipients")

            return DeliveryResult(
                success=True,
                message_id=message.message_id or "",
                channel=self.channel,
                recipient=", ".join(recipients),
            )

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication error: {e}")
            return DeliveryResult(
                success=False,
                message_id=message.message_id or "",
                channel=self.channel,
                error="Authentication failed",
                retryable=False,
            )

        except smtplib.SMTPRecipientsRefused as e:
            logger.error(f"Recipients refused: {e}")
            return DeliveryResult(
                success=False,
                message_id=message.message_id or "",
                channel=self.channel,
                error="Recipients refused",
                retryable=False,
            )

        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return DeliveryResult(
                success=False,
                message_id=message.message_id or "",
                channel=self.channel,
                error=str(e),
                retryable=True,
            )

        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return DeliveryResult(
                success=False,
                message_id=message.message_id or "",
                channel=self.channel,
                error=str(e),
                retryable=True,
            )

    def _build_email(self, message: NotificationMessage) -> MIMEMultipart:
        """Build email MIME message."""
        email_msg = MIMEMultipart("alternative")

        # Headers
        email_msg["Subject"] = message.title
        email_msg["From"] = f"{self._email_config.from_name} <{self._email_config.from_email}>"

        if self._email_config.reply_to:
            email_msg["Reply-To"] = self._email_config.reply_to

        # Message ID for tracking
        email_msg["X-Message-ID"] = message.message_id or ""

        # Priority headers
        priority_map = {
            NotificationPriority.LOW: ("5", "Non-Urgent"),
            NotificationPriority.NORMAL: ("3", "Normal"),
            NotificationPriority.HIGH: ("2", "Urgent"),
            NotificationPriority.URGENT: ("1", "Urgent"),
            NotificationPriority.CRITICAL: ("1", "Urgent"),
        }
        priority_num, priority_text = priority_map.get(
            message.priority, ("3", "Normal")
        )
        email_msg["X-Priority"] = priority_num
        email_msg["Importance"] = priority_text

        # Plain text body
        if self._email_config.include_plain:
            plain_body = EmailTemplate.render_plain(message)
            plain_part = MIMEText(plain_body, "plain", "utf-8")
            email_msg.attach(plain_part)

        # HTML body
        if self._email_config.include_html:
            html_body = EmailTemplate.render_html(
                message,
                self._email_config.footer_text,
            )
            html_part = MIMEText(html_body, "html", "utf-8")
            email_msg.attach(html_part)

        # Attachments
        for attachment in message.attachments:
            part = MIMEApplication(
                attachment.content,
                Name=attachment.filename,
            )
            part["Content-Disposition"] = f'attachment; filename="{attachment.filename}"'
            email_msg.attach(part)

        return email_msg

    def _send_email(
        self,
        email_msg: MIMEMultipart,
        recipients: list[str],
    ) -> None:
        """Send email via SMTP (blocking)."""
        server = self._connect_smtp()
        try:
            email_msg["To"] = ", ".join(recipients)
            server.sendmail(
                self._email_config.from_email,
                recipients,
                email_msg.as_string(),
            )
        finally:
            server.quit()

    async def send_digest(
        self,
        messages: list[NotificationMessage],
        subject: str = "Daily Trading Digest",
        recipients: list[str] | None = None,
    ) -> DeliveryResult:
        """
        Send a digest of multiple notifications.

        Args:
            messages: Messages to include in digest
            subject: Email subject
            recipients: Optional specific recipients

        Returns:
            Delivery result
        """
        if not messages:
            return DeliveryResult(
                success=False,
                message_id="digest",
                channel=self.channel,
                error="No messages to include in digest",
            )

        # Build digest body
        body_parts = [f"# Trading Digest - {len(messages)} Items\n"]

        for i, msg in enumerate(messages, 1):
            body_parts.append(f"## {i}. {msg.title}")
            body_parts.append(f"*{msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*")
            body_parts.append(msg.body)
            body_parts.append("")

        # Create digest message
        digest = NotificationMessage(
            title=subject,
            body="\n".join(body_parts),
            message_type=NotificationType.INFO,
            priority=NotificationPriority.NORMAL,
            recipients=recipients or self._email_config.default_recipients,
        )

        return await self.send(digest)


def create_email_notifier(config: EmailConfig | None = None) -> EmailNotifier:
    """
    Create an email notifier instance.

    Args:
        config: Email configuration

    Returns:
        EmailNotifier instance
    """
    return EmailNotifier(config)


def create_email_config(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    from_email: str,
    **kwargs: Any,
) -> EmailConfig:
    """
    Create an email configuration.

    Args:
        smtp_host: SMTP server host
        smtp_port: SMTP server port
        smtp_user: SMTP username
        smtp_password: SMTP password
        from_email: Sender email address
        **kwargs: Additional configuration

    Returns:
        EmailConfig instance
    """
    return EmailConfig(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_password=smtp_password,
        from_email=from_email,
        **kwargs,
    )


# Module version
__version__ = "2.2.0"
