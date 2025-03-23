import smtplib
from email.message import EmailMessage
import logging

def send_email(smtp_server, port, sender_email, sender_password, recipient_email, subject, body):
    """Send an email using the provided SMTP settings"""
    # Create the email message
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg.set_content(body)
    
    try:
        # Connect securely to the SMTP server using SSL and send the email
        with smtplib.SMTP_SSL(smtp_server, port) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        print(f"Email sent successfully to {recipient_email}")
        return True
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False
