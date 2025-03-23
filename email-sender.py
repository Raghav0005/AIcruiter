import smtplib
from email.message import EmailMessage

def send_email(smtp_server, port, sender_email, sender_password, recipient_email, subject, body):
    # Create the email message
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg.set_content(body)
    
    # Connect securely to the SMTP server using SSL and send the email
    with smtplib.SMTP_SSL(smtp_server, port) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)
    print("Email sent successfully!")

if __name__ == '__main__':
    # Test settings (update these values with your own credentials and recipient)
    smtp_server = 'smtp.gmail.com'
    port = 465  # SSL port for Gmail
    sender_email = 'vasuraghav04@gmail.com'
    sender_password = 'utgn zjdp tyhs udnc'  # Use an app password if required
    recipient_email = 'lakyvasu08@gmail.com'
    subject = 'Test Email from Python'
    body = 'Hello,\n\nThis is a test email sent using Python.\n\nBest regards,\nPython Script'

    # Send the test email
    send_email(smtp_server, port, sender_email, sender_password, recipient_email, subject, body)
