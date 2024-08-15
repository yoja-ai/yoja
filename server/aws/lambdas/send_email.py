import json
import boto3
from botocore.exceptions import ClientError
import base64
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Initialize the SES client
ses_client = boto3.client('ses', region_name='us-east-1')

def send_email(event, context):
    print("send_email function triggered")
    
    try:
        # Log the entire event for debugging purposes
        print(f"Received event: {json.dumps(event, indent=2)}")
        
        # Extract the data from the event
        body = json.loads(event['body'])
        recipient = body.get('recipient')
        pdf_data = body.get('pdfData')

        if not recipient or not pdf_data:
            raise ValueError('Missing recipient or PDF data in the request')

        # Create a MIME email message
        msg = MIMEMultipart()
        msg['From'] = 'service@yoja.ai'
        msg['To'] = recipient
        msg['Subject'] = "Chat Transcript"
        
        # Attach the email body
        body_html = "PFA chat transcript."
        msg.attach(MIMEText(body_html, 'html'))

        # Attach the PDF
        pdf_attachment = MIMEBase('application', 'octet-stream')
        pdf_attachment.set_payload(base64.b64decode(pdf_data))
        encoders.encode_base64(pdf_attachment)
        pdf_attachment.add_header('Content-Disposition', 'attachment', filename="chat-transcript.pdf")
        msg.attach(pdf_attachment)

        # Send the email using SES
        response = ses_client.send_raw_email(
            Source=msg['From'],
            Destinations=[msg['To']],
            RawMessage={'Data': msg.as_string()}
        )
        print("Email sent successfully")
        print(f"SES Response: {response}")

        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Email sent successfully'})
        }
    except ClientError as e:
        print(f"SES client error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Failed to send email'})
        }
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Failed due to an unexpected error'})
        }
