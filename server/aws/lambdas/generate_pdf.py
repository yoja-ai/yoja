import re
import base64  # Make sure to import the base64 module
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from io import BytesIO
from utils import respond  # Assuming you have a respond utility for Lambda responses

# Custom prettify function to format text with respect to numbered lists
def prettify_text(text):
    # Ensure numbered lists are placed on different lines
    text = re.sub(r'(\d+\.\s)', r'\n\1', text)
    text = re.sub(r'(Sources\s+)', r'\n\1', text)  # Ensure "Sources" is on a new line
    return text.strip()

def generate_pdf(event, context):
    # Assuming event['body'] is the JSON-encoded ChatHistory object
    body = json.loads(event['body'])
    chat_log = body.get('chatHistory', {}).get('content', [])

    # Initialize a bytes buffer for the PDF
    buffer = BytesIO()

    # Create PDF document in memory
    pdf = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    # Get default styles and define custom styles
    styles = getSampleStyleSheet()

    # Enhanced title style
    style_title = ParagraphStyle(
        name="TitleStyle",
        fontSize=22,  # Larger font size
        fontName="Helvetica-Bold",  # Bold font
        spaceAfter=20,
        alignment=1,  # Center alignment
        textColor=colors.HexColor("#333333"),  # Darker, professional color
    )

    style_chatbox_user = ParagraphStyle(
        name="ChatboxUser",
        fontSize=12,
        fontName="Helvetica",
        spaceAfter=10,  # Reduced space after each message
        leading=14,
        textColor='black',
        backColor=colors.HexColor("#f0f0f0"),
        borderColor=colors.HexColor("#dddddd"),
        borderWidth=1.5,
        borderPadding=(10, 10, 10, 10),
        leftIndent=100,  # Widen user messages
        rightIndent=20,
        borderRadius=10
    )

    style_chatbox_assistant = ParagraphStyle(
        name="ChatboxAssistant",
        fontSize=12,
        fontName="Helvetica",
        spaceAfter=10,  # Reduced space after each message
        leading=14,
        textColor='black',
        backColor=colors.white,  # White background for assistant messages
        borderColor=colors.HexColor("#333333"),  # Dark border for assistant messages
        borderWidth=1.5,
        borderPadding=(10, 10, 10, 10),
        leftIndent=20,  # Widen assistant messages
        rightIndent=100,
        borderRadius=10
    )

    # Add title with the enhanced style
    title = Paragraph("Your chat with Yoja", style_title)
    elements.append(title)

    # Add spacing
    elements.append(Spacer(1, 24))

    # Add chat messages with appropriate alignment and prettified text
    for i, message in enumerate(chat_log):
        speaker = message['role'].capitalize()
        content = prettify_text(message['content'])
        para = Paragraph(content.replace('\n', '<br />'), style_chatbox_user if speaker == "User" else style_chatbox_assistant)
        elements.append(para)
        
        # Add extra space between different speakers
        if i < len(chat_log) - 1:
            elements.append(Spacer(1, 28))  # Increase the space between messages

    # Add closing spacing
    elements.append(Spacer(1, 28))

    # Build PDF in memory
    pdf.build(elements)
    
    # Move to the beginning of the BytesIO buffer
    buffer.seek(0)

    # Return the PDF as a base64-encoded string
    pdf_data = buffer.getvalue()
    buffer.close()

    pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')

    return respond(None, res={'pdf_data': pdf_base64})
