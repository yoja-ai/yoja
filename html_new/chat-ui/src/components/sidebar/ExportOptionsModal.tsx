import React, { useState } from 'react';
import { Modal, Button, Input } from 'antd';
import { MailOutlined, DownloadOutlined } from '@ant-design/icons';
import jsPDF from 'jspdf';
import { ChatHistory, ExportOptionsModalProps } from '../../type'; // Ensure this path is correct
import { sendEmail } from "../../services/EmailService"; // Import the sendEmail function

const generatePDF = (chatHistory: ChatHistory): Blob => {
    const doc = new jsPDF();
    doc.setFont('Helvetica');
    doc.setFontSize(10);
    let y = 60;

    chatHistory.content.forEach(msg => {
        const maxWidth = 180;
        const messageContent = doc.splitTextToSize(msg.content, maxWidth);
        const lineHeight = 10;
        const boxHeight = messageContent.length * lineHeight + 10;

        const xPosition = msg.role === 'user' ? doc.internal.pageSize.width - maxWidth - 10 : 10;
        doc.setDrawColor(0);
        doc.setFillColor(msg.role === 'user' ? 232 : 255, msg.role === 'user' ? 240 : 255, msg.role === 'user' ? 254 : 255);
        doc.roundedRect(xPosition, y, maxWidth, boxHeight, 3, 3, 'FD');
        doc.text(messageContent, xPosition + 5, y + lineHeight);

        y += boxHeight + 10;
        if (y > doc.internal.pageSize.height - 20) {
            doc.addPage();
            y = 20;
        }
    });

    return doc.output('blob'); // Return the PDF as a blob
};

const ExportOptionsModal: React.FC<ExportOptionsModalProps> = ({ chat, isVisible, onClose }) => {
    const [email, setEmail] = useState('');

    const handleEmailTranscript = async () => {
        const pdfBlob = generatePDF(chat);
        console.log("Generated PDF Blob:", pdfBlob); // Debugging output
        console.log("Recipient Email:", email); // Debugging output

        // Convert Blob to Base64
        const reader = new FileReader();
        reader.readAsDataURL(pdfBlob);

        reader.onloadend = async () => {
            const base64data = reader.result?.toString().split(",")[1]; // Strip out the data URL prefix

            if (base64data) {
                console.log("Base64 PDF Data:", base64data); // Debugging output
                try {
                    await sendEmail({ recipient: email, pdfData: base64data }); // Use the sendEmail service
                    alert('Email sent successfully!');
                } catch (error) {
                    console.error("Error sending email:", error);
                    alert('Failed to send email. Please try again.');
                }
            } else {
                console.error("Failed to convert PDF to Base64.");
                alert('Failed to prepare the PDF for email. Please try again.');
            }
        };
    };

    const handleDownloadTranscript = () => {
        const pdfBlob = generatePDF(chat);
        const url = window.URL.createObjectURL(pdfBlob);
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'chat-transcript.pdf');
        document.body.appendChild(link);
        link.click();
    };

    return (
        <Modal
            title="Export Chat Transcript"
            visible={isVisible}
            onCancel={onClose}
            footer={[<Button key="back" onClick={onClose}>Cancel</Button>]}
        >
            <p>Enter the email address to receive the chat transcript:</p>
            <Input
                placeholder="Email address"
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                style={{ marginBottom: '20px' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-evenly', margin: '20px 0' }}>
                <Button icon={<MailOutlined />} type="primary" onClick={handleEmailTranscript}>
                    Email Transcript
                </Button>
                <Button icon={<DownloadOutlined />} onClick={handleDownloadTranscript}>
                    Download Transcript
                </Button>
            </div>
        </Modal>
    );
};

export default ExportOptionsModal;
