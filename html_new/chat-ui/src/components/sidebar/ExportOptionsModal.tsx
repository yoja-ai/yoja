import React, { useState } from 'react';
import { Modal, Button, Input } from 'antd';
import { MailOutlined, DownloadOutlined, CheckOutlined, CloseOutlined, LoadingOutlined } from '@ant-design/icons';
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
    const [buttonState, setButtonState] = useState<'default' | 'loading' | 'success' | 'error'>('default');

    const handleEmailTranscript = async () => {
        const pdfBlob = generatePDF(chat);
        console.log("Generated PDF Blob:", pdfBlob); // Debugging output
        console.log("Recipient Email:", email); // Debugging output

        setButtonState('loading'); // Set the button to loading state

        // Convert Blob to Base64
        const reader = new FileReader();
        reader.readAsDataURL(pdfBlob);

        reader.onloadend = async () => {
            const base64data = reader.result?.toString().split(",")[1]; // Strip out the data URL prefix

            if (base64data) {
                console.log("Base64 PDF Data:", base64data); // Debugging output
                try {
                    await sendEmail({ recipient: email, pdfData: base64data }); // Use the sendEmail service
                    setTimeout(() => {
                        setButtonState('success'); // Set to success state
                        setTimeout(() => setButtonState('default'), 2000); // Revert to default after 2 seconds
                    }, 2000);
                } catch (error) {
                    console.error("Error sending email:", error);
                    setTimeout(() => {
                        setButtonState('error'); // Set to error state
                        setTimeout(() => setButtonState('default'), 2000); // Revert to default after 2 seconds
                    }, 2000);
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

    const renderButtonContent = () => {
        switch (buttonState) {
            case 'loading':
                return { text: 'Sending Email', icon: <LoadingOutlined spin /> };
            case 'success':
                return { text: 'Email Sent', icon: <CheckOutlined />, style: { backgroundColor: '#52c41a', borderColor: '#52c41a' } };
            case 'error':
                return { text: 'Failed to send', icon: <CloseOutlined />, type: 'primary', danger: true };
            default:
                return { text: 'Email Transcript', icon: <MailOutlined /> };
        }
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
                <Button
                    type={buttonState === 'error' ? 'primary' : 'primary'}
                    danger={buttonState === 'error'}
                    style={buttonState === 'success' ? { backgroundColor: '#52c41a', borderColor: '#52c41a' } : {}}
                    icon={renderButtonContent().icon}
                    onClick={handleEmailTranscript}
                >
                    {renderButtonContent().text}
                </Button>
                <Button icon={<DownloadOutlined />} onClick={handleDownloadTranscript}>
                    Download Transcript
                </Button>
            </div>
        </Modal>
    );
};

export default ExportOptionsModal;
