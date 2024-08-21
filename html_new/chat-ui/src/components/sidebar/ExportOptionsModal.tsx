import React, { useState } from 'react';
import { Modal, Button, Input } from 'antd';
import { MailOutlined, DownloadOutlined, CheckOutlined, CloseOutlined, LoadingOutlined } from '@ant-design/icons';
import { ChatHistory, ExportOptionsModalProps } from '../../type'; 
import { sendEmail } from "../../services/EmailService"; // Import the sendEmail function
import { generatePdfFromChatHistory } from '../../services/GeneratePdfService';

interface PdfGenerationRequest {
    chatHistory: ChatHistory;  // This should match the structure you are currently using
}

async function generatePDF(request: PdfGenerationRequest): Promise<Blob> {
    try {
        const pdfBlob = await generatePdfFromChatHistory(request);
        return pdfBlob;
    } catch (error) {
        console.error('Error generating PDF:', error);
        throw new Error('Failed to generate PDF');
    }
}

const ExportOptionsModal: React.FC<ExportOptionsModalProps> = ({ chat, isVisible, onClose }) => {
    const [email, setEmail] = useState('');
    const [buttonState, setButtonState] = useState<'default' | 'loading' | 'success' | 'error'>('default');

    const handleEmailTranscript = async () => {
        try {
            setButtonState('loading');
            const pdfBlob = await generatePDF({ chatHistory: chat });
            const reader = new FileReader();
            reader.onerror = () => {
                console.error("Error reading PDF Blob.");
                setButtonState('error');
                setTimeout(() => setButtonState('default'), 2000);
            };
            reader.readAsDataURL(pdfBlob);
            reader.onloadend = async () => {
                const base64data = reader.result?.toString().split(",")[1];
                if (base64data) {
                    await sendEmail({ recipient: email, pdfData: base64data });
                    setButtonState('success');
                    setTimeout(() => setButtonState('default'), 2000);
                } else {
                    console.error("Failed to convert PDF to Base64.");
                    setButtonState('error');
                    setTimeout(() => setButtonState('default'), 2000);
                }
            };
        } catch (error) {
            console.error("Error in generating or sending email:", error);
            setButtonState('error');
            setTimeout(() => setButtonState('default'), 2000);
        }
    };

    const handleDownloadTranscript = async () => {
        try {
            const pdfBlob = await generatePDF({ chatHistory: chat });
            const url = window.URL.createObjectURL(pdfBlob);
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', 'chat-transcript.pdf');
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (error) {
            console.error("Error downloading transcript:", error);
            alert('Failed to download the PDF. Please try again.');
        }
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
                    type={buttonState === 'error' ? 'primary' : 'default'}
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
