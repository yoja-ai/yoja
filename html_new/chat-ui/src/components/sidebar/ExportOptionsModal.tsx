import React from 'react';
import { Modal, Button } from 'antd';
import { MailOutlined, DownloadOutlined } from '@ant-design/icons';
import { ChatHistory, ExportOptionsModalProps, Message } from '../../type'; // Ensure this path is correct
import jsPDF from 'jspdf';

const generatePDF = (chatHistory: ChatHistory) => {
    const doc = new jsPDF();

    // Optional: Load a custom font
    // Example: doc.addFont('path/to/your/font.ttf', 'fontName', 'normal');
    doc.setFont('Helvetica'); // Helvetica is generally good for modern looks
    doc.setFontSize(10); // Adjust font size to your liking

    const imageURL = 'icon.png'; // Path to your image
    // Load the image
    doc.addImage(imageURL, 'PNG', 10, 10, 50, 20); // Adjust size as needed

    let y = 60; // Start position for the first text, considering image height

    chatHistory.content.forEach((msg: Message) => {
        const maxWidth = 180; // Max width for the text
        const messageContent = doc.splitTextToSize(msg.content, maxWidth); // Handles long text
        const lineHeight = 10; // Line height for each line of text
        const boxHeight = messageContent.length * lineHeight + 10; // Calculate box height

        if (msg.role === 'user') {
            // Right-aligned text for the user
            const xPosition = doc.internal.pageSize.width - maxWidth - 10; // Right margin
            doc.setDrawColor(0); // Set border color
            doc.setFillColor(232, 240, 254); // Light blue bubble
            doc.roundedRect(xPosition, y, maxWidth, boxHeight, 3, 3, 'FD'); // Filled and bordered
            doc.text(messageContent, xPosition + 5, y + lineHeight, { maxWidth: maxWidth - 10 });
        } else {
            // Left-aligned text for the assistant
            doc.setDrawColor(0); // Set border color
            doc.setFillColor(255, 255, 255); // White bubble
            doc.roundedRect(10, y, maxWidth, boxHeight, 3, 3, 'FD'); // Filled and bordered
            doc.text(messageContent, 15, y + lineHeight, { maxWidth: maxWidth - 10 });
        }

        y += boxHeight + 10; // Add space between messages

        if (y > doc.internal.pageSize.height - 20) {
            doc.addPage();
            y = 20; // Margin top for new page
        }
    });

    doc.save('chat-transcript.pdf');
};

const ExportOptionsModal: React.FC<ExportOptionsModalProps> = ({ chat, isVisible, onClose }) => {
  return (
    <Modal
      title="Export Chat Transcript"
      visible={isVisible}
      onCancel={onClose}
      footer={[
        <Button key="back" onClick={onClose}>Cancel</Button>,
      ]}
    >
      <p>How would you like to receive the chat transcript?</p>
      <div style={{ display: 'flex', justifyContent: 'space-evenly', margin: '20px 0' }}>
        <Button icon={<MailOutlined />} type="primary" onClick={() => console.log('Emailing transcript...')}>Email Transcript</Button>,
        <Button icon={<DownloadOutlined />} onClick={() => generatePDF(chat)}>Download Transcript</Button>
      </div>
    </Modal>
  );
};

export default ExportOptionsModal;
