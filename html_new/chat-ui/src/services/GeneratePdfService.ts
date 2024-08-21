import { ChatHistory } from '../type';
const servicesConfig = (window as any).ServiceConfig;

interface PdfGenerationRequest {
    chatHistory: ChatHistory; // Assuming ChatHistory is an interface defined elsewhere
}

export const generatePdfFromChatHistory = async (request: PdfGenerationRequest): Promise<Blob> => {
    const API_URL = "/entrypoint/generate-pdf";  // Update this path to your PDF generation endpoint
    const requestUrl = servicesConfig.envAPIEndpoint + API_URL;

    const requestBody = JSON.stringify({
        chatHistory: request.chatHistory
    });

    try {
        const response = await fetch(requestUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: requestBody,
        });

        if (!response.ok) {
            throw new Error(`Failed to generate PDF: ${response.status} ${response.statusText}`);
        }

        const responseData = await response.json();

        // Assuming the PDF is returned as a base64-encoded string in 'pdf_data'
        const pdfBase64 = responseData.pdf_data;

        // Decode base64 string to binary data
        const byteCharacters = atob(pdfBase64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);

        // Create a Blob from the binary data
        const pdfBlob = new Blob([byteArray], { type: 'application/pdf' });
        return pdfBlob;

    } catch (error) {
        console.error("Error generating PDF:", error);
        throw error; // Rethrow the error to be handled by the caller
    }
};
