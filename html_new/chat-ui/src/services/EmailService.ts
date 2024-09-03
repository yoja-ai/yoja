const servicesConfig = (window as any).ServiceConfig;

interface EmailContent {
    recipient: string; // Recipient's email address
    pdfData: string;   // Base64-encoded PDF file
}

export const sendEmail = async (emailContent: EmailContent): Promise<Response> => {
    const API_URL = "/v1/send-email";  // Update this path to your email sending endpoint
    const requestUrl = servicesConfig.envAPIEndpoint + API_URL;
    console.log(requestUrl);

    // Log the email content for debugging
    console.log("Recipient Email:", emailContent.recipient);
    console.log("PDF Data (Base64):", emailContent.pdfData);

    const requestBody = JSON.stringify({
        recipient: emailContent.recipient,
        pdfData: emailContent.pdfData // Using pdfData directly from the interface
    });

    try {
        const response = await fetch(requestUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: requestBody, // Send JSON data
        });

        if (!response.ok) {
            throw new Error(`Failed to send email: ${response.status} ${response.statusText}`);
        }
        return response;  // Return the response for successful requests
    } catch (error) {
        console.error("Error sending email:", error);
        throw error;  // Rethrow the error to be handled by the caller
    }
};