export interface Message {
    id?: number;
    role: string;
    content: string;
    source?: SourceFile[];
    like?: boolean;
    dislike?: boolean;
    copied?: boolean;
}

export interface ChatHistory {
    name: string;
    content: Message[];
    time: Date;
    isNew?: boolean; // Add this line to indicate if the chat is new
}


export interface ExportOptionsModalProps {
    chat: ChatHistory; // Adjust according to the actual properties of a chat
    isVisible: boolean;
    onClose: () => void; // Define the type as a function that returns nothing
}

export interface SourceFile {
    id: string;
    name: string;
    extension: string;
    fullPath: string;
}

export interface User {
    id: number;
    avatar: string;
    messages: Message[];
    name: string;
}
export type UserInfo = any;

export const loggedInUserData = {
    id: 5,
    avatar: '/LoggedInUser.jpg',
    name: 'Jakob Hoeg',
};

export type LoggedInUserData = (typeof loggedInUserData);
