export interface Message {
    id?: number;
    role: string;
    content: string;
    source?: SourceFile[];
    like?: boolean;
    dislike?: boolean;
    copied?: boolean;
    tokens?: {   
        prompt: number;
        completion: number;
    };
    sample_source?:string[];
    searchsubdir?:string;
}

export interface ChatHistory {
    name: string;
    content: Message[];
    time: Date;
    isNew?: boolean;
    selected: boolean; 
}



export interface ExportOptionsModalProps {
    chat: ChatHistory; 
    isVisible: boolean;
    onClose: () => void; 
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

