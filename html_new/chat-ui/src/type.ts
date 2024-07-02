export interface Message {
    id?: number;
    role: string;
    content: string;
    source?: SourceFile[];
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
