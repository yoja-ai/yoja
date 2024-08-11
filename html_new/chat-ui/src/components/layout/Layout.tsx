import React, { useEffect, useState } from "react";
import Sidebar from "../sidebar/Sidebar";
import { ChatLayout } from "./ChatLayout";
import { userData } from "../chat/data";
import { ChatHistory, Message, SourceFile } from "../../type";
import { chatApi } from "../../services/ChatService";
import { Notyf } from 'notyf';
import 'notyf/notyf.min.css';

interface LayoutProps {
  defaultLayout: number[] | undefined;
  defaultCollapsed?: boolean;
}

const Layout = () => {
  const defaultLayout = [320, 480];
  const navCollapsedSize = 8;
  const [isCollapsed, setIsCollapsed] = React.useState(false);
  const [user, setuser] = useState();
  const [isMobile, setIsMobile] = useState(false);
  const [currentChat, setCurrentChat] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([]);

  useEffect(() => {
    const items = JSON.parse(localStorage.getItem('current_chat') || '[]');
    if (items.length > 0) {
      setCurrentChat(items);
    }
    const historyItems = JSON.parse(localStorage.getItem('chat_history') || '[]');
    if (historyItems.length) {
      setChatHistory(historyItems);
    } else {
      const newChat = {
        name: 'New Chat',
        content: [],
        time: new Date(),
        isNew: true // Chat is initially invisible
      };
      setChatHistory([newChat]);
      setCurrentChat(newChat.content); // Focus on the new chat even though it's invisible
    }
  }, []);
  

  const notyf = new Notyf({
    duration: 1000,
    position: {
      x: 'right',
      y: 'top',
    },
    types: [
      {
        className:'notyficss',
        type: 'warning',
        duration: 3000,
      }
    ]
  });

  useEffect(() => {
    const checkScreenWidth = () => {
      setIsMobile(window.innerWidth <= 768);
    };

    // Initial check
    checkScreenWidth();

    // Event listener for screen width changes
    window.addEventListener("resize", checkScreenWidth);

    // Cleanup the event listener on component unmount
    return () => {
      window.removeEventListener("resize", checkScreenWidth);
    };
  }, []);
  
  const sendMessage = (newMessage: Message) => {
    setIsLoading(true);
    const updatedCurrentChat = [...currentChat, newMessage];
    setCurrentChat(updatedCurrentChat);
    localStorage.setItem("current_chat", JSON.stringify(updatedCurrentChat));

    chatApi(updatedCurrentChat).then(async (res: any) => {
        const text = await res.text();
        const result = JSON.parse(text.slice(5));
        if (result) {
            const resMessage = {
                ...result.choices[0].delta,
                source: convertFileNameAndID(result.choices[0].delta.content)
            };
            const fullUpdatedChat = [...updatedCurrentChat, resMessage];

            // Update the current chat in state and localStorage
            setCurrentChat(fullUpdatedChat);
            localStorage.setItem("current_chat", JSON.stringify(fullUpdatedChat));

            // Handling the chat history to ensure no duplicates and correct ordering
            const chatIndex = chatHistory.findIndex(chat => chat.content === currentChat);

            if (chatIndex !== -1) {
                // If the chat was initially marked as new, update and make it visible
                chatHistory[chatIndex] = {
                    ...chatHistory[chatIndex],
                    content: fullUpdatedChat,
                    name: fullUpdatedChat[0]?.content || 'Chat on ' + new Date().toLocaleDateString(),
                    time: new Date(),
                    isNew: false  // Making the chat visible after the first response
                };
            } else {
                // This is genuinely new content, add new chat to the history
                chatHistory.push({
                    name: fullUpdatedChat[0]?.content || 'Chat on ' + new Date().toLocaleDateString(),
                    content: fullUpdatedChat,
                    time: new Date(),
                    isNew: false  // Ensure new chats are visible if not initially marked as new
                });
            }

            // Ensure time is a Date object and sort to ensure the most recently updated chat is at the top
            chatHistory.forEach(chat => chat.time = new Date(chat.time));
            chatHistory.sort((a, b) => b.time.getTime() - a.time.getTime());

            setChatHistory([...chatHistory]);
            localStorage.setItem("chat_history", JSON.stringify(chatHistory));
        }
    }).catch((error: any) => {
        setIsLoading(false);
        notyf.open({
            type: 'warning',
            message: `Error: ${error.message}`
        });
    }).finally(() => {
        setIsLoading(false);
    });
};

  const getFileFullPath = (extension: string, id: string) => {
    switch(extension) {
      case 'doc':
        return `https://docs.google.com/document/d/${id}`
      case 'docx':
        return `https://docs.google.com/document/d/${id}`
      case 'pdf':
        return `https://drive.google.com/file/d/${id}`
      case 'pptx':
        return `https://docs.google.com/presentation/d/${id}`
      case 'ppt':
        return `https://docs.google.com/presentation/d/${id}`
      default:
        return `https://docs.google.com/presentation/d/${id}`
    }
  }

  const convertFileNameAndID = (sourceString: string) => {
    const fileLists = sourceString.split("**Context Source: ");
    const sourceFiles = []; 
    for (let index = 1; index < fileLists.length; index++) {
      const element = fileLists[index];
      const files = element.split("**\t<!-- ID=");
      const name = files[0];
      const extension = files[0].split('.').pop() || 'doc';
      const id = files[1].substring(0, files[1].indexOf('/'));
      const fullPath = getFileFullPath(extension, id);

      const fileInfo: SourceFile = {
        name: name,
        id: id,
        extension: extension,
        fullPath: fullPath
      };
      sourceFiles.push(fileInfo);
    }
    return sourceFiles;
  }
  
  return (
    <div className="app-layout">
      <Sidebar
        isCollapsed={isCollapsed}
        isMobile={isMobile}
        currentChat={currentChat}
        setCurrentChat={setCurrentChat}
        chatHistory = {chatHistory}
        setChatHistory ={setChatHistory}
      />
      <ChatLayout
        currentChat={currentChat}
        setCurrentChat={setCurrentChat}
        userInfo={user}
        setIsCollapsed={setIsCollapsed}
        isCollapsed={isCollapsed}
        isMobile={isMobile}
        isLoading={isLoading}
        sendMessage={sendMessage}
      />
    </div>
  );
}

export default Layout;
