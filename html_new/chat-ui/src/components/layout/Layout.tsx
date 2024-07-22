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
    if (items) {
      setCurrentChat(items);
    }
    const historyItems = JSON.parse(localStorage.getItem('chat_history') || '[]');
    if (historyItems.length) {
      console.log('historyItemsss', historyItems);
      setChatHistory(historyItems);
    } else {
      const newChat: ChatHistory = {
        name: 'New Chat',
        content: [],
        time: new Date()
      };
      setChatHistory([newChat]);
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
    setCurrentChat([...currentChat, newMessage]);
    const newState = [...currentChat, newMessage];
    localStorage.setItem("current_chat", JSON.stringify(newState));
    chatApi([...currentChat, newMessage]).then(async (res: any) => {
      const text = await res.text();
      const result = JSON.parse(text.slice(5)); 
      if(result) {
        const resMessage: Message = result.choices[0].delta;
        resMessage.source = convertFileNameAndID(resMessage.content);
        resMessage.id = currentChat?.length + 1;
        setIsLoading(false);
        setCurrentChat([...newState, resMessage]);
        localStorage.setItem("current_chat", JSON.stringify([...newState, resMessage]));
        
        const history = [...newState, resMessage];
        const chatHistoryStorage: any[] = JSON.parse(localStorage.getItem('chat_history') || '[]');
        const newChat = {
          name: history[0].content,
          content: history,
          time: new Date()
        };
        const ff = chatHistoryStorage.filter( chat => chat.name === newChat.name);
        const updatedChatHistory = chatHistoryStorage.map((chat=> {
          if(chat.name === newChat.name) {
            chat.content = history;
            chat.time = new Date();
          }
          return chat;
        }));

        if(updatedChatHistory.length === 0) {
          updatedChatHistory[0] = newChat;
        } else if(currentChat.length <= 1) {
          const last = (updatedChatHistory.length - 1);
          updatedChatHistory[last] = newChat;
        }
        setChatHistory(updatedChatHistory);
        localStorage.setItem("chat_history", JSON.stringify(updatedChatHistory));
      }
    }).catch((error: any) => {
      setIsLoading(false);
      notyf.open({
        type: 'warning',
        message: `<b> Error : </b> ${ error.message }`
      });
    }).finally(()=> {
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