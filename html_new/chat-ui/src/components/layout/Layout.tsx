import React, { useEffect, useState } from "react";
import Sidebar from "../sidebar/Sidebar";
import { ChatLayout } from "./ChatLayout";
import { userData } from "../chat/data";
import { ChatHistory, Message, SourceFile } from "../../type";
import { chatApi, searchSubdirApi } from "../../services/ChatService";

interface LayoutProps {
  defaultLayout?: number[];
  defaultCollapsed?: boolean;
}

const Layout: React.FC<LayoutProps> = () => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [currentChat, setCurrentChat] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([]);
  const [searchSubdir, setSearchSubdir] = useState("");

  useEffect(() => {
    var cookieValue = document.cookie.split('; ').filter(row => row.startsWith('__Host-yoja-searchsubdir=')).map(c=>c.split('=')[1])[0];
    if (cookieValue != undefined) {
      setSearchSubdir(cookieValue);
    }
  }, []);
  
  useEffect(() => {
    const items = JSON.parse(localStorage.getItem('current_chat') || '[]');
    if (items.length > 0) {
      setCurrentChat(items);
    }
    
    const historyItems = JSON.parse(localStorage.getItem('chat_history') || '[]');
    if (historyItems.length) {
      // First, set the chat history as it is
      setChatHistory(historyItems);
      
      // Then, process the history to set the first chat as selected
      const updatedHistory = historyItems.map((chat: ChatHistory, index: number) => ({
        ...chat,
        selected: index === 0 // Only the top (first) chat is selected
      }));
      
      setChatHistory(updatedHistory);
      setCurrentChat(updatedHistory[0].content); // Set current chat to the top chat's content
    } else {
      const newChat: ChatHistory = {
        name: 'New Chat',
        content: [],
        time: new Date(),
        isNew: true,
        selected: true // Initialize with true since it's the only chat
      };
      setChatHistory([newChat]);
      setCurrentChat(newChat.content);
    }
  }, []);
  

  useEffect(() => {
    const checkScreenWidth = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    window.addEventListener("resize", checkScreenWidth);
    return () => {
      window.removeEventListener("resize", checkScreenWidth);
    };
  }, []);

  const handleChatSelect = (selectedChat: ChatHistory) => {
    const updatedChatHistory = chatHistory.map(chat => ({
      ...chat,
      selected: chat === selectedChat
    }));
    setChatHistory(updatedChatHistory);
    setCurrentChat(selectedChat.content);
  };

  const convertFileNameAndID = (sourceString: string) => {
    // Split the source string into parts based on a unique marker that separates file entries
    const fileLists = sourceString.split("**Context Source: ");
    const sourceFiles = [];
    
    // Iterate over each file entry extracted from the source string
    for (let index = 1; index < fileLists.length; index++) {
        const element = fileLists[index];
        // Split the entry to separate the file name from the ID, assuming ID is commented at the end
        const files = element.split("**\t<!-- ID=");
        const name = files[0].trim();
        const extension = files[0].split('.').pop() || 'doc';  // Extract extension or default to 'doc'
        const id = files[1].substring(0, files[1].indexOf('-->'));  // Extract ID up to closing comment

        // Construct the full URL for accessing the document based on its type
        const fullPath = getFileFullPath(extension, id);

        // Create an object representing the file and add it to the result list
        const fileInfo = {
            name: name,
            id: id,
            extension: extension,
            fullPath: fullPath
        };
        sourceFiles.push(fileInfo);
    }

    return sourceFiles;
}

const getFileFullPath = (extension: string, id: string) => {
    switch(extension) {
        case 'doc':
        case 'docx':
            return `https://docs.google.com/document/d/${id}`;
        case 'pdf':
            return `https://drive.google.com/file/d/${id}`;
        case 'ppt':
        case 'pptx':
            return `https://docs.google.com/presentation/d/${id}`;
        default:
            return `https://docs.google.com/presentation/d/${id}`; // Default case if no extension matches
    }
}

const sendSearchSubdir = (newSearchSubdir: string) => {
  setIsLoading(true);
  searchSubdirApi(newSearchSubdir).then(async (res: any) => {
    const text = await res.text();
    const result = JSON.parse(text.slice(5));
  }).catch((error: any) => {
      setIsLoading(false);
      alert(`Error: ${error.message}`);
  }).finally(() => {
      setIsLoading(false);
  });
};

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
          if (result.choices[0].hasOwnProperty("sample_source")) {
            resMessage.sample_source = result.choices[0].sample_source;
          }
          if (result.choices[0].hasOwnProperty("searchsubdir")) {
            resMessage.searchsubdir = result.choices[0].searchsubdir;
          }
          const fullUpdatedChat = [...updatedCurrentChat, resMessage];

          // Update the current chat in state and localStorage
          setCurrentChat(fullUpdatedChat);
          localStorage.setItem("current_chat", JSON.stringify(fullUpdatedChat));

          // Handling the chat history to ensure no duplicates and correct ordering
          let chatIndex = chatHistory.findIndex(chat => chat.content === currentChat);

          if (chatIndex !== -1) {
              // If the chat was initially marked as new, update and make it visible
              chatHistory[chatIndex] = {
                  ...chatHistory[chatIndex],
                  content: fullUpdatedChat,
                  name: fullUpdatedChat[0]?.content || 'Chat on ' + new Date().toLocaleDateString(),
                  time: new Date(),
                  isNew: false,  // Making the chat visible after the first response
                  selected: true  // Ensure this chat is selected
              };
          } else {
              // This is genuinely new content, add new chat to the history
              chatHistory.forEach(chat => chat.selected = false); // Unselect other chats
              chatHistory.push({
                  name: fullUpdatedChat[0]?.content || 'Chat on ' + new Date().toLocaleDateString(),
                  content: fullUpdatedChat,
                  time: new Date(),
                  isNew: false,  // Ensure new chats are visible if not initially marked as new
                  selected: true  // Select this new chat
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
      alert(`Error: ${error.message}`);
  }).finally(() => {
      setIsLoading(false);
  });
};


  return (
    <div className="app-layout">
      <Sidebar
        isCollapsed={isCollapsed}
        isMobile={isMobile}
        currentChat={currentChat}
        setCurrentChat={setCurrentChat}
        chatHistory={chatHistory}
        setChatHistory={setChatHistory}
        handleChatSelect={handleChatSelect}
      />
      <ChatLayout
        currentChat={currentChat}
        setCurrentChat={setCurrentChat}
        userInfo={userData}
        setIsCollapsed={setIsCollapsed}
        isCollapsed={isCollapsed}
        isMobile={isMobile}
        isLoading={isLoading}
        sendMessage={sendMessage}
        searchSubdir={searchSubdir}
        sendSearchSubdir={sendSearchSubdir}
      />
    </div>
  );
}

export default Layout;
