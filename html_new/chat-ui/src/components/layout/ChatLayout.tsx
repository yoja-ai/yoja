import { Message, SourceFile, UserInfo } from "../../type";
import React, { useEffect } from "react";
import { Info, Phone, Video, Menu, Trash2 } from 'lucide-react';
import { Chat } from "../chat/Chat";
import { chatApi } from "../../services/ChatService";
import { Notyf } from 'notyf';
import 'notyf/notyf.min.css';

interface ChatProps {
  messages?: Message[];
  setMessages: any;
  userInfo: UserInfo;
  isMobile: boolean;
  isCollapsed: boolean;
  setIsCollapsed: any;
}

export function ChatLayout({ messages, userInfo, isMobile,  setIsCollapsed, isCollapsed}: ChatProps) {
  const [isLoading, setIsLoading] = React.useState(false);
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
  const [messagesState, setMessages] = React.useState<Message[]>(
    messages ?? []
  );

  useEffect(() => {
    if(messages) {
      setMessages(messages);
    }
  }, [messages]);
  
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

  const sendMessage = (newMessage: Message) => {
    setIsLoading(true);
    setMessages([...messagesState, newMessage]);
    const newState = [...messagesState, newMessage];
    localStorage.setItem("current_chat", JSON.stringify(newState));
    chatApi([...messagesState, newMessage]).then(async (res: any) => {
      const text = await res.text();
      const result = JSON.parse(text.slice(5)); 
      if(result) {
        const resMessage: Message = result.choices[0].delta;
        resMessage.source = convertFileNameAndID(resMessage.content);
        setIsLoading(false);
        setMessages([...newState, resMessage]);
        localStorage.setItem("current_chat", JSON.stringify([...newState, resMessage]));
      }
    }).catch((e: any) => {
      console.log('Error', e);
      notyf.open({
        type: 'warning',
        message: '<b> Error : </b> Something went wrong. Try again..'
      });
    }).finally(()=> {
      setIsLoading(false);
    });
  };

  const clearChat = () => {
    setMessages([]);
    localStorage.setItem("current_chat", JSON.stringify([]));
  }

  return (
    <div style={{flex: '80 1 0px'}} >
      <div className="chat">
        <div className="chat-header-bar"> 
          <div className="chat-header">
            <Menu size={20} style={{cursor: 'pointer'}} onClick={()=> {setIsCollapsed(!isCollapsed)}}/> 
            <span className="chat-header-text"> Yoja.ai </span>
          </div>
          <button className='clear-chat-btn' onClick={ clearChat }> 
            <Trash2 size={16} /> <span className="clear-chat-btn-text"> Clear chat </span>
          </button>
        </div>
        <Chat userInfo={undefined} messages={messagesState} sendMessage={sendMessage} isLoading={isLoading} isMobile={false} />
      </div>
    </div>
  );
}
