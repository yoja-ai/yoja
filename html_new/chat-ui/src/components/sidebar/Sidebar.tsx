import React, { useEffect, useState } from 'react';
import { FolderPlus, SquarePen } from "lucide-react";
import SideBarSearch from './SideBarSearch';
import { Avatar, AvatarImage } from '../ui/avatar';
import { Message } from '../../type';
import PopupMenu from './popup';
import UserMenu from './UserMenu';

interface SidebarProps {
  isCollapsed: boolean;
  onClick?: () => void;
  isMobile: boolean;
  messages?: Message[];
  setMessages: any;
}

const Sidebar = ({ isCollapsed, isMobile, setMessages }: SidebarProps) => {
  const [chatHistory, setChatHistory] = useState<Message[]>([]);
  const [filteredChatHistory, setFilteredChatHistory] = useState<Message[]>([]);
  const [searchTerm, setSearchTerm] = useState<string>('');

  useEffect(() => {
    const historyItems = JSON.parse(localStorage.getItem('chat_history') || '[]');
    const filteredHistory = historyItems.filter((item: any) => {
      return (item.name).toLowerCase().includes(searchTerm.toLowerCase().trim());
    });
    setFilteredChatHistory(filteredHistory);
  }, [chatHistory, searchTerm]);
  
  useEffect(() => {
    const historyItems = JSON.parse(localStorage.getItem('chat_history') || '[]');
    setChatHistory(historyItems);
  }, []);

  const newChat = () => {
    setMessages([]);
    const currentChat: Message[]  = JSON.parse(localStorage.getItem('current_chat') || '[]');
    const chatHistoryStorage: any[] = JSON.parse(localStorage.getItem('chat_history') || '[]');

    if(currentChat.length) {
      const newChat = {
        name: currentChat[0]?.content,
        content: currentChat,
        time: new Date()
      };
      chatHistoryStorage.push(newChat)
    }
    setChatHistory(chatHistoryStorage);
    localStorage.setItem("chat_history", JSON.stringify(chatHistoryStorage));
    localStorage.setItem("current_chat", JSON.stringify([]));
  }

  return (
    <div data-collapsed={isCollapsed} style={isCollapsed ? {flex: '0 1 0px'} : {flex: '20 1 0px'}}>
      <div className="sidebar">
        {!isCollapsed && (
          <div className="sidebar-head">
            <div className="sidebar-header">
              <div className="flex">
                <img style={{width:'100%', height: '24px'}} src="Yoja.svg"/>
              </div>
              <div className='flex' style={{gap: '8px'}}>
                <div className="sidebar-header-icon" onClick={newChat}>
                  <SquarePen size={16} strokeWidth={2}/>
                </div>
                {/* 
                  <div className="sidebar-header-icon">
                    <FolderPlus size={16} strokeWidth={2} />
                  </div> 
                */}
              </div>
            </div>
            <SideBarSearch isMobile={isMobile} handleSearch= {setSearchTerm}/>
          </div>
        )}
        {isCollapsed && (
          <div className="sidebar-head">
            <div className="sidebar-header" style={{justifyContent: 'center'}}>
              <div className="flex">
                <img style={{width:'100%', height: '24px'}} src="Yoja_logo.svg"/>
              </div>
            </div>
          </div>
        )}
        <div className="sidebar-body" style={{padding: '24px'}}>
          <div className="sidebar-body">
            {!isCollapsed && (
              <div className='chat-history'>
                <div>
                  <div className='history-header'>
                    <span className='history-header-text'> History </span>
                  </div>
                  {filteredChatHistory?.map((chats: any, index: number) => (
                    <div className='history-item' key={index} onClick={() => {
                      localStorage.setItem("current_chat", JSON.stringify(chats.content));
                      setMessages(chats.content);
                      const chatHistoryStorage: any[] = JSON.parse(localStorage.getItem('chat_history') || '[]');
                     // chatHistoryStorage.filter
                      const filteredHistory = chatHistoryStorage.filter((item: any) => {
                        let date1 = new Date(item.time);
                        let date2 = new Date(chats.time);
                        return (date1.getTime() !== date2.getTime());
                      });
                     // localStorage.setItem("chat_history", JSON.stringify(filteredHistory));
                    }}>
                    <span className='history-item-text'> {chats.content[0].content} </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          <UserMenu isCollapsed={isCollapsed}/>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;