import React, { Fragment, useEffect, useState } from 'react';
import { EllipsisVertical, FolderPlus, LogOut, Pencil, Settings, SquarePen, Trash2 } from "lucide-react";
import SideBarSearch from './SideBarSearch';
import { Avatar, AvatarImage } from '../ui/avatar';
import { ChatHistory, Message } from '../../type';
import UserMenu from './UserMenu';
import { Menu, Modal } from 'antd';
import SubMenu from "antd/es/menu/SubMenu";

interface SidebarProps {
  isCollapsed: boolean;
  onClick?: () => void;
  isMobile: boolean;
  currentChat: Message[];
  setCurrentChat: any;
  chatHistory: ChatHistory[];
  setChatHistory: any;
}

const Sidebar = ({ isCollapsed, isMobile, setCurrentChat, chatHistory, setChatHistory, currentChat }: SidebarProps) => {
  const [filteredChatHistory, setFilteredChatHistory] = useState<ChatHistory[]>([]);
  const [searchTerm, setSearchTerm] = useState<string>('');

  useEffect(() => {
    const historyItems = JSON.parse(localStorage.getItem('chat_history') || '[]');
    const filteredHistory = chatHistory?.filter((item: any) => {
      return (item.name).toLowerCase().includes(searchTerm.toLowerCase().trim());
    });
    setFilteredChatHistory(filteredHistory);
  }, [chatHistory, searchTerm]);

  const newChat = () => {
    if(currentChat.length) {
      setCurrentChat([]);
      localStorage.setItem("current_chat", JSON.stringify([]));
      const newChat: ChatHistory = {
        name: 'New Chat',
        content: [],
        time: new Date()
      };
      const updateHistory = [...chatHistory, newChat];
      setChatHistory(updateHistory);
      localStorage.setItem("chat_history", JSON.stringify(updateHistory));
    }
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
                  <SquarePen size={14} strokeWidth={2}/>
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
                      setCurrentChat(chats.content);
                      const chatHistoryStorage: any[] = JSON.parse(localStorage.getItem('chat_history') || '[]');
                      const filteredHistory = chatHistoryStorage.filter((item: any) => {
                        let date1 = new Date(item.time);
                        let date2 = new Date(chats.time);
                        return (date1.getTime() !== date2.getTime());
                      });
                    }}>
                    <span className='history-item-text'> {chats.content[0] ? chats.content[0].content : 'New Chat'} </span>
                    <div className="history-item-menu" onClick={()=> {
                      setCurrentChat([]);
                      localStorage.setItem("current_chat", JSON.stringify([]));
                      filteredChatHistory.splice(index, 1)
                      setFilteredChatHistory(filteredChatHistory);
                      setChatHistory(filteredChatHistory);
                      localStorage.setItem("chat_history", JSON.stringify(filteredChatHistory));
                      setCurrentChat([]);
                      localStorage.setItem("current_chat", JSON.stringify([]));
                    }}> <Trash2 size={16} color="#71717a"/> </div>

                      {/* <div className="history-menu-container">
                        <Menu key="chat-history-item" className="history-item-menu">
                          <SubMenu
                            title={
                              <div className="history-item-menu"> <EllipsisVertical size={16} color="#71717a"/> </div> 
                            }
                          >
                            <Menu.Item key="Settings" onClick={()=> {
                              filteredChatHistory.splice(index, 1); 
                               const chatHistoryStorage: any[] = JSON.parse(localStorage.getItem('chat_history') || '[]');
                               const filteredHistory = filteredChatHistory.filter((item: any) => item.name === chats.name);
                             }}>
                              <span style={{display: 'flex', justifyContent:'center', alignItems:'center', gap: '5px'}}> <Pencil size={16} /> Rename </span>
                            </Menu.Item>
                            <Menu.Item key="SignOut"  onClick={()=> {}}>
                              <div style={{display: 'flex', justifyContent:'center', alignItems:'center',  gap: '5px', color: 'red'}}> <Trash2 size={16} /> Delete </div>
                            </Menu.Item>
                          </SubMenu>
                        </Menu>
                      </div> */}
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