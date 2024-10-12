import React, { useEffect, useState } from 'react';
import { EllipsisVertical, Pencil, Trash2, Download, SquarePen } from "lucide-react";
import SideBarSearch from './SideBarSearch';
import UserMenu from './UserMenu';
import { Dropdown, Menu } from 'antd';
import { ChatHistory, Message } from '../../type';
import ExportOptionsModal from './ExportOptionsModal';
import ProgressBar from "../chat/IndexingProgress";

interface SidebarProps {
  isCollapsed: boolean;
  isMobile: boolean;
  currentChat: Message[];
  setCurrentChat: (chat: Message[]) => void;
  chatHistory: ChatHistory[];
  setChatHistory: (history: ChatHistory[]) => void;
  handleChatSelect: (selectedChat: ChatHistory) => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  isCollapsed,
  isMobile,
  currentChat,
  setCurrentChat,
  chatHistory,
  setChatHistory,
  handleChatSelect
}) => {
  const [filteredChatHistory, setFilteredChatHistory] = useState<ChatHistory[]>([]);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedChatForExport, setSelectedChatForExport] = useState<ChatHistory | null>(null);

  useEffect(() => {
    const filteredHistory = chatHistory.filter((item) =>
      item.name.toLowerCase().includes(searchTerm.toLowerCase().trim()) && !item.isNew
    );
    setFilteredChatHistory(filteredHistory);
  }, [chatHistory, searchTerm]);

  const categorizeChats = (chats: ChatHistory[]): { [key: string]: ChatHistory[] } => {
    const now = new Date();
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterdayStart = new Date(todayStart);
    yesterdayStart.setDate(todayStart.getDate() - 1);
    const sevenDaysAgo = new Date(todayStart);
    sevenDaysAgo.setDate(todayStart.getDate() - 7);
    const thirtyDaysAgo = new Date(todayStart);
    thirtyDaysAgo.setDate(todayStart.getDate() - 30);

    const categories: { [key: string]: ChatHistory[] } = {
      'Today': [],
      'Yesterday': [],
      'Previous 7 Days': [],
      'Previous 30 Days': []
    };

    chats.forEach(chat => {
      const chatDate = new Date(chat.time);
      if (chatDate >= todayStart) {
        categories['Today'].push(chat);
      } else if (chatDate >= yesterdayStart) {
        categories['Yesterday'].push(chat);
      } else if (chatDate >= sevenDaysAgo) {
        categories['Previous 7 Days'].push(chat);
      } else if (chatDate >= thirtyDaysAgo) {
        categories['Previous 30 Days'].push(chat);
      }
    });

    return categories;
  };

  const categorizedChatHistory = categorizeChats(filteredChatHistory);

  const newChat = () => {
    if (currentChat.length) {
      // Clear the current chat
      setCurrentChat([]);
  
      // Find the currently selected chat and deselect it
      const updatedHistory = chatHistory.map(chat => ({
        ...chat,
        selected: false
      }));
  
      // Create the new chat
      const newChat: ChatHistory = {
        name: 'New Chat',
        content: [],
        time: new Date(),
        isNew: true,
        selected: true // This new chat is now selected
      };
  
      // Add the new chat to the updated history
      updatedHistory.push(newChat);
  
      // Update the state with the new history and set the new chat as current
      setChatHistory(updatedHistory);
      setCurrentChat(newChat.content);
    }
  };
  

  const handleChatClick = (chat: ChatHistory) => {
    handleChatSelect(chat);
  };

  const handleMenuClick = (e: { key: string }, chat: ChatHistory) => {
    const action = e.key;
    if (action === "delete") {
      // Filter out the deleted chat from both histories
      const newChatHistory = chatHistory.filter(item => item !== chat);
      const newFilteredChatHistory = filteredChatHistory.filter(item => item !== chat);
  
      // Temporarily set a dummy state to force a re-render
      setChatHistory([]);
      setFilteredChatHistory([]);
  
      // Update state and localStorage
      setTimeout(() => {
        setChatHistory(newChatHistory);
        setFilteredChatHistory(newFilteredChatHistory);
        localStorage.setItem("chat_history", JSON.stringify(newChatHistory));
  
        // Clear the current chat if it was the one being deleted
        if (currentChat === chat.content) {
          setCurrentChat([]);
        }
      }, 0);
    } else if (action === "download") {
      setSelectedChatForExport(chat);
      setIsModalVisible(true);
    }
  };
  
  

  const menu = (chat: ChatHistory) => (
    <Menu onClick={(e) => handleMenuClick(e, chat)}>
      <Menu.Item key="edit" icon={<Pencil size={16} />}>Rename</Menu.Item>
      <Menu.Item key="download" icon={<Download size={16} />}>Export</Menu.Item>
      <Menu.Item key="delete" icon={<Trash2 size={16} style={{ color: 'red' }} />} style={{ color: 'red' }}>
        Delete
      </Menu.Item>
    </Menu>
  );

  return (
    <div data-collapsed={isCollapsed} style={isCollapsed ? { flex: '0 1 0px' } : { flex: '20 1 0px' }}>
      <div className="sidebar">
        <div className="sidebar-head">
          <div className="sidebar-header">
            <div className="flex">
              <img style={{ width: '100%', height: '24px' }} src="Yoja.svg" />
            </div>
            <div className='flex' style={{ gap: '8px' }}>
              <div className="sidebar-header-icon" onClick={newChat}>
                <SquarePen size={14} strokeWidth={2} />
              </div>
            </div>
          </div>
          <SideBarSearch isMobile={isMobile} handleSearch={setSearchTerm} />
        </div>
        <div className="sidebar-body" style={{ padding: '24px' }}>
          <div className="sidebar-body">
            {Object.entries(categorizedChatHistory).map(([category, chats]) => (
              chats.length > 0 && ( // Render category only if it has chats
                <div key={category} className="chat-category">
                  <h3>{category}</h3>
                  <div className='chat-history'>
                    {chats.map((chat) => (
                      <div
                        key={`${chat.name}-${chat.time}`} // Use a combination of name and time as a key
                        className={`history-item ${chat.selected ? 'selected' : ''}`}
                        onClick={() => handleChatClick(chat)}
                      >
                        <span className='history-item-text'>{chat.name}</span>
                        <Dropdown overlay={menu(chat)} trigger={['click']}>
                          <div className="history-item-menu"><EllipsisVertical size={16} color="#71717a" /></div>
                        </Dropdown>
                      </div>
                    ))}
                  </div>
                </div>
              )
            ))}
          </div>
          <UserMenu isCollapsed={isCollapsed} />
          <ProgressBar/>
        </div>
        {selectedChatForExport && (
          <ExportOptionsModal
            chat={selectedChatForExport}
            isVisible={isModalVisible}
            onClose={() => setIsModalVisible(false)}
          />
        )}
      </div>
    </div>
  );
};

export default Sidebar;
