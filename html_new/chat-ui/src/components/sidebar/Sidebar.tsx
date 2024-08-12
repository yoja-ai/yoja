import React, { useEffect, useState } from 'react';
import { EllipsisVertical, Pencil, Trash2, Download, SquarePen } from "lucide-react";
import SideBarSearch from './SideBarSearch';
import UserMenu from './UserMenu';
import { Dropdown, Menu } from 'antd';
import { ChatHistory, Message } from '../../type';
import ExportOptionsModal from './ExportOptionsModal';

interface SidebarProps {
  isCollapsed: boolean;
  isMobile: boolean;
  currentChat: Message[];
  setCurrentChat: (chat: Message[]) => void;
  chatHistory: ChatHistory[];
  setChatHistory: (history: ChatHistory[]) => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  isCollapsed,
  isMobile,
  currentChat,
  setCurrentChat,
  chatHistory,
  setChatHistory
}) => {
  const [filteredChatHistory, setFilteredChatHistory] = useState<ChatHistory[]>([]);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedChatForExport, setSelectedChatForExport] = useState<ChatHistory | null>(null);
  const [selectedChatIndex, setSelectedChatIndex] = useState<number | null>(null);

  useEffect(() => {
    const filteredHistory = chatHistory.filter((item) =>
      item.name.toLowerCase().includes(searchTerm.toLowerCase().trim()) && !item.isNew
    );
    setFilteredChatHistory(filteredHistory);
  }, [chatHistory, searchTerm]);

  useEffect(() => {
    if (chatHistory.length === 0) { // Only initialize if chatHistory is empty
      const historyItems = JSON.parse(localStorage.getItem('chat_history') || '[]');
      if (historyItems.length > 0) {
        setChatHistory(historyItems);
        setSelectedChatIndex(0); // Select the first chat in the list
        setCurrentChat(historyItems[0].content); // Set the content of the first chat as the current chat
        localStorage.setItem("current_chat", JSON.stringify(historyItems[0].content));
      } else {
        console.log("No chats found, consider initializing a new chat if needed");
      }
    }
  }, []);

  const newChat = () => {
    if (currentChat.length) {
      setCurrentChat([]);
      localStorage.setItem("current_chat", JSON.stringify([]));
      const newChat: ChatHistory = {
        name: 'New Chat',
        content: [],
        time: new Date(),
        isNew: true // Set this chat as new
      };
      const updatedHistory = [...chatHistory, newChat];
      setChatHistory(updatedHistory);
      localStorage.setItem("chat_history", JSON.stringify(updatedHistory));
      setCurrentChat(newChat.content);
      setSelectedChatIndex(updatedHistory.length - 1); // Index of the new chat at the end of the list
    }
  };

  const handleChatClick = (chat: ChatHistory, index: number) => {
    if (!chat.isNew) { // Ensure the chat is not new before selecting it
      setSelectedChatIndex(index);
      localStorage.setItem("current_chat", JSON.stringify(chat.content));
      setCurrentChat(chat.content);
    }
  };

  const handleMenuClick = (e: { key: string }, chatIndex: number) => {
    const action = e.key;
    if (action === "delete") {
      const newFilteredChatHistory = filteredChatHistory.filter((_, idx) => idx !== chatIndex);
      setFilteredChatHistory(newFilteredChatHistory);
      setChatHistory(newFilteredChatHistory);
      localStorage.setItem("chat_history", JSON.stringify(newFilteredChatHistory));
      setCurrentChat([]);
      localStorage.setItem("current_chat", JSON.stringify([]));
    } else if (action === "download") {
      setSelectedChatForExport(filteredChatHistory[chatIndex]);
      setIsModalVisible(true);
    }
  };

  const menu = (chatIndex: number) => (
    <Menu onClick={(e) => handleMenuClick(e, chatIndex)}>
      <Menu.Item key="edit" icon={<Pencil size={16} />}>Edit</Menu.Item>
      <Menu.Item key="download" icon={<Download size={16} />}>Export</Menu.Item>
      <Menu.Item key="delete" icon={<Trash2 size={16} style={{ color: 'red' }} />} style={{ color: 'red' }}>
        Delete
      </Menu.Item>
    </Menu>
  );

  // Function to categorize chat history based on time
  const categorizeChats = () => {
    const now = new Date();
    const today: ChatHistory[] = [];
    const yesterday: ChatHistory[] = [];
    const previous7Days: ChatHistory[] = [];
    const previous30Days: ChatHistory[] = [];

    filteredChatHistory.forEach(chat => {
      const chatDate = new Date(chat.time);
      const diffTime = now.getTime() - chatDate.getTime();
      const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));

      if (diffDays === 0) {
        today.push(chat);
      } else if (diffDays === 1) {
        yesterday.push(chat);
      } else if (diffDays <= 7) {
        previous7Days.push(chat);
      } else if (diffDays <= 30) {
        previous30Days.push(chat);
      }
    });

    return { today, yesterday, previous7Days, previous30Days };
  };

  const { today, yesterday, previous7Days, previous30Days } = categorizeChats();

  return (
    <div data-collapsed={isCollapsed} style={isCollapsed ? {flex: '0 1 0px'} : {flex: '20 1 0px'}}>
      <div className="sidebar">
        <div className="sidebar-head">
          <div className="sidebar-header">
            <div className="flex">
              <img style={{width:'100%', height: '24px'}} src="Yoja.svg"/>
            </div>
            <div className='flex' style={{gap: '8px'}}>
              <div className="sidebar-header-icon" onClick={newChat}>
                <SquarePen size={14} strokeWidth={2}/>
              </div>
            </div>
          </div>
          <SideBarSearch isMobile={isMobile} handleSearch={setSearchTerm} />
        </div>
        <div className="sidebar-body" style={{ padding: '24px' }}>
          <div className="sidebar-body">
            <div className='chat-history'>
              {today.length > 0 && (
                <div>
                  <div className='history-header'>
                    <span className='history-header-text'>Today</span>
                  </div>
                  {today.map((chat, index) => (
                    <div
                      key={`today-${index}`}
                      className={`history-item ${selectedChatIndex === index ? 'selected' : ''}`}
                      onClick={() => handleChatClick(chat, index)}
                    >
                      <span className='history-item-text'>{chat.name}</span>
                      <Dropdown overlay={menu(index)} trigger={['click']}>
                        <div className="history-item-menu"><EllipsisVertical size={16} color="#71717a"/></div>
                      </Dropdown>
                    </div>
                  ))}
                </div>
              )}
              {yesterday.length > 0 && (
                <div>
                  <div className='history-header'>
                    <span className='history-header-text'>Yesterday</span>
                  </div>
                  {yesterday.map((chat, index) => (
                    <div
                      key={`yesterday-${index}`}
                      className={`history-item ${selectedChatIndex === index ? 'selected' : ''}`}
                      onClick={() => handleChatClick(chat, index)}
                    >
                      <span className='history-item-text'>{chat.name}</span>
                      <Dropdown overlay={menu(index)} trigger={['click']}>
                        <div className="history-item-menu"><EllipsisVertical size={16} color="#71717a"/></div>
                      </Dropdown>
                    </div>
                  ))}
                </div>
              )}
              {previous7Days.length > 0 && (
                <div>
                  <div className='history-header'>
                    <span className='history-header-text'>Previous 7 Days</span>
                  </div>
                  {previous7Days.map((chat, index) => (
                    <div
                      key={`previous7Days-${index}`}
                      className={`history-item ${selectedChatIndex === index ? 'selected' : ''}`}
                      onClick={() => handleChatClick(chat, index)}
                    >
                      <span className='history-item-text'>{chat.name}</span>
                      <Dropdown overlay={menu(index)} trigger={['click']}>
                        <div className="history-item-menu"><EllipsisVertical size={16} color="#71717a"/></div>
                      </Dropdown>
                    </div>
                  ))}
                </div>
              )}
              {previous30Days.length > 0 && (
                <div>
                  <div className='history-header'>
                    <span className='history-header-text'>Previous 30 Days</span>
                  </div>
                  {previous30Days.map((chat, index) => (
                    <div
                      key={`previous30Days-${index}`}
                      className={`history-item ${selectedChatIndex === index ? 'selected' : ''}`}
                      onClick={() => handleChatClick(chat, index)}
                    >
                      <span className='history-item-text'>{chat.name}</span>
                      <Dropdown overlay={menu(index)} trigger={['click']}>
                        <div className="history-item-menu"><EllipsisVertical size={16} color="#71717a"/></div>
                      </Dropdown>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
          <UserMenu isCollapsed={isCollapsed} />
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
