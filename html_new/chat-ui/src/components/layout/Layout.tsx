import React, { useEffect, useState } from "react";
import Sidebar from "../sidebar/Sidebar";
import { ChatLayout } from "./ChatLayout";
import { userData } from "../chat/data";
import { Message } from "../../type";

interface LayoutProps {
  defaultLayout: number[] | undefined;
  defaultCollapsed?: boolean;
}

const Layout = () => {
  const defaultLayout = [320, 480];
  const navCollapsedSize = 8;
  const [isCollapsed, setIsCollapsed] = React.useState(false);
  const [user, setuser] = React.useState();
  const [isMobile, setIsMobile] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);

  useEffect(() => {
    const items = JSON.parse(localStorage.getItem('current_chat') || '[]');
    if (items) {
      setMessages(items);
    }
  }, []);

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
  
  return (
    <div className="app-layout">
      <Sidebar
        isCollapsed={isCollapsed}
        isMobile={isMobile}
        messages={messages}
        setMessages={setMessages}
      />
      <ChatLayout
        messages={messages}
        setMessages={setMessages}
        userInfo={user}
        setIsCollapsed={setIsCollapsed}
        isCollapsed={isCollapsed}
        isMobile={isMobile}
      />
    </div>
  );
}

export default Layout;