import {
    FileImage,
    Mic,
    Paperclip,
    X,
    EllipsisVertical,
    SendHorizontal,
    Search
  } from "lucide-react";
  import React, { useRef, useState } from "react";
  import { AnimatePresence, motion } from "framer-motion";
  import { Message, loggedInUserData } from "../../type";
  import { Textarea } from "../ui/textarea";
  interface SideBarSearchProps {
    isMobile: boolean;
    handleSearch: any;
  }
  
  export const BottombarIcons = [{ icon: FileImage }, { icon: Paperclip }];
  
  export default function SideBarSearch({
    isMobile,
    handleSearch
  }: SideBarSearchProps) {
    const [message, setMessage] = useState("");
    const inputRef = useRef<HTMLTextAreaElement>(null);
  
    const handleInputChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      setMessage(event.target.value);
      handleSearch(event.target.value);
    };
      
    const handleSend = () => {
      if (message.trim()) {
        const newMessage: Message = {
          id: message.length + 1,
          role: loggedInUserData.name,
          content: message.trim(),
        };
        setMessage("");
        if (inputRef.current) {
          inputRef.current.focus();
        }
      }
    };
  
    const handleKeyPress = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        handleSend();
      }
  
      if (event.key === "Enter" && event.shiftKey) {
        event.preventDefault();
        setMessage((prev) => prev + "\n");
      }
    };
  
    return (
      <div className="sidebar-search">
        <AnimatePresence initial={false}>
          <motion.div
            key="input"
            className="text-box-div"
            layout
            initial={{ opacity: 0, scale: 1 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1 }}
            transition={{
              opacity: { duration: 0.05 },
              layout: {
                type: "spring",
                bounce: 0.15,
              },
            }}
          >
            <div className="search-icon">
              <Search size={16} color="#71717a" />
            </div>
            <Textarea
              autoComplete="off"
              value={message}
              ref={inputRef}
              onKeyDown={handleKeyPress}
              onChange={handleInputChange}
              name="message"
              placeholder="Search"
              style={{paddingLeft: '30px'}}
            ></Textarea>
            <div className="chat-box-icons">
              {message.trim() ? (
                <div className="chat-box-icon" onClick={() => {setMessage(""); handleSearch("");}}> <X size={16} color="#71717a" /> </div>
              ) : (
                  null
              )}
              <div className="chat-box-icon"> <EllipsisVertical size={16} color="#71717a"/> </div>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    );
  }
  