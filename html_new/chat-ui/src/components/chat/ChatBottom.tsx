import {
    FileImage,
    Mic,
    Paperclip,
    PlusCircle,
    X,
    SendHorizontal,
    Smile,
    ThumbsUp,
    SmileIcon
  } from "lucide-react";
  import React, { useRef, useState } from "react";
  import { AnimatePresence, motion } from "framer-motion";
  import { Message, loggedInUserData } from "../../type";
  import { Textarea } from "../ui/textarea";
  interface ChatBottomProps {
    sendMessage: (newMessage: Message) => void;
    isMobile: boolean;
    messages: any;
    isLoading: boolean;
  }
  
  export const BottombarIcons = [{ icon: FileImage }, { icon: Paperclip }];
  
  export default function ChatBottom({
    sendMessage, isMobile, messages, isLoading
  }: ChatBottomProps) {
    const [message, setMessage] = useState("");
    const inputRef = useRef<HTMLTextAreaElement>(null);
  
    const handleInputChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      setMessage(event.target.value);
    };
  
    const handleThumbsUp = () => {
      const newMessage: Message = {
        id: message.length + 1,
        role: 'user',
        content: "👍",
      };
      sendMessage(newMessage);
      setMessage("");
    };
  
    const handleSend = () => {
      if (message.trim()) {
        const newMessage: Message = {
          id: messages.length + 1,
          role: 'user',
          content: message.trim(),
        };
        sendMessage(newMessage);
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
      <div className="chat-bottom">
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
            <Textarea
              autoComplete="off"
              value={message}
              ref={inputRef}
              onKeyDown={handleKeyPress}
              onChange={handleInputChange}
              name="message"
              placeholder="Type your message here"
              disabled={isLoading}
            ></Textarea>
            <div className="chat-box-icons">
                <SendHorizontal size={16} className="chat-box-send-icon" opacity={ isLoading ? 0.5 : 1} onClick={handleSend}/>
                {message.trim() ? (
                    <X size={16} color="#71717a" className="chat-box-icon" onClick={() => setMessage("")}/>
                ) : (
                    null
                )}
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    );
  }
  