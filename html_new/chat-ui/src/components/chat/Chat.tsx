import { Message, UserInfo } from "../../type";
import React, { useEffect, useRef, useState } from "react";
import ChatBottom from "./ChatBottom";
import { AnimatePresence, m, motion } from "framer-motion";
import { Avatar, AvatarImage } from "../ui/avatar";
import { EllipsisVertical, Files, RotateCcw, ThumbsDown, ThumbsUp } from "lucide-react";
import { ThreeDots } from "react-loader-spinner";

interface ChatProps {
  currentChat: Message[];
  userInfo: UserInfo;
  sendMessage: (newMessage: Message) => void;
  updateMessage: (newMessage: Message[]) => void;
  isMobile: boolean;
  isLoading: boolean;
}

export function Chat({
  currentChat,
  userInfo,
  sendMessage,
  updateMessage,
  isMobile,
  isLoading
}: ChatProps) {
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop =
        messagesContainerRef.current.scrollHeight;
    }
  }, [currentChat]);

  function dispaly(message: any): React.ReactNode {
    if(message.content) {
      let length = message.content.indexOf('**Context Source');
      if(length < 1) {
        length = message.content.indexOf('<!--');
        if(length < 1) {
          return message.content;
        } else {
          return message.content.substring(0, length);
        }
      } else {
        return message.content.substring(0, length);
      }
    } else {
      const dis = `\n\nI'm sorry, but I'm not sure what you're asking.`;
      return dis;
    }
  }

  const copyMessage = (message: any) => {
    let msg: string = "";
    if(message.content) {
      let length = message.content.indexOf('**Context Source');
      if(length < 1) {
        length = message.content.indexOf('<!--');
        if(length < 1) {
          msg = message.content;
        } else {
          msg = message.content.substring(0, length);
        }
      } else {
        msg = message.content.substring(0, length);
      }
    }
    navigator.clipboard.writeText(msg);
  }

  const disLikeMessage = (message: Message) => {
    if(message.dislike) {
      message.dislike = false;
      message.like = false;
    } else {
      message.dislike = true;
      message.like = false;
    }
    message.dislike = true;
    message.like = false;
    const copyActivevChat = currentChat;
    const ss = copyActivevChat.filter((chat=> chat.id === message.id))
    copyActivevChat.map((chat=> {
      if(chat.id === message.id) {
        chat.like = message.like;
        chat.dislike = message.dislike;
      } 
    }));
    //ss[0].content = 'kjhjhkjjk';
    //sendMessage(message)
    updateMessage(copyActivevChat);
  }

  const likeMessage = (message: Message) => {
    if(message.like) {
      message.dislike = false;
      message.like = false;

    } else {
      message.dislike = false;
      message.like = true;
    }
    // const copyActivevChat = currentChat;
    // copyActivevChat.map((chat=> {
    //   if(chat.id === message.id) {
    //     chat.like = message.like;
    //     chat.dislike = message.dislike;
    //   } 
    // }));
    // //setActiveChat(copyActivevChat);
    // updateMessage(copyActivevChat);
    const copyActivevChat = currentChat;
    const ss = copyActivevChat.filter((chat=> chat.id === message.id))
    copyActivevChat.map((chat=> {
      if(chat.id === message.id) {
        chat.like = message.like;
        chat.dislike = message.dislike;
      } 
    }));
    //ss[0].content = 'kjhjhkjjk';
    //sendMessage(message)
    updateMessage(copyActivevChat);
  }

  const modifyMessage = (msg: Message) => {
    currentChat?.map((message=> {
      if(message.id === msg.id) {
        return msg;
      } else {
        return message;
      }
    }));
    console.log('mess', currentChat);
    //setActiveChat([]);
  };

  return (
    <div className="chat-body chat-body-background">
      <div
        ref={messagesContainerRef}
        className="chat-body"
      > 
      {currentChat?.length ?
        <AnimatePresence>
          {currentChat?.map((msg, index) => (
            <div key={index}>
              { msg.role === "user" ? 
              <motion.div
                key={index}
                layout
                initial={{ opacity: 0, scale: 1, y: 50, x: 0 }}
                animate={{ opacity: 1, scale: 1, y: 0, x: 0 }}
                exit={{ opacity: 0, scale: 1, y: 1, x: 0 }}
                transition={{
                  opacity: { duration: 0.1 },
                  layout: {
                    type: "spring",
                    bounce: 0.3,
                    duration: currentChat.indexOf(msg) * 0.05 + 0.2,
                  },
                }}
                style={{
                  originX: 0.5,
                  originY: 0.5,
                }}
                className="chat-message-view"
              >
                <div className="chat-message-view1">
                  <div className="chat-message-view2">
                    <span className="chat-message-text"> {msg.content} </span>
                  </div>
                  <Avatar className="flex justify-center items-center">
                    <AvatarImage
                      src="./user.png"
                      width={6}
                      height={6}
                    />
                  </Avatar>
                </div>
              </motion.div> :
              
              <motion.div
                key={index}
                layout
                initial={{ opacity: 0, scale: 1, y: 50, x: 0 }}
                animate={{ opacity: 1, scale: 1, y: 0, x: 0 }}
                exit={{ opacity: 0, scale: 1, y: 1, x: 0 }}
                transition={{
                  opacity: { duration: 0.1 },
                  layout: {
                    type: "spring",
                    bounce: 0.3,
                    duration: currentChat.indexOf(msg) * 0.05 + 0.2,
                  },
                }}
                style={{
                  originX: 0.5,
                  originY: 0.5,
                }}
                className="chat-message-view"
              >
                <div className="chat-gpt-message">
                  <div style={{display: 'flex'}}>
                    <img style={{ height: '16px'}} src="Yoja.svg"/>
                    <span className="gpt-msg-header-text"> for Google Drive </span>
                  </div>
                  <div style={{width: '100%', padding: '12px'}}>
                    <span className="gpt-msg-text"> { dispaly(msg) } </span>
                  </div>
                  {  (msg.source &&  msg.source.length > 0) && 
                    <div>
                      <span className="gpt-msg-text"> Sources </span>
                      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap'}}>
                        {
                          msg.source?.map((source) =>  
                            <a className="gpt-msg-sourece" href={source.fullPath} target="_blank">
                              <img style={{width:'16px', height: '16px'}} src={source.extension === "doc" || source.extension === "docx" ? "docs.png" : "slide.png"}/>
                              <span className="gpt-source-name"> {source.name} </span>
                            </a>
                          )
                        }
                      </div>
                    </div>
                  }
                  <div className="gpt-msg-options">
                    <Files size={15} color="#616161" className="gpt-msg-icon" onClick={()=> {copyMessage(msg)}}/>
                    {/* <RotateCcw size={13} color="#616161"/> */}
                    <ThumbsUp size={15} color="#616161" fill={msg.like ? '#616161' : '#FFFFFF' } className="gpt-msg-icon" onClick={()=> {likeMessage(msg)}}/>
                    <ThumbsDown size={15} color="#616161" fill={msg.dislike ? '#616161' : '#FFFFFF' } className="gpt-msg-icon" onClick={()=> {disLikeMessage(msg)}}/>
                    {/* <EllipsisVertical size={13} color="#616161"/> */}
                  </div>
                </div>
              </motion.div> 
            }
            </div>
          ))}
          { isLoading ? 
            <ThreeDots
              visible={true}
              height="30"
              width="30"
              color="#4285F4"
              radius="3"
              ariaLabel="three-dots-loading"
              wrapperStyle={{}}
              wrapperClass=""
            /> : null
          }
        </AnimatePresence> : <div>  {/* </New chat view> */}  </div>}
      </div>
      <ChatBottom sendMessage={sendMessage} messages={currentChat} isMobile={isMobile} isLoading={isLoading}/>
    </div>
  );
}
