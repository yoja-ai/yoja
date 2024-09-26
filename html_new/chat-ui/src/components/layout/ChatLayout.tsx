import { Message, SourceFile, UserInfo } from "../../type";
import React, { useEffect, useRef, useState } from "react";
import { Menu, Copy, ThumbsDown, ThumbsUp, CopyCheck } from 'lucide-react';
import ChatBottom from "../chat/ChatBottom";
import ChatTop from "../chat/ChatTop";
import ProgressBar from "../chat/IndexingProgress";
import { AnimatePresence, motion } from "framer-motion";
import { Avatar, AvatarImage } from "../ui/avatar";
import { ThreeDots } from "react-loader-spinner";
import { Tooltip, Popover } from "antd";

interface ChatProps {
  currentChat: Message[];
  setCurrentChat: any;
  userInfo: UserInfo;
  isMobile: boolean;
  isLoading: boolean;
  isCollapsed: boolean;
  setIsCollapsed: any;
  sendMessage: any;
  searchSubdir: any;
  sendSearchSubdir: any;
}

export function ChatLayout({ currentChat, userInfo, isMobile, setIsCollapsed, isCollapsed, setCurrentChat, sendMessage, isLoading, searchSubdir, sendSearchSubdir }: ChatProps) {
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messagesContainerRef.current) { 
      messagesContainerRef.current.scrollTop =
        messagesContainerRef.current.scrollHeight;
    }
  }, [currentChat]);

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

  const clearChat = () => {
    setCurrentChat([]);
    localStorage.setItem("current_chat", JSON.stringify([]));
  }

  function displayMessage(message: Message): React.ReactNode {
    let content = message.content;
    let tokenInfo: React.ReactNode = null;

    // Regex to match the token information pattern
    const tokenRegex = /\*\*Tokens:\*\*\s*prompt=(\d+)\(([\d.]+)%\sof\s2048\),\s*completion=(\d+)/;

    // Extract token information if present in the content
    const match = content.match(tokenRegex);
    if (match) {
        const promptTokens = match[1];
        const promptPercentage = match[2];
        const completionTokens = match[3];

        tokenInfo = (
            <div className="token-info">
                <span className="gpt-msg-text" style={{ fontWeight: 'bold' }}>Tokens</span>
                <div>
                    <div>Prompt: {promptTokens} ({promptPercentage}% of 2048)</div>
                    <div>Completion: {completionTokens}</div>
                </div>
            </div>
        );

        // Remove the token info from the content, but preserve context sources
        content = content.replace(tokenRegex, '').trim();
    }

    // Now process the message content to remove the context source indicators
    let length = content.indexOf('**Context Source');
    if (length >= 0) {
        content = content.substring(0, length);
    }

    return (
        <div>
            <div>{content}</div>
            {tokenInfo}
        </div>
    );
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
    const updatedChat = currentChat.map((chat=> {
      if(chat.id === message.id) {
        chat.copied = true;
      }
      return chat;
    }));
    setCurrentChat(updatedChat);
    setTimeout(() => {
      const updatedChat = currentChat.map((chat=> {
        if(chat.id === message.id) {
          chat.copied = false;
        }
        return chat;
      }));
      setCurrentChat(updatedChat);
    }, 4000);
  }

  const disLikeMessage = (message: Message) => {
    if(message.dislike) {
      message.dislike = false;
      message.like = false;
    } else {
      message.dislike = true;
      message.like = false;
    }
    const updatedChat = currentChat.map((chat=> {
      if(chat.id === message.id) {
        chat.like = message.like;
        chat.dislike = message.dislike;
      }
      return chat;
    }));
    setCurrentChat(updatedChat);
    localStorage.setItem("current_chat", JSON.stringify(updatedChat));
  }

  const likeMessage = (message: Message) => {
    if(message.like) {
      message.dislike = false;
      message.like = false;
    } else {
      message.dislike = false;
      message.like = true;
    }
    const updatedChat = currentChat.map((chat=> {
      if(chat.id === message.id) {
        chat.like = message.like;
        chat.dislike = message.dislike;
      }
      return chat;
    }));
    setCurrentChat(updatedChat);
    localStorage.setItem("current_chat", JSON.stringify(updatedChat));
  }

  return (
    <div style={{flex: '80 1 0px'}} >
      <div className="chat">
        {/* <div className="chat-header-bar"> 
          <div className="chat-header">
            <Menu size={20} style={{cursor: 'pointer'}} onClick={()=> {setIsCollapsed(!isCollapsed)}}/> 
            <span className="chat-header-text"> Yoja.ai </span>
          </div>
        </div> */}
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
                      <div style={{display: 'flex', alignItems: 'center'}}>
                        <img style={{ height: '18px'}} src="Yoja.svg"/>
                        {/* <span className="gpt-msg-header-text"> for Google Drive </span> */}
                      </div>
                      <div style={{width: '100%', padding: '12px'}}>
                        <span className="gpt-msg-text"> { displayMessage(msg) } </span>
                      </div>
                      {  (msg.source &&  msg.source.length > 0) && 
                        <div>
                          <span className="gpt-msg-source-title">
                            {msg.source.length > 1 ? "Sources" : "Source"}
                          </span>
                          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap'}}>
                            {
                              msg.source?.map((source) =>  
                                <a className="gpt-msg-source" href={source.fullPath} target="_blank">
                                  <img style={{width:'16px', height: '16px'}} src={source.extension === "doc" || source.extension === "docx" ? "docs.png" : "slide.png"}/>
                                  <span className="gpt-source-name"> {source.name} </span>
                                </a>
                              )
                            }
                          </div>
                        </div>
                      }
                      {  (msg.sample_source &&  msg.sample_source.length > 0) && 
                        <div>
                          <span className="gpt-msg-source-title">
                            Note: Indexing of your personal files is in progress. In the meantime, a sample dataset of the following files is used
                          </span>
                          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap'}}>
                            {
                              msg.sample_source?.map((sample_source) =>  
                                <span className="gpt-msg-source">
                                  <span className="gpt-source-name"> {sample_source} </span>
                                </span>
                              )
                            }
                          </div>
                        </div>
                      }
                      <div className="gpt-msg-options">
                        { !msg.copied && <Copy size={16} color="#616161" className="gpt-msg-icon" onClick={()=> {copyMessage(msg)}}/>}
                        { msg.copied &&  <Tooltip placement="topLeft" title='copyed' open>
                          <CopyCheck size={16} color="#616161"  className="gpt-msg-icon"/>
                        </Tooltip> }
                        {/* <Files size={15} color="#616161" className="gpt-msg-icon" onClick={()=> {copyMessage(msg)}}/> */}
                        {/* <RotateCcw size={13} color="#616161"/> */}
                        <ThumbsUp size={16} color="#616161" fill={msg.like ? '#616161' : '#FFFFFF' } className="gpt-msg-icon" onClick={()=> {likeMessage(msg)}}/>
                        <ThumbsDown size={16} color="#616161" fill={msg.dislike ? '#616161' : '#FFFFFF' } className="gpt-msg-icon" onClick={()=> {disLikeMessage(msg)}}/>
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
          <ProgressBar/>
          <ChatTop searchSubdir={searchSubdir} sendSearchSubdir={sendSearchSubdir}/>
          <ChatBottom sendMessage={sendMessage} messages={currentChat} isMobile={isMobile} isLoading={isLoading}/>
        </div>
      </div>
    </div>
  );
}
