import { Message, UserInfo } from "../../type";
import React, { useRef } from "react";
import ChatBottom from "./ChatBottom";
import { AnimatePresence, m, motion } from "framer-motion";
import { Avatar, AvatarImage } from "../ui/avatar";
import { EllipsisVertical, Files, RotateCcw, ThumbsDown, ThumbsUp } from "lucide-react";
import { ThreeDots } from "react-loader-spinner";

declare var window: any

interface ChatProps {
  messages?: Message[];
  userInfo: UserInfo;
  sendMessage: (newMessage: Message) => void;
  isMobile: boolean;
  isLoading: boolean;
}

export function Chat({
  messages,
  userInfo,
  sendMessage,
  isMobile,
  isLoading
}: ChatProps) {
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop =
        messagesContainerRef.current.scrollHeight;
    }
  }, [messages]);

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

  return (
    <div className="chat-body chat-body-background">
      <div
        ref={messagesContainerRef}
        className="chat-body"
      > 
      {messages?.length ?
        <AnimatePresence>
          {messages?.map((message, index) => (
            <div key={index}>
              { message.role === "user" ? 
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
                    duration: messages.indexOf(message) * 0.05 + 0.2,
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
                    <span className="chat-message-text"> {message.content} </span>
                  </div>
                  <Avatar className="flex justify-center items-center">
                    <AvatarImage
                      src={window.picture}
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
                    duration: messages.indexOf(message) * 0.05 + 0.2,
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
                  <div style={{width: '100%', padding: '24px'}}>
                    <span className="gpt-msg-text"> { dispaly(message) } </span>
                  </div>
                  <div>
                    <span className="gpt-msg-text"> Sources </span>
                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap'}}>
                      {
                        message.source?.map((source) =>  
                          <a className="gpt-msg-sourece" href={source.fullPath} target="_blank">
                            <img style={{width:'16px', height: '16px'}} src={source.extension === "doc" || source.extension === "docx" ? "docs.png" : "slide.png"}/>
                            <span className="gpt-source-name"> {source.name} </span>
                          </a>
                          // <div style={{ display: 'flex', gap: '8px'}}>
                          //   <div className="gpt-msg-sourece">
                          //     <img style={{width:'16px', height: '16px'}} src="slide.png"/>
                          //     <span className="gpt-source-name"> {source.name} </span>
                          //   </div>
                          //   {/* <div className="gpt-msg-sourece">
                          //     <img style={{width:'16px', height: '16px'}} src="docs.png"/>
                          //     <span className="gpt-source-name"> InfinStor One Pager.docx </span>
                          //   </div> */}
                          // </div>
                        )
                      }
                    </div>
                  </div>
                  <div className="gpt-msg-options">
                    <Files size={13} color="#616161"/>
                    {/* <RotateCcw size={13} color="#616161"/> */}
                    <ThumbsUp size={13} color="#616161"/>
                    <ThumbsDown size={13} color="#616161"/>
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
      <ChatBottom sendMessage={sendMessage} messages={messages} isMobile={isMobile}/>
    </div>
  );
}
