import {
    Pencil
  } from "lucide-react";
  import React, { useRef, useState, useEffect } from "react";
  import { AnimatePresence, motion } from "framer-motion";
  import { SearchsubdirTitleTextarea, SearchsubdirTextarea} from "../ui/textarea";
  interface ChatTopProps {
    searchSubdir: string;
    sendSearchSubdir: (newSearchSubdir: string) => void;
  }
  
  export default function ChatTop({
    searchSubdir,
    sendSearchSubdir
  }: ChatTopProps) {
    const [newSearchSubdir, setNewSearchSubdir] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const inputRef = useRef<HTMLTextAreaElement>(null);
 
    useEffect(() => {
      setNewSearchSubdir(searchSubdir);
    }, [searchSubdir]);

    const handleInputChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      setNewSearchSubdir(event.target.value);
    };
  
    const handleSend = () => {
      if (newSearchSubdir.trim()) {
        if (inputRef.current) {
          inputRef.current.focus();
        }
        sendSearchSubdir(newSearchSubdir);
      }
    };
  
    const handleKeyPress = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === "Enter") {
        event.preventDefault();
        handleSend();
      }
    };
  
    return (
      <div className="searchsubdir-bar">
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
            <SearchsubdirTitleTextarea
              autoComplete="off"
              value="Search Subdirectory"
              disabled={true}
            ></SearchsubdirTitleTextarea>
            <SearchsubdirTextarea
              autoComplete="off"
              value={newSearchSubdir}
              ref={inputRef}
              onKeyDown={handleKeyPress}
              onChange={handleInputChange}
              name="searchsubdir"
              placeholder="None specified"
              disabled={isLoading}
            ></SearchsubdirTextarea>
            <div className="searchsubdir-box-icons">
                <Pencil size={16} className="searchsubdir-box-send-icon" opacity={ isLoading ? 0.5 : 1} onClick={handleSend}/>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    );
  }
  
