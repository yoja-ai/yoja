import os
import sys
import io
from typing import Tuple, List, Dict, Any, Self
import faiss_rm
from documentchunk import DocumentType, DocumentChunkDetails, DocumentChunkRange
from chatconfig import ChatConfiguration, RetrieverStrategyEnum
import ollama

ASSISTANTS_MODEL="llama3.2"

# Tool definitions
TOOL_SEARCH_QUESTION_IN_DB = {
    "type": "function",
    "function": {
        "name": "search_question_in_db",
        "description": "Search confidential and private information and return relevant passages for the given question or search and return relevant passages that provide details of the mentioned subject",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question for which passages need to be searched"
                },
            },
            "required": ["question"]
        }
    }
}
TOOL_SEARCH_QUESTION_IN_DB_RETURN_MORE = {
    "type": "function",
    "function": {
        "name": "search_question_in_db_return_more",
        "description": "Search confidential and private information and return relevant passages for the given question or search and return relevant passages that provide details of the mentioned subject. Use this tool only if you want additional results than those returned by the tool search_question_in_db",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question for which passages need to be searched"
                },
            },
            "required": ["question"]
        }
    }
}
TOOL_LIST_OF_FILES_FOR_GIVEN_QUESTION = {
    "type": "function",
    "function": {
        "name": "list_of_files_for_given_question",
        "description": "Search confidential and private information and return file names that can answer the given question. Use this tool only if the search_question_in_db tool does not work",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question for which file names need to be listed"
                },
                "number_of_files": {
                    "type":"integer",
                    "description":"The number of file names to return.  Default is to return 10 file names"
                }
            },
            "required": ["question"] #, "number_of_files"]
        }
    }
}

def retrieve_using_ollama_assistant(faiss_rms:List[faiss_rm.FaissRM], documents_list:List[Dict[str,str]],
                                    index_map_list:List[Tuple[str,str]], index_type, tracebuf:List[str],
                                    filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]],
                                    assistants_thread_id:str, chat_config:ChatConfiguration, last_msg:str,
                                    searchsubdir=None, toolprompts=None) -> Tuple[str, str]:
    """
    documents is a dict like {fileid: finfo}; 
    index_map is a list of tuples: [(fileid, paragraph_index)];  the index into this list corresponds to the index of the embedding vector in the faiss index 
    Returns the tuple (output, thread_id).  REturns (None, NOne) on failure.
    """
    if toolprompts:
        tools = toolprompts
        print(f"retrieve_using_ollama_assistant: toolprompts specified as {tools}")
    else:
        tools = [TOOL_SEARCH_QUESTION_IN_DB, TOOL_SEARCH_QUESTION_IN_DB_RETURN_MORE, TOOL_LIST_OF_FILES_FOR_GIVEN_QUESTION]
        print(f"retrieve_using_ollama_assistant: toolprompts not specified. Using default of {tools}")
    response = ollama.chat(model=ASSISTANTS_MODEL,
            messages = [{'role': 'user', 'content': last_msg}],
            tools = tools
            )
    print(response)
