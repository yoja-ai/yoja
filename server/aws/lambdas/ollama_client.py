import os
import sys
import io
import json
from typing import Tuple, List, Dict, Any, Self
from utils import prtime
import faiss_rm
from documentchunk import DocumentType, DocumentChunkDetails, DocumentChunkRange
from chatconfig import ChatConfiguration, RetrieverStrategyEnum
import tiktoken
import ollama
from ollama import chat
from yoja_retrieve import get_context, get_filelist_using_retr_and_rerank

ASSISTANTS_MODEL="llama3.2"

# tokenizer for llama 3.2 is supposed to be tiktoken
# Hopefully the following model gives us a reasonable approximation
encoding_model=tiktoken.encoding_for_model('gpt-3.5-turbo')

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

def _extract_main_theme(text):
    messages = [{'role': 'user', 'content': f"Extract the main topic as a single word in the following sentence and return the result as a single word: {text}"}]
    response = chat(ASSISTANTS_MODEL, messages=messages)
    return response['message']['content']

def _extract_named_entities(text):
    messages = [{'role': 'user', 'content': f"Extract any named entities present in the sentence. Return a JSON object in this format: {{ 'entities': ['entity1', 'entity2'] }}, without any additional text or explanation. Particularly, do not include text before or after the parseable JSON: {text}"}]
    response = chat(ASSISTANTS_MODEL, messages=messages)
    content = response['message']['content']
    try:
        js = json.loads(content)
    except Exception as ex:
        print(f"ollama_client._extract_named_entities: caught {ex}")
        return None
    if 'entities' in js and js['entities']:
        return js['entities']
    else:
        return None

def _calc_tokens(context):
    global encoding_model
    return len(encoding_model.encode(context))

def _process_tool_call(supplied_tools, tool,
                        faiss_rms, documents_list, index_map_list, index_type,
                        tracebuf, filekey_to_file_chunks_dict, chat_config,
                        searchsubdir):
    tool_outputs = ""
    func = tool.function
    for t1 in supplied_tools:
        if t1['function']['name'] == func.name:
            args_dict = {}
            for arg, val in func.arguments.items():
                args_dict[arg] = val
            print(f"function={func.name}, arguments={args_dict}")
            if tool.function.name == "search_question_in_db":
                tool_arg_question = args_dict.get('question')
                context:str = get_context(faiss_rms, documents_list, index_map_list, index_type, tracebuf,
                                            filekey_to_file_chunks_dict, chat_config, tool_arg_question,
                                            True, False, searchsubdir=searchsubdir, calc_tokens=_calc_tokens,
                                            extract_main_theme=_extract_main_theme,
                                            extract_named_entities=_extract_named_entities)
                print(f"{prtime()}: Tool output: context={context}")
                tracebuf.append(f"{prtime()}: Tool output: context={context[:64]}...")
                tool_outputs += context
            elif tool.function.name == 'list_of_files_for_given_question':
                tool_arg_question = args_dict.get('question')
                num_files = int(args_dict.get('number_of_files')) if args_dict.get('number_of_files') else 10
                tool_output = get_filelist_using_retr_and_rerank(faiss_rms, documents_list, index_map_list, index_type, tracebuf,
                                            filekey_to_file_chunks_dict, chat_config, tool_arg_question, num_files, searchsubdir=searchsubdir)
                print(f"{prtime()}: Tool output: tool_output={tool_output}")
                tracebuf.append(f"{prtime()}: Tool output: tool_output={tool_output[:64]}...")
                tool_outputs += tool_output
    return tool_outputs

class ollama_run_usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

def chat_using_ollama_assistant(faiss_rms:List[faiss_rm.FaissRM], documents_list:List[Dict[str,str]],
                                    index_map_list:List[Tuple[str,str]], index_type, tracebuf:List[str],
                                    filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]],
                                    assistants_thread_id:str, chat_config:ChatConfiguration, messages,
                                    searchsubdir=None, toolprompts=None) -> Tuple[str, str]:
    """
    documents is a dict like {fileid: finfo}; 
    index_map is a list of tuples: [(fileid, paragraph_index)];  the index into this list corresponds to the index of the embedding vector in the faiss index 
    Returns the tuple (output, thread_id).  Returns (None, None) on failure.
    """
    print(f"chat_using_ollama_assistant: Entered. faiss_rms={faiss_rms}")
    if toolprompts:
        tools = toolprompts
        print(f"chat_using_ollama_assistant: toolprompts specified as {tools}")
    else:
        tools = [TOOL_SEARCH_QUESTION_IN_DB] #, TOOL_LIST_OF_FILES_FOR_GIVEN_QUESTION]
        print(f"chat_using_ollama_assistant: toolprompts not specified. Using default of {tools}")
    ollama_messages = []
    for msg in messages:
        ollama_messages.append({'role': msg['role'], 'content': msg['content']})
    response = ollama.chat(model=ASSISTANTS_MODEL,
            messages = ollama_messages,
            tools = tools
            )
    #print(response)
    if response.message and response.message.tool_calls:
        tool_calls = response.message.tool_calls
        for tool in tool_calls:
            to = _process_tool_call(tools, tool,
                                faiss_rms, documents_list, index_map_list, index_type,
                                tracebuf, filekey_to_file_chunks_dict, chat_config,
                                searchsubdir)
            ollama_messages.append({'role': 'tool', 'content': to})
    response = ollama.chat(model=ASSISTANTS_MODEL,
            messages = ollama_messages,
            tools = tools
            )
    print(response)
    if response.message and response.message.content:
        return response.message.content, "notused", ollama_run_usage(response.prompt_eval_count, response.eval_count)
    else:
        return None, None, None
