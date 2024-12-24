import io
import json
import os
import sys
import traceback
import tempfile
import uuid
import time
import base64
import enum
import zlib
import datetime
from urllib.parse import unquote
import numpy as np
import faiss_rm
from typing import Tuple, List, Dict, Any, Self
from chatconfig import ChatConfiguration, RetrieverStrategyEnum
from documentchunk import DocumentType, DocumentChunkDetails, DocumentChunkRange
import tiktoken
from utils import prtime
import openai
import openai.types
import openai.types.beta
import openai.types.beta.threads
import openai.types.beta.threads.message
import openai.types.beta.threads.run
import openai.pagination
from openai import OpenAI
from openai import AssistantEventHandler
from text_utils import format_paragraph
from sentence_transformers.cross_encoder import CrossEncoder
from yoja_retrieve import get_context

#ASSISTANTS_MODEL="gpt-4"
ASSISTANTS_MODEL="gpt-4-1106-preview"

encoding_model=tiktoken.encoding_for_model(ASSISTANTS_MODEL)

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

def _lg(tracebuf, lgstr):
    print(lgstr)
    tracebuf.append(lgstr)

def _calc_tokens(context):
    global encoding_model
    return len(encoding_model.encode(context))

def _extract_named_entities(text):
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": 
f"Extract any named entities present in the sentence. Return a JSON object in this format: {{ 'entities': ['entity1', 'entity2'] }}, without any additional text or explanation. Particularly, do not include text before or after the parseable JSON: {text}"}
        ]
    )
    content = completion.choices[0].message.content
    print(f"_extract_named_entities: chatgpt returned content {content}")
    try:
        js = json.loads(content)
    except Exception as ex:
        print(f"_extract_named_entities: caught {ex}")
        return None
    print(f"_extract_named_entities: js= {js}")
    if 'entities' in js and js['entities']:
        return js['entities']
    else:
        return None

def _extract_main_theme(text):
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"Extract the main topic as a single word in the following sentence and return the result as a single word: {text}"}
        ]
    )
    retval = completion.choices[0].message.content
    print(f"_extract_main_theme: chatgpt returned {retval}")
    return retval

def _get_filelist_using_retr_and_rerank(faiss_rms:List[faiss_rm.FaissRM], documents_list:List[Dict[str,str]], index_map_list:List[Tuple[str,str]],
                                         index_type, tracebuf:List[str], filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]],
                                         chat_config:ChatConfiguration, last_msg:str, number_of_files:int = 10, searchsubdir:str=None):
    cross_sorted_scores:List[DocumentChunkDetails]
    status, cross_sorted_scores = _retrieve_rerank(faiss_rms, documents_list, index_map_list, index_type, tracebuf,
                            filekey_to_file_chunks_dict, chat_config, last_msg, searchsubdir)
    files_dict:Dict[str,DocumentChunkDetails] = {}
    for chunk_det in cross_sorted_scores:
        if not files_dict.get(chunk_det._get_file_key()):
            files_dict[chunk_det._get_file_key()] = chunk_det
            msg = f"{prtime()}: adding {chunk_det.file_path}{chunk_det.file_name} to listing"
            print(msg)
            tracebuf.append(msg)
        if len(files_dict) >= number_of_files: break
    return ",".join([  f"[{val.file_path}/{val.file_name}]({val.file_path}/{val.file_name})" for key, val in files_dict.items() ])

def retrieve_using_openai_assistant(faiss_rms:List[faiss_rm.FaissRM], documents_list:List[Dict[str,str]],
                                    index_map_list:List[Tuple[str,str]], index_type, tracebuf:List[str],
                                    filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]],
                                    assistants_thread_id:str, chat_config:ChatConfiguration, last_msg:str,
                                    searchsubdir=None, toolprompts=None) -> Tuple[str, str]:
    """
    documents is a dict like {fileid: finfo}; 
    index_map is a list of tuples: [(fileid, paragraph_index)];  the index into this list corresponds to the index of the embedding vector in the faiss index 
    Returns the tuple (output, thread_id).  REturns (None, NOne) on failure.
    """
    client = OpenAI()
 
    if toolprompts:
        tools = toolprompts
        print(f"retrieve_using_openai_assistant: toolprompts specified as {tools}")
    else:
        tools = [TOOL_SEARCH_QUESTION_IN_DB, TOOL_SEARCH_QUESTION_IN_DB_RETURN_MORE, TOOL_LIST_OF_FILES_FOR_GIVEN_QUESTION]
        print(f"retrieve_using_openai_assistant: toolprompts not specified. Using default of {tools}")
    assistant = client.beta.assistants.create(
        # Added 'or provide details of the mentioned subject." since openai was not
        # calling our function for a bare chat line such as 'android crypto policy' instead
        # of a full instruction such as 'give me details of android crypto policy'
        instructions="You are a helpful assistant. Use the provided functions to access confidential and private information and answer questions or provide details of the mentioned subject. Fix any spelling errors before calling the tools provided.",
        # BadRequestError: Error code: 400 - {'error': {'message': "The requested model 'gpt-4o' cannot be used with the Assistants API in v1. Follow the migration guide to upgrade to v2: https://platform.openai.com/docs/assistants/migration.", 'type': 'invalid_request_error', 'param': 'model', 'code': 'unsupported_model'}}
        model=ASSISTANTS_MODEL,
        tools=tools
    )

    if not assistants_thread_id:
        thread:openai.types.beta.Thread = client.beta.threads.create()
        assistants_thread_id = thread.id
        
    message:openai.types.beta.threads.Message = client.beta.threads.messages.create(
        thread_id=assistants_thread_id,
        role="user",
        content=last_msg,
    )
    print(f"{prtime()}: Adding message to openai thread and running the thread: message={message}")
    logmsgs = [f"{prtime()}: Adding message to openai thread and running the thread:"]
    logmsgs.append(f"role= {message.role}")
    if message.content:
        for c1 in message.content:
            logmsgs.append(f"message_content= {c1.text.value}")
    tracebuf.extend(logmsgs)

    # runs.create_and_poll sometimes fails with run.status==failed
    # and run.last_error=LastError(code='server_error', message='Sorry, something went wrong.')
    retries = 0
    while True:
        run:openai.types.beta.threads.Run = client.beta.threads.runs.create_and_poll(
            thread_id=assistants_thread_id,
            assistant_id=assistant.id,
        )
 
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(
                thread_id=assistants_thread_id
            )
            print(f"{prtime()}: run completed: run={run}")
            logmsgs = [f"{prtime()}: run completed:"]
            for msg in messages:
                logmsgs.append(f"{msg.content[0].text.value[:64]}...")
            tracebuf.extend(logmsgs)
            message = next(iter(messages))
            return message.content[0].text.value, assistants_thread_id, run.usage
        elif run.status == 'failed':
            seconds = 2**retries
            logmsg = f"{prtime()}: retrieve_using_openai_assistant: run.status failed. last_error={run.last_error}. sleeping {seconds} seconds and retrying"
            print(logmsg); tracebuf.append(logmsg)
            time.sleep(seconds)
            retries += 1
            if retries >= 5:
                return None, None, None
        # run.status='requires_action'
        else:
            print(f"{prtime()}: run incomplete: run result after running thread with above messages: run={run}")
            logmsgs = [f"{prtime()}: run incomplete: run result after running thread with above messages"]
            logmsgs.append(f"**instructions=** {run.instructions}")
            if run.required_action and run.required_action.submit_tool_outputs and run.required_action.submit_tool_outputs.tool_calls:
                for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                    logmsgs.append(f"**Tool name=** {tool_call.function.name}, **Tool Arguments=** {tool_call.function.arguments}")
            tracebuf.extend(logmsgs)
            break

    loopcnt:int = 0
    while loopcnt < 5:
        loopcnt += 1

        if run.status == 'completed':
            # messages=SyncCursorPage[Message](data=[
                # Message(id='msg_uwg..', assistant_id='asst_M5wN...', attachments=[], completed_at=None, content=[TextContentBlock(text=Text(annotations=[], value='Here are two bean-based recipes ...!'), type='text')], created_at=1715949656, incomplete_at=None, incomplete_details=None, metadata={}, object='thread.message', role='assistant', run_id='run_w32ljc..', status=None, thread_id='thread_z2KDBGP...'), 
                # Message(id='msg_V8Gf0S...', assistant_id=None, attachments=[], completed_at=None, content=[TextContentBlock(text=Text(annotations=[], value='Can you give me some recipes that involve beans?'), type='text')], created_at=1715949625, incomplete_at=None, incomplete_details=None, metadata={}, object='thread.message', role='user', run_id=None, status=None, thread_id='thread_z2KDBGPNy....')], object='list', first_id='msg_uwgAz...', last_id='msg_V8Gf0...', has_more=False)
            messages:openai.pagination.SyncCursorPage[openai.types.beta.threads.Message] = client.beta.threads.messages.list(thread_id=assistants_thread_id)
            print(f"{prtime()}: run completed: run={run}")
            if run.usage:
                logmsgs = [f"{prtime()}: run completed: completion_tokens={run.usage.completion_tokens}, prompt_tokens={run.usage.prompt_tokens}, total_tokens={run.usage.total_tokens}"]
            else:
                logmsgs = [f"{prtime()}: run completed:"]
            for msg in messages:
                logmsgs.append(f"{msg.content[0].text.value[:64]}...")
            tracebuf.extend(logmsgs)
            message = next(iter(messages))
            return message.content[0].text.value, assistants_thread_id, run.usage
        elif run.status == 'failed':
            seconds = 2**loopcnt
            logmsg = f"{prtime()}: retrieve_using_openai_assistant: tool processing. run.status failed. last_error={run.last_error}. sleeping {seconds} seconds"
            print(logmsg); tracebuf.append(logmsg)
            continue
        else: # run is incomplete
            print(f"{prtime()}: run incomplete: result after running thread with above messages={run}")
            tracebuf.append(f"{prtime()}: run incomplete:")
    
            # Define the list to store tool outputs
            tool_outputs = []; tool:openai.types.beta.threads.RequiredActionFunctionToolCall
            # Loop through each tool in the required action section        
            for tool in run.required_action.submit_tool_outputs.tool_calls:
                # function=Function(arguments='{\n  "question": "bean recipes"\n}', name='search_question_in_db'), type='function')
                args_dict:dict = json.loads(tool.function.arguments)
                print(f"{prtime()}: Running tool {tool}")
                if tool.function.name == "search_question_in_db" or tool.function.name == 'search_question_in_db.controls':
                    tool_arg_question = args_dict.get('question')
                    context:str = get_context(faiss_rms, documents_list, index_map_list, index_type, tracebuf,
                                                filekey_to_file_chunks_dict, chat_config, tool_arg_question,
                                                True, False, searchsubdir=searchsubdir, calc_tokens=_calc_tokens,
                                                extract_main_theme=_extract_main_theme, extract_named_entities=_extract_named_entities)
                    print(f"{prtime()}: Tool output: context={context}")
                    tracebuf.append(f"{prtime()}: Tool output: context={context[:64]}...")
                    tool_outputs.append({
                        "tool_call_id": tool.id,
                        "output": context
                    })
                elif tool.function.name == "search_question_in_db_return_more" or tool.function.name == 'search_question_in_db_return_more.controls':
                    tool_arg_question = args_dict.get('question')
                    context:str = get_context(faiss_rms, documents_list, index_map_list, index_type, tracebuf,
                                                filekey_to_file_chunks_dict, chat_config, tool_arg_question,
                                                False, True, searchsubdir=searchsubdir, calc_tokens=_calc_tokens,
                                                extract_main_theme=_extract_main_theme, extract_named_entities=_extract_named_entities)
                    print(f"{prtime()}: Tool output: context={context}")
                    tracebuf.append(f"{prtime()}: Tool output: context={context[:64]}...")
                    tool_outputs.append({
                        "tool_call_id": tool.id,
                        "output": context
                    })
                elif tool.function.name == "list_of_files_for_given_question" or tool.function.name == 'list_of_files_for_given_question.controls':
                    args_dict:dict = json.loads(tool.function.arguments)
                    tool_arg_question = args_dict.get('question')
                    num_files = int(args_dict.get('number_of_files')) if args_dict.get('number_of_files') else 10
                    tool_output = _get_filelist_using_retr_and_rerank(faiss_rms, documents_list, index_map_list, index_type, tracebuf,
                                                filekey_to_file_chunks_dict, chat_config, tool_arg_question, num_files, searchsubdir=searchsubdir)
                    print(f"{prtime()}: Tool output: tool_output={tool_output}")
                    tracebuf.append(f"{prtime()}: Tool output: tool_output={tool_output[:64]}...")
                    tool_outputs.append({"tool_call_id": tool.id, "output": tool_output })
                else:
                    raise Exception(f"**Unknown function call:** {tool.function.name}")
  
            # Submit all tool outputs at once after collecting them in a list
            if tool_outputs:
                try:
                    print(f"{prtime()}: calling submit_tool_outputs_and_poll: run={run}.")
                    tracebuf.append(f"{prtime()}: calling submit_tool_outputs_and_poll:")
                    run = client.beta.threads.runs.submit_tool_outputs_and_poll(
                        thread_id=assistants_thread_id,
                        run_id=run.id,
                        poll_interval_ms=500,
                        tool_outputs=tool_outputs
                    )
                    print(f"{prtime()}: Return from submit_tool_outputs_and_poll: run={run}.")
                    tracebuf.append(f"{prtime()}: Return from submit_tool_outputs_and_poll:")
                except Exception as e:
                    print("Failed to submit tool outputs: ", e)
            else:
                logmsg = "{prtime()}: No tool outputs to submit."
                print(logmsg); tracebuf.append(logmsg)
    logmsg = f"{prtime()}: retrieve_using_openai_assistant: tool processing exited loop without reaching complete. returning None"
    print(logmsg); tracebuf.append(logmsg)
    return None, None, None


