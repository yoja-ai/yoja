import io
import json
import os
import traceback
import tempfile
import uuid
import time
import base64
import zlib
import datetime
from urllib.parse import unquote
import numpy as np
from utils import respond, get_service_conf, check_cookie, set_start_time
from index_utils import init_vdb, lock_sample_dir, unlock_sample_dir, create_sample_index
import boto3
from openai_ner import OpenAiNer
from openai import OpenAI
import traceback_with_variables
import cryptography.fernet 
from sentence_transformers.cross_encoder import CrossEncoder
# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from typing import Tuple, List, Dict, Any, Self
import faiss_rm
import re
import dataclasses
import enum
import copy
import jsons
import tiktoken
from searchsubdir import do_set_searchsubdir
from scripts.fetch_tool_prompts import fetch_tool_prompts

MAX_TOKEN_LIMIT=2048
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

def calc_tokens(context):
    global encoding_model
    return len(encoding_model.encode(context))

def _prtime():
    nw=datetime.datetime.now()
    return f"{nw.hour}:{nw.minute}:{nw.second}"

def _get_agent_thread_id(messages:List[dict]) -> str:
    for msg in messages:
        lines:List[str] = msg['content'].splitlines()
        # get the last line of the last 'message'
        l_line:str = lines[-1].strip()
        # Line format example: **Context Source: ....docx**\t<!-- ID=0B-qJbLgl53j..../0 -->; thread_id=<thread_id>
        match:re.Match = re.search("thread_id=([^\\s;]+)", l_line, re.IGNORECASE)
        if match:
            thread_id = match.group(1)
            print(f"Extracted thread_id={thread_id}")
            return thread_id
    return None

class DocumentType(enum.Enum):
    DOCX = 1
    PPTX = 2
    PDF = 3
    TXT = 4

    @classmethod
    def fromString(clz, doc_type_str:str):
        str_to_type_dict:Dict[str, Self] = { 'pptx':clz.PPTX, 'docx':clz.DOCX, 'pdf':clz.PDF, 'txt':clz.TXT }
        
        if not str_to_type_dict.get(doc_type_str):
            raise Exception(f"Unknown document type:  {doc_type_str}")
            
        return str_to_type_dict.get(doc_type_str)

    def generate_link(self, doc_storage_type:faiss_rm.DocStorageType, file_id:str) -> str:
        if doc_storage_type == faiss_rm.DocStorageType.GoogleDrive:
            # If word document (ends with doc or docx), use the link  https://docs.google.com/document/d/<file_id>
            # if a pdf file (ends with pdf), use the link https://drive.google.com/file/d/<file_id>
            # if pptx file (ends with ppt or pptx) https://docs.google.com/presentation/d/<file_id>
            doc_type_to_link:Dict[Self, str] = { self.__class__.DOCX:'https://docs.google.com/document/d/{file_id}', self.__class__.PPTX:'https://docs.google.com/presentation/d/{file_id}', self.__class__.PDF:'https://drive.google.com/file/d/{file_id}', self.__class__.TXT:'https://drive.google.com/file/d/{file_id}' }
            if not doc_type_to_link.get(self):
                raise Exception(f"generate_link(): Unknown document type:  self={self}; doc_storage_type={doc_storage_type}; file_id={file_id}")
            return doc_type_to_link[self].format(file_id=file_id)
        elif doc_storage_type == faiss_rm.DocStorageType.DropBox:
            doc_type_to_link:Dict[Self, str] = { self.__class__.DOCX:'https://docs.google.com/document/d/{file_id}', self.__class__.PPTX:'https://docs.google.com/presentation/d/{file_id}', self.__class__.PDF:'https://drive.google.com/file/d/{file_id}', self.__class__.TXT:'https://drive.google.com/file/d/{file_id}' }
            if not doc_type_to_link.get(self):
                raise Exception(f"generate_link(): Unknown document type:  self={self}; doc_storage_type={doc_storage_type}; file_id={file_id}")
            return doc_type_to_link[self].format(file_id=file_id)
        else:
            raise Exception(f"generate_link(): Unknown document type:  self={self}; doc_storage_type={doc_storage_type}; file_id={file_id}")
    
class RetrieverStrategyEnum(enum.Enum):
    FullDocStrategy = 1
    PreAndPostChunkStrategy = 2
    
@dataclasses.dataclass
class ChatConfiguration:
    print_trace:bool
    use_ivfadc:bool
    cross_encoder_10:bool
    use_ner:bool
    print_trace_context_choice:bool
    file_details:bool
    retreiver_strategy:RetrieverStrategyEnum
    dbg_set_searchsubdir:bool

@dataclasses.dataclass
class DocumentChunkDetails:
    index_in_faiss:int
    faiss_rm_vdb:faiss_rm.FaissRM
    faiss_rm_vdb_id:int   # currently the index of the vdb in a list of VDBs.  TODO: we need to store this ID in the DB or maintain an ordered list in the DB or similar
    doc_storage_type:faiss_rm.DocStorageType
    distance:float           # similarity score from vdb
    file_type:DocumentType 
    file_path:str = None
    file_name:str = None
    file_id:str = None
    file_info:dict = dataclasses.field(default=None, repr=False)    # finfo or details about the file
    para_id:int = None       # the paragraph number
    para_dict:dict = dataclasses.field(default=None, repr=False) # details of this paragraph
    para_text_formatted:str = dataclasses.field(default=None, repr=False)
    cross_encoder_score:float = None
    retr_sorted_idx:int = None # the position of this chunk when sorted by the retriever
    reranker_sorted_idx:int = None # the position of this chunk when sorted by the reranker
    
    def _get_file_key(self):
        return f"{self.faiss_rm_vdb_id}/{self.file_id}"

@dataclasses.dataclass
class DocumentChunkRange:
    """ end_para_id is inclusive """
    doc_chunk:DocumentChunkDetails
    start_para_id:int = None
    end_para_id:int = None
    """ end_para_id is inclusive """
    chunk_overlap_processed = False

    def _get_file_key(self):
        return self.doc_chunk._get_file_key()
    
    def isChunkRangeInsideRangeList(self, chunk_range_list:List[Self]):
        for chunk_range in chunk_range_list:
            if chunk_range._get_file_key() == self._get_file_key() and chunk_range.start_para_id <= self.start_para_id and chunk_range.end_para_id >= self.end_para_id:
                return chunk_range
        
        return None
    
    @classmethod
    def _get_merged_chunk_ranges(clz, context_chunk_range_list:List[Self]) -> List[Self]:
        merged_chunks:List[Self] = []
        for curr_chunk_range in context_chunk_range_list:
            # check if we have already created merged chunks for this file.
            curr_file_key:str = curr_chunk_range.doc_chunk._get_file_key()
            merged_chunks_curr_file = [ merged_chunk for merged_chunk in merged_chunks if merged_chunk.doc_chunk._get_file_key() == curr_file_key ]
            # if so, skip this chunk range.
            if len(merged_chunks_curr_file): continue
            
            all_chunks_curr_file:List[Self] = [ chunk_range for chunk_range in context_chunk_range_list if chunk_range.doc_chunk._get_file_key() == curr_file_key ]
            
            sorted_all_chunks_curr_file:List[Self] = sorted(all_chunks_curr_file, key= lambda elem_chunk_range: elem_chunk_range.start_para_id )
            
            curr_merged_chunk:Self = None
            for sorted_elem in sorted_all_chunks_curr_file:
                if not curr_merged_chunk: 
                    # we only do a shallow copy.  ok, since we only modify non reference data
                    curr_merged_chunk = copy.copy(sorted_elem)
                    continue
                
                # check if there is a overlap between chunks
                if curr_merged_chunk.end_para_id >= sorted_elem.start_para_id:
                    if curr_merged_chunk.end_para_id <= sorted_elem.end_para_id:
                        curr_merged_chunk.end_para_id = sorted_elem.end_para_id
                # there is no overlap
                else:  
                    merged_chunks.append(curr_merged_chunk)
                    curr_merged_chunk = copy.copy(sorted_elem)
            merged_chunks.append(curr_merged_chunk)
        
        return merged_chunks
            
    @classmethod
    # https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
    def process_overlaps(clz, context_chunk_range_list:List[Self]) -> List[Self]:
        merged_chunk_range_list = DocumentChunkRange._get_merged_chunk_ranges(context_chunk_range_list)

        out_chunk_range_list:List[Self] = []
        merged_chunk_range_list = DocumentChunkRange._get_merged_chunk_ranges(context_chunk_range_list)
        for chunk_range in context_chunk_range_list:
            merged_chunk_range = chunk_range.isChunkRangeInsideRangeList(merged_chunk_range_list)
            if not merged_chunk_range: 
                print(f"Raising exception: {chunk_range} not found in {merged_chunk_range_list}")
                raise Exception(f"Unable to find chunk range with vbd_id={chunk_range.doc_chunk.faiss_rm_vdb_id} file_name={chunk_range.doc_chunk.file_name} para_id={chunk_range.doc_chunk.para_id} start_para_id={chunk_range.start_para_id} end_para_id={chunk_range.end_para_id}")
            if merged_chunk_range.isChunkRangeInsideRangeList(out_chunk_range_list):
                continue
            else:
                out_chunk_range_list.append(merged_chunk_range)
                
        return out_chunk_range_list
        
def ongoing_chat(event, body, faiss_rms:List[faiss_rm.FaissRM], documents_list:List[Dict[str, dict]],
                    index_map_list:List[List[Tuple[str, str]]], index_type:str = 'flat', sample_source=None,
                    searchsubdir=None):
    """
    documents is a dict like {fileid: finfo}; 
    index_map is a list of tuples: [(fileid, paragraph_index)]; the index into this list corresponds to the index of the embedding vector in the faiss index
    """    
    print(f"ongoing_chat: entered")
    
    thread_id:str = _get_agent_thread_id(body['messages'])
    if not thread_id:
        emsg = "ongoing_chat: Error. Unable to extract thread_id from messages"
        print(emsg)
        return respond({"error_msg": emsg}, status=500)

    # get the user message, which is the last line
    last_msg:str = body['messages'][-1]['content']
    tracebuf = ['**Begin Trace**']; filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]] = {}; srp:str
    chat_config:ChatConfiguration; last_msg:str
    chat_config, last_msg = _debug_flags(last_msg, tracebuf)
    srp, thread_id, run_usage = retrieve_using_openai_assistant(faiss_rms,documents_list, index_map_list, index_type,
                        tracebuf, filekey_to_file_chunks_dict, thread_id, chat_config, last_msg)
    if not srp:
        return respond({"error_msg": "Error. retrieve using assistant failed"}, status=500)
    if run_usage:
        pct=(float(run_usage.prompt_tokens)/float(MAX_TOKEN_LIMIT))*100.0
        srp = srp +f"  \n**Tokens:** prompt={run_usage.prompt_tokens}({pct}% of {MAX_TOKEN_LIMIT}), completion={run_usage.completion_tokens}"
    
    context_srcs:List[str]; context_srcs_links:List[ContextSource]
    context_srcs, context_srcs_links = _generate_context_sources(filekey_to_file_chunks_dict)
    if chat_config.print_trace:
        tstr = ""
        for tt in tracebuf:
            tstr += f"  \n{tt}"
        srp = srp +tstr + f"  \n{';  '.join(context_srcs)}" + "<!-- ; thread_id=" + thread_id + " -->"
    else:
        srp = srp +f"  \n{';  '.join(context_srcs)}" + "<!-- ; thread_id=" + thread_id + " -->"
    print(f"ongoing_chat: srp={srp}")
        
    res = {}
    res['id'] = event['requestContext']['requestId']
    res['object'] = 'chat.completion.chunk'
    res['created'] = int(time.time_ns()/1000000)
    res['model'] = 'gpt-3.5-turbo-0125'
    res['choices'] = [
        {
            "index": 0,
            # "finish_reason": "stop",
            "delta": {
                "role": "assistant",
                "content": f"\n\n{srp}"
                },
            "context_sources": json.loads(jsons.dumps(context_srcs_links))
        }
    ]
    res_str = json.dumps(res)
    return {
        'statusCode': 200,
        'body': f"data:{res_str}",
        'headers': {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache, no-store, must-revalidate, private',
            'Pragma': 'no-cache',
            'Expires': '0'
        },
    }
    return respond(None, res=res)

def replace_tools_in_assistant(new_tools):
    global assistant
    assistant = client.beta.assistants.create(
        instructions="You are a helpful assistant. Use the provided functions to access confidential and private information and answer questions or provide details of the mentioned subject.",
        model=ASSISTANTS_MODEL,
        tools=new_tools
    )


def update_tool_prompts(tool_data: List[Dict[str, str]]):
    tools = []
    for row in tool_data:
        tool_name = row['Tool Name']
        description = row['Description']
        tools.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question for which passages need to be searched"
                        }
                    },
                    "required": ["question"]
                }
            }
        })
    # Replace the hardcoded tools with the dynamic tools
    replace_tools_in_assistant(tools)

def _debug_flags(query:str, tracebuf:List[str]) -> Tuple[ChatConfiguration, str]:
    """ returns the tuple (print_trace, use_ivfadc, cross_encoder_10, enable_NER)"""
    print_trace, use_ivfadc, cross_encoder_10, use_ner, file_details, print_trace_context_choice, retriever_stratgey, dbg_set_searchsubdir = \
                (False, False, False, False, False, False, RetrieverStrategyEnum.PreAndPostChunkStrategy, False)
    idx = 0
    for idx in range(len(query)):
        # '+': print_trace
        # '@': use ivfadc index
        # '#': print only 10 results from cross encoder.
        # '$': enable NER
        # '^': print_trace with info about choice of context
        # '!': print details of file
        c = query[idx]
        if c not in ['+','@','#','$', '^', '!', '/', '~']: break
        
        if c == '+': print_trace = True
        if c == '@': use_ivfadc = True
        if c == '#': cross_encoder_10 = True
        if c == '$': use_ner = True
        if c == '^': print_trace_context_choice = True
        if c == '!': file_details = True
        if c == '/': retriever_stratgey = RetrieverStrategyEnum.FullDocStrategy
        if c == '~': dbg_set_searchsubdir = True
    
    # strip the debug flags from the question
    if idx == len(query) - 1:
        last_msg = ""
    else:
        last_msg = query[idx:]
    chat_config = ChatConfiguration(print_trace, use_ivfadc, cross_encoder_10, use_ner, print_trace_context_choice, file_details, retriever_stratgey, dbg_set_searchsubdir)
    logmsg = f"**{_prtime()}: Debug Flags**: chat_config={chat_config}, last_={last_msg}"
    print(logmsg)

    return (chat_config, last_msg)

def retrieve_using_openai_assistant(faiss_rms:List[faiss_rm.FaissRM], documents_list:List[Dict[str,str]],
                                    index_map_list:List[Tuple[str,str]], index_type, tracebuf:List[str],
                                    filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]],
                                    assistants_thread_id:str, chat_config:ChatConfiguration, last_msg:str,
                                    searchsubdir=None) -> Tuple[str, str]:
    """
    documents is a dict like {fileid: finfo}; 
    index_map is a list of tuples: [(fileid, paragraph_index)];  the index into this list corresponds to the index of the embedding vector in the faiss index 
    Returns the tuple (output, thread_id).  REturns (None, NOne) on failure.
    """
    import openai
    import openai.types
    import openai.types.beta
    import openai.types.beta.threads
    import openai.types.beta.threads.message
    import openai.types.beta.threads.run
    import openai.pagination
    from openai import OpenAI
    from openai import AssistantEventHandler
    client = OpenAI()
 
    assistant = client.beta.assistants.create(
        # Added 'or provide details of the mentioned subject." since openai was not
        # calling our function for a bare chat line such as 'android crypto policy' instead
        # of a full instruction such as 'give me details of android crypto policy'
        instructions="You are a helpful assistant. Use the provided functions to access confidential and private information and answer questions or provide details of the mentioned subject.",
        # BadRequestError: Error code: 400 - {'error': {'message': "The requested model 'gpt-4o' cannot be used with the Assistants API in v1. Follow the migration guide to upgrade to v2: https://platform.openai.com/docs/assistants/migration.", 'type': 'invalid_request_error', 'param': 'model', 'code': 'unsupported_model'}}
        model=ASSISTANTS_MODEL,
        tools=[TOOL_SEARCH_QUESTION_IN_DB, TOOL_SEARCH_QUESTION_IN_DB_RETURN_MORE]
    )

    if not assistants_thread_id:
        thread:openai.types.beta.Thread = client.beta.threads.create()
        assistants_thread_id = thread.id
        
    message:openai.types.beta.threads.Message = client.beta.threads.messages.create(
        thread_id=assistants_thread_id,
        role="user",
        content=last_msg,
    )
    print(f"**{_prtime()}: Adding message to openai thread and running the thread:** message={message}")
    logmsgs = [f"**{_prtime()}: Adding message to openai thread and running the thread:**"]
    logmsgs.append(f"**role=** {message.role}")
    if message.content:
        for c1 in message.content:
            logmsgs.append(f"**message_content=** {c1.text.value}")
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
            print(f"**{_prtime()}: run completed:**run={run}")
            logmsgs = [f"**{_prtime()}: run completed:**"]
            for msg in messages:
                logmsgs.append(f"{msg.content[0].text.value[:64]}...")
            tracebuf.extend(logmsgs)
            message = next(iter(messages))
            return message.content[0].text.value, assistants_thread_id, run.usage
        elif run.status == 'failed':
            seconds = 2**retries
            logmsg = f"**{_prtime()}: retrieve_using_openai_assistant:** run.status failed. last_error={run.last_error}. sleeping {seconds} seconds and retrying"
            print(logmsg); tracebuf.append(logmsg)
            time.sleep(seconds)
            retries += 1
            if retries >= 5:
                return None, None, None
        # run.status='requires_action'
        else:
            print(f"**{_prtime()}: run incomplete:** run result after running thread with above messages: run={run}")
            logmsgs = [f"**{_prtime()}: run incomplete:** run result after running thread with above messages"]
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
            print(f"**{_prtime()}: run completed:**run={run}")
            if run.usage:
                logmsgs = [f"**{_prtime()}: run completed:** completion_tokens={run.usage.completion_tokens}, prompt_tokens={run.usage.prompt_tokens}, total_tokens={run.usage.total_tokens}"]
            else:
                logmsgs = [f"**{_prtime()}: run completed:**"]
            for msg in messages:
                logmsgs.append(f"{msg.content[0].text.value[:64]}...")
            tracebuf.extend(logmsgs)
            message = next(iter(messages))
            return message.content[0].text.value, assistants_thread_id, run.usage
        elif run.status == 'failed':
            seconds = 2**loopcnt
            logmsg = f"**{_prtime()}: retrieve_using_openai_assistant:** tool processing. run.status failed. last_error={run.last_error}. sleeping {seconds} seconds"
            print(logmsg); tracebuf.append(logmsg)
            continue
        else: # run is incomplete
            print(f"**{_prtime()}: run incomplete:** result after running thread with above messages={run}")
            tracebuf.append(f"**{_prtime()}: run incomplete:**")
    
            # Define the list to store tool outputs
            tool_outputs = []; tool:openai.types.beta.threads.RequiredActionFunctionToolCall
            # Loop through each tool in the required action section        
            for tool in run.required_action.submit_tool_outputs.tool_calls:
                # function=Function(arguments='{\n  "question": "bean recipes"\n}', name='search_question_in_db'), type='function')
                args_dict:dict = json.loads(tool.function.arguments)
                print(f"**{_prtime()}: Running tool {tool}")
                if tool.function.name == "search_question_in_db" or tool.function.name == 'search_question_in_db.controls':
                    tool_arg_question = args_dict.get('question')
                    context:str = _get_context_using_retr_and_rerank(faiss_rms, documents_list, index_map_list, index_type, tracebuf,
                                                filekey_to_file_chunks_dict, chat_config, tool_arg_question, True, False, searchsubdir=searchsubdir)
                    print(f"**{_prtime()}: Tool output:** context={context}")
                    tracebuf.append(f"**{_prtime()}: Tool output:** context={context[:64]}...")
                    tool_outputs.append({
                        "tool_call_id": tool.id,
                        "output": context
                    })
                elif tool.function.name == "search_question_in_db_return_more" or tool.function.name == 'search_question_in_db_return_more.controls':
                    tool_arg_question = args_dict.get('question')
                    context:str = _get_context_using_retr_and_rerank(faiss_rms, documents_list, index_map_list, index_type, tracebuf,
                                                filekey_to_file_chunks_dict, chat_config, tool_arg_question, False, True, searchsubdir=searchsubdir)
                    print(f"**{_prtime()}: Tool output:** context={context}")
                    tracebuf.append(f"**{_prtime()}: Tool output:** context={context[:64]}...")
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
                    print(f"**{_prtime()}: Tool output:** tool_output={tool_output}")
                    tracebuf.append(f"**{_prtime()}: Tool output:** tool_output={tool_output[:64]}...")
                    tool_outputs.append({"tool_call_id": tool.id, "output": tool_output })
                else:
                    raise Exception(f"**Unknown function call:** {tool.function.name}")
  
            # Submit all tool outputs at once after collecting them in a list
            if tool_outputs:
                try:
                    print(f"**{_prtime()}: calling submit_tool_outputs_and_poll:** run={run}.")
                    tracebuf.append(f"**{_prtime()}: calling submit_tool_outputs_and_poll:**")
                    run = client.beta.threads.runs.submit_tool_outputs_and_poll(
                        thread_id=assistants_thread_id,
                        run_id=run.id,
                        poll_interval_ms=500,
                        tool_outputs=tool_outputs
                    )
                    print(f"**{_prtime()}: Return from submit_tool_outputs_and_poll:** run={run}.")
                    tracebuf.append(f"**{_prtime()}: Return from submit_tool_outputs_and_poll:**")
                except Exception as e:
                    print("Failed to submit tool outputs: ", e)
            else:
                logmsg = "**{_prtime()}: No tool outputs to submit.**"
                print(logmsg); tracebuf.append(logmsg)
    logmsg = f"**{_prtime()}: retrieve_using_openai_assistant:** tool processing exited loop without reaching complete. returning None"
    print(logmsg); tracebuf.append(logmsg)
    return None, None, None

def _gen_context(context_chunk_range_list:List[DocumentChunkRange], handle_overlaps:bool = True) -> Tuple[dict, str]:
    """returns the tupe (error_dict, context)"""
    
    print(f"_gen_context(): context_chunk_range_list={context_chunk_range_list}")
    if handle_overlaps: 
        context_chunk_range_list = DocumentChunkRange.process_overlaps(context_chunk_range_list)
        print(f"_gen_context(): context_chunk_range_list after overlapping merge={context_chunk_range_list}")
    
    # generate the context
    new_context:str = ''    
    for chunk_range in context_chunk_range_list:
        chunk_det = chunk_range.doc_chunk
        fparagraphs = []
        finfo = chunk_range.doc_chunk.file_info
        if 'slides' in finfo:
            key = 'slides'
        elif 'paragraphs' in finfo:
            key = 'paragraphs'
        else:
            emsg = f"ERROR! Could not get key in document for {finfo}"
            print(emsg)
            return respond({"error_msg": emsg}, status=500), None
        for idx in range(chunk_range.start_para_id, chunk_range.end_para_id+1):
            formatted_para:str = chunk_range.doc_chunk.faiss_rm_vdb.format_paragraph(finfo[key][idx])
            fparagraphs.append(formatted_para)
            print(f"Context: Included chunk from file_name={chunk_det.file_name} para_id={idx} faiss_rm_vdb_id={chunk_det.faiss_rm_vdb_id}")
        prelude = f"Name of the file is {chunk_det.file_name}"
        new_context = new_context + "\n" + prelude + "\n" + ". ".join(fparagraphs)
        
    return None, new_context

def retrieve_and_rerank_using_faiss(faiss_rms:List[faiss_rm.FaissRM], documents_list:List[Dict[str,str]], index_map_list:List[Tuple[str,str]],
                                    index_type, tracebuf:List[str], filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]],
                                    chat_config:ChatConfiguration, last_msg:str, searchsubdir:str=None) -> Tuple[np.ndarray, List[DocumentChunkDetails]]:
    """
    documents is a dict like {fileid: finfo}; 
    index_map is a list of tuples: [(fileid, paragraph_index)];  the index into this list corresponds to the index of the embedding vector in the faiss index
    
    returns the reranked indices (index into list of documentChunkDetails) and the list of DocumentChunkDetails  """
    use_ivfadc:bool; cross_encoder_10:bool; use_ner:bool; print_trace_context_choice:bool; retreiver_strategy:RetrieverStrategyEnum
    use_ivfadc, cross_encoder_10, use_ner, print_trace_context_choice, retreiver_strategy = ( chat_config.use_ivfadc, chat_config.cross_encoder_10, chat_config.use_ner, chat_config.print_trace_context_choice, chat_config.retreiver_strategy)
    
    if use_ner:
        openai_ner = OpenAiNer()
        ner_result = openai_ner(last_msg)
        print(f"entities={ner_result}")
        ner_as_conversation =  openai_ner.convert_to_conversation(ner_result)
        queries = [last_msg] + ner_as_conversation
    else:
        queries = [last_msg]
    print(f"queries after ner={queries}")
    tracebuf.append(f"**{_prtime()} Tool call:search_question_in_db: Entered. Queries:**")
    tracebuf.extend(queries)

    sorted_summed_scores:List[DocumentChunkDetails] = []
    for i in range(len(faiss_rms)):
        faiss_rm_vdb = faiss_rms[i]
        documents = documents_list[i]
        index_map = index_map_list[i]
        
        # dict of { index_in_faiss:[distance1, distance2] }
        passage_scores_dict:Dict[int, List] = {}
        for qind in range(len(queries)):
            qr = queries[qind]
            distances, indices_in_faiss = faiss_rm_vdb(qr, k=128, index_type='ivfadc' if use_ivfadc else 'flat' )
            for idx in range(len(indices_in_faiss[0])):
                ind_in_faiss = indices_in_faiss[0][idx]
                finfo = documents[index_map[ind_in_faiss][0]]
                # if searchsubdir is specified, then take just the entries that exist in searchsubdir or its subdirectories
                if not searchsubdir or finfo['path'].startswith(searchsubdir):
                    print(f"retrieve_and_rerank_using_faiss: searchsubdir={searchsubdir}. Accepting {finfo['filename']} path={finfo['path']}")
                    # the first query in queries[] is the actual user chat text. we give that twice the weight
                    dist = distances[0][idx] if qind == 0 else distances[0][idx]/2.0
                    if ind_in_faiss in passage_scores_dict:
                        passage_scores_dict[ind_in_faiss].append(dist)
                    else:
                        passage_scores_dict[ind_in_faiss] = [dist]
                else:
                    print(f"retrieve_and_rerank_using_faiss: searchsubdir={searchsubdir}. Rejecting {finfo['filename']} path={finfo['path']}")
        print(f"retrieve_and_rerank_using_faiss: passage_scores=")
        if print_trace_context_choice:
            tracebuf.append(f"**{_prtime()}: Passage Scores**")
        for ind_in_faiss, ps in passage_scores_dict.items():
            print(f"    index_in_faiss={ind_in_faiss}, file={documents[index_map[ind_in_faiss][0]]['filename']}, paragraph_num={index_map[ind_in_faiss][1]}, passage_score={ps}")
            if print_trace_context_choice:
                tracebuf.append(f"file={documents[index_map[ind_in_faiss][0]]['filename']}, paragraph_num={index_map[ind_in_faiss][1]}, ps={ps}")

        # faiss returns METRIC_INNER_PRODUCT - larger number means better match
        # sum the passage scores

        summed_scores:List[DocumentChunkDetails] = [] # array of (summed_score, index_in_faiss)
        for index_in_faiss, scores in passage_scores_dict.items():
            summed_scores.append(DocumentChunkDetails(index_in_faiss, faiss_rm_vdb, i, faiss_rm_vdb.get_doc_storage_type(), sum(scores), 
                                                      DocumentType.fromString(documents[index_map[index_in_faiss][0]]['filetype']),
                                                      documents[index_map[index_in_faiss][0]].get('path') if documents[index_map[index_in_faiss][0]].get('path') else None,
                                                      documents[index_map[index_in_faiss][0]]['filename'],
                                                      documents[index_map[index_in_faiss][0]]['fileid'],
                                                      documents[index_map[index_in_faiss][0]],
                                                      index_map[index_in_faiss][1]))
        if use_ner:
            print(f"retrieve_and_rerank_using_faiss: summed_scores:")
            if print_trace_context_choice:
                tracebuf.append(f"**{_prtime()}: Summed Scores**")
            for chunk_det in summed_scores:
                print(f"    index_in_faiss={chunk_det.index_in_faiss}, score={chunk_det.distance}, file={chunk_det.file_name}, paragraph_num={chunk_det.para_id}")
                if print_trace_context_choice:
                    tracebuf.append(f"file={chunk_det.file_name}, paragraph_num={chunk_det.para_id}, score={chunk_det.distance}")
        sorted_summed_scores.extend( summed_scores )
    
    sorted_summed_scores = sorted(sorted_summed_scores, key=lambda x: x.distance, reverse=True)
    if use_ner:
        print(f"retrieve_and_rerank_using_faiss: sorted_summed_scores:")
        if print_trace_context_choice:
            tracebuf.append(f"**{_prtime()}: Sorted Summed Scores**")
        for chunk_det in sorted_summed_scores:
            print(f"    index_in_faiss={chunk_det.index_in_faiss}, score={chunk_det.distance}, file={chunk_det.file_name}, paragraph_num={chunk_det.para_id}")
            if print_trace_context_choice:
                tracebuf.append(f"file={chunk_det.file_name}, paragraph_num={chunk_det.para_id}, score={chunk_det.distance}")

    # Note that these three arrays are aligned: using the same index in these 3 arrays retrieves corresponding elements: reranker_map (array of faiss_indexes), reranker_input (array of (query, formatted para)) and cross_scores (array of cross encoder scores)
    reranker_map = [] # array of index_in_faiss
    reranker_input = [] # array of (query, formatted_para)
    for idx in range(min(len(sorted_summed_scores), 128)):
        curr_chunk:DocumentChunkDetails = sorted_summed_scores[idx]
        curr_chunk.retr_sorted_idx = idx
        index_in_faiss = curr_chunk.index_in_faiss
        curr_chunk.para_dict = curr_chunk.faiss_rm_vdb.get_paragraph(index_in_faiss)
        # force an empty formatted_paragraph from format_paragraph() below, by using an empty dict
        if not curr_chunk.para_dict: curr_chunk.para_dict = {}
        curr_chunk.para_text_formatted = f"Name of the file is {curr_chunk.file_name}\n" + curr_chunk.faiss_rm_vdb.format_paragraph(curr_chunk.para_dict)
        reranker_input.append([last_msg, curr_chunk.para_text_formatted])

    global g_cross_encoder
    # https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-2-v2
    if not g_cross_encoder: g_cross_encoder = CrossEncoder('/var/task/cross-encoder/ms-marco-MiniLM-L-6-v2') if os.path.isdir('/var/task/cross-encoder/ms-marco-MiniLM-L-6-v2') else CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    # retrieve_and_rerank_using_faiss: cross_scores=[-10.700319  -11.405142   -3.5650876  -8.041701   -9.972779   -9.609493 -10.653023   -6.8494396  -7.601103  -11.405787  -10.690331  -10.050377 ...
    # Note that these three arrays are aligned: using the same index in these 3 arrays retrieves corresponding elements: reranker_map (array of faiss_indexes), reranker_input (array of (query, formatted para)) and cross_scores (array of cross encoder scores)
    # 
    # Negative Scores for cross-encoder/ms-marco-MiniLM-L-6-v2 #1058: https://github.com/UKPLab/sentence-transformers/issues/1058
    cross_scores:np.ndarray = g_cross_encoder.predict(reranker_input)
    print(f"retrieve_and_rerank_using_faiss: cross_scores={cross_scores}")
    # Returns the indices into the given cross_scores array, that would sort the given cross_scores array.
    # Perform an indirect sort along the given axis using the algorithm specified by the kind keyword. It returns an array of indices of the same shape as a that index data along the given axis in sorted order.
    reranked_indices = np.argsort(cross_scores)[::-1]
    print(f"retrieve_and_rerank_using_faiss: reranked_indices={reranked_indices}")
    reranker_sorted_idx:int = 0
    for retr_sorted_idx in reranked_indices:
        sorted_summed_scores[retr_sorted_idx].cross_encoder_score = cross_scores[retr_sorted_idx]
        sorted_summed_scores[retr_sorted_idx].reranker_sorted_idx = reranker_sorted_idx
        print(f"retrieve_and_rerank_using_faiss: reranker_sorted_idx={reranker_sorted_idx}; retr_sorted_idx={retr_sorted_idx}; cross_encoder_score={sorted_summed_scores[retr_sorted_idx].cross_encoder_score}; vdb_similarity={sorted_summed_scores[retr_sorted_idx].distance}; faiss_rm_vdb_id={sorted_summed_scores[retr_sorted_idx].faiss_rm_vdb_id}; file_name={sorted_summed_scores[retr_sorted_idx].file_name}; para_id={sorted_summed_scores[retr_sorted_idx].para_id}")
        reranker_sorted_idx += 1

    if print_trace_context_choice:
        tracebuf.append(f"**{_prtime()}: Reranker Output**")
    print(f"**{_prtime()}: Reranker Output**")
    for i in range(len(reranked_indices)):
        # ri == reranked_idx
        ri = reranked_indices[i]
        chunk_det = sorted_summed_scores[ri]
        index_in_faiss = chunk_det.index_in_faiss
        fileid, para_index = chunk_det.file_id, chunk_det.para_id
        finfo = chunk_det.file_info
        if print_trace_context_choice:
            tracebuf.append(f"file={finfo['filename']}, paragraph_num={para_index}, cross_score={cross_scores[ri]}")
        print(f"file={finfo['filename']}, paragraph_num={para_index}, cross_score={cross_scores[ri]}")
        
        # if we asked to only print the 10 cross encoded outputs
        if cross_encoder_10 and i >= 10: break
    
    return reranked_indices, sorted_summed_scores

def _calc_cross_score_diffs(reranked_indices, sorted_summed_scores):
    rv = []
    for ind in range(1, len(reranked_indices)):
        prev_chosen_reranked_index = reranked_indices[ind-1]
        prev_chunk_det:DocumentChunkDetails = sorted_summed_scores[prev_chosen_reranked_index]
        chosen_reranked_index = reranked_indices[ind]
        chunk_det:DocumentChunkDetails = sorted_summed_scores[chosen_reranked_index]
        dif = prev_chunk_det.cross_encoder_score - chunk_det.cross_encoder_score
        rv.append((ind, dif))
    return rv

def _truncate_reranked_indices(reranked_indices, sorted_summed_scores, most_relevant):
    print(f"_truncate_reranked_indices: before truncating. reranked_indices={reranked_indices}, most_relevant={most_relevant}")
    csd=_calc_cross_score_diffs(reranked_indices, sorted_summed_scores)
    sorted_csd = sorted(csd, key=lambda x: x[1], reverse=True)

    ps=""
    for i in range(len(sorted_csd)):
        ps = ps + f"    {sorted_csd[i][0]},{sorted_csd[i][1]}"
    print(f"_truncate_reranked_indices: sorted cross score diffs={ps}")

    if most_relevant:
        rv = reranked_indices[:sorted_csd[0][0]]
    else:
        rv = reranked_indices[sorted_csd[0][0]:]
    print(f"_truncate_reranked_indices: after truncating. reranked_indices={rv}")
    return rv

def _get_context_using_retr_and_rerank(faiss_rms:List[faiss_rm.FaissRM], documents_list:List[Dict[str,str]], index_map_list:List[Tuple[str,str]],
                                    index_type, tracebuf:List[str], filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]],
                                    chat_config:ChatConfiguration, last_msg:str, most_relevant_only:bool, least_relevant_only:bool, searchsubdir:str=None):
    """
    documents is a dict like {fileid: finfo}; 
    index_map is a list of tuples: [(fileid, paragraph_index)];  the index into this list corresponds to the index of the embedding vector in the faiss index
    
    the context to be sent to the LLM.  Does a similarity search in faiss to fetch the context
    """
    reranked_indices:np.ndarray; sorted_summed_scores:List[DocumentChunkDetails]
    reranked_indices, sorted_summed_scores = retrieve_and_rerank_using_faiss(faiss_rms, documents_list, index_map_list, index_type, tracebuf,
                            filekey_to_file_chunks_dict, chat_config, last_msg, searchsubdir)

    # if most_relevant_only and least_relevant_only are both False, return all
    if most_relevant_only:
        reranked_indices = _truncate_reranked_indices(reranked_indices, sorted_summed_scores, True)
    elif least_relevant_only:
        reranked_indices = _truncate_reranked_indices(reranked_indices, sorted_summed_scores, False)

    context:str = ''
    all_docs_token_count = 0
    max_token_limit:int = MAX_TOKEN_LIMIT
    max_pre_and_post_token_limit = 512
    context_chunk_range_list:List[DocumentChunkRange] = []
    for i in range(len(reranked_indices)):
        chosen_reranked_index = reranked_indices[i]
        chunk_det:DocumentChunkDetails = sorted_summed_scores[chosen_reranked_index]
        print(f"_get_context_using_retr_and_rerank:: {i} :: chosen_reranked_index={chosen_reranked_index}, chunk details={chunk_det}")
        chunk_range:DocumentChunkRange = DocumentChunkRange(chunk_det)
        index_in_faiss = chunk_det.index_in_faiss
        fileid, para_index = chunk_det.file_id, chunk_det.para_id
        finfo = chunk_det.file_info
        if 'slides' in finfo:
            key = 'slides'
        elif 'paragraphs' in finfo:
            key = 'paragraphs'
        else:
            emsg = f"ERROR! Could not get key in document for {finfo}"
            print(emsg)
            return respond({"error_msg": emsg}, status=500)
        
        context_chunk_range_list.append(chunk_range)
        
        if chat_config.retreiver_strategy == RetrieverStrategyEnum.FullDocStrategy:
            fparagraphs = []
            for para in finfo[key]:
                fparagraphs.append(chunk_det.faiss_rm_vdb.format_paragraph(para))
            prelude = f"Name of the file is {chunk_det.file_name}"
            if len(context) + len(". ".join(fparagraphs)) > max_token_limit*3:  # each token on average is 3 bytes..
                # if the document is too long, just use the top hit paragraph and some subseq paras
                paras:List[str]; start_para_idx:int; end_para_idx:int
                paras, start_para_idx, end_para_idx = chunk_det.faiss_rm_vdb.get_paragraphs(index_in_faiss, 8)
                print(f"all paras in the file={chunk_det.file_name} > {max_token_limit} tokens in vdb_id={chunk_det.faiss_rm_vdb_id}: so retricting to paragraph number = {chunk_det.para_id} and max 7 more: start_para_idx={start_para_idx}; end_para_idx={end_para_idx}")
                if not paras:
                    emsg = f"ERROR! Could not get paragraph for reranked index {reranked_indices[0]}"
                    print(emsg)
                    return respond({"error_msg": emsg}, status=500)
                chunk_range.start_para_id = start_para_idx
                chunk_range.end_para_id = end_para_idx
                context = context + "\n" + prelude + "\n" + ". ".join(paras)
                break
            else:
                print(f"all paras in the file={chunk_det.file_name} para_id={chunk_det.para_id} vdb_id={chunk_det.faiss_rm_vdb_id} included in the context")
                chunk_range.start_para_id = 0
                chunk_range.end_para_id = len(finfo[key]) - 1
                context = context + "\n" + prelude + "\n" + ". ".join(fparagraphs)
        else:
            fparagraphs = []
            token_count:int = 0            
            for idx in range(chunk_det.para_id, -1, -1):
                formatted_para:str = chunk_det.faiss_rm_vdb.format_paragraph(finfo[key][idx])
                fparagraphs.insert(0,formatted_para)
                chunk_range.start_para_id = idx
                print(f"including prior     chunk: {chunk_det.file_name} para_number={chunk_det.para_id}:  Including para_number={idx}/{len(finfo[key])} in the context.  printed ordering may look incorrect but processing order is correct")
                tiktoken_count = calc_tokens(formatted_para)
                token_count += tiktoken_count
                all_docs_token_count += tiktoken_count
                if token_count >= max_pre_and_post_token_limit or all_docs_token_count >= max_token_limit: break

            token_count:int = 0
            chunk_range.end_para_id = chunk_det.para_id
            # if there are chunks after the current para_id
            if not (chunk_det.para_id + 1) == len(finfo[key]):
                for idx in range(chunk_det.para_id + 1, len(finfo[key])):
                    formatted_para:str = chunk_det.faiss_rm_vdb.format_paragraph(finfo[key][idx])
                    fparagraphs.append(formatted_para)
                    chunk_range.end_para_id = idx
                    print(f"including posterior chunk: {chunk_det.file_name} para_number={chunk_det.para_id}:  Including para_number={idx}/{len(finfo[key])} in the context.  printed ordering may look incorrect but processing order is correct")
                    
                    tiktoken_count = calc_tokens(formatted_para)
                    token_count += tiktoken_count
                    all_docs_token_count += tiktoken_count
                    if token_count >= max_pre_and_post_token_limit or all_docs_token_count >= max_token_limit: break
            
            prelude = f"Name of the file is {chunk_det.file_name}"
            context = context + "\n" + prelude + "\n" + ". ".join(fparagraphs)
            
            if all_docs_token_count >= max_token_limit: break

    err_dict, new_context = _gen_context(context_chunk_range_list)
    # TODO: fix this; caller doesn't handle returned err_dict
    if err_dict: return err_dict
    # if not context == new_context: 
    #     print(f"*** contexts are not the same:\ncontext=\n{context}\nnew_context=\n{new_context}")
    # else:
    #     print(f"*** contexts are the same")

    # file --> [ DocumentChunkDetails ]; dict of file to all chunks in the file that need to go into the context    
    for chunk_range in context_chunk_range_list:
        chunk_det = chunk_range.doc_chunk
        file_key:str = chunk_det._get_file_key()
        if filekey_to_file_chunks_dict.get(file_key):
            filekey_to_file_chunks_dict.get(file_key).append(chunk_det)
        else:
            filekey_to_file_chunks_dict[file_key] = [chunk_det]
    
    return new_context

@dataclasses.dataclass
class ContextSource:
    file_path:str
    file_name:str
    file_url:str
    
def _generate_context_sources(filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]]) -> Tuple[List[str],List[ContextSource]] :

    context_srcs:List[str] = []; context_srcs_links:List[str] = []
    for file_key, chunks in filekey_to_file_chunks_dict.items():
        para_indexes_str = ""
        for chunk_det in chunks:
            #chunk_det = chunks[0]
            para_indexes_str += f"ID={chunk_det.file_id}/{chunk_det.para_id}; "
        # "<content...>.  \n**Context Source: My Drive/Castle/Castle/Learning/Body/Tight Hamstrings Solutions – Low Back Pain Program.pdf**\t<!-- ID=1p3VhQOv_opCCWxaaYknduz-WepUDifH3/5; ID=1p3VhQOv_opCCWxaaYknduz-WepUDifH3/3;   index_id=0 -->;  **Context Source: My Drive/Castle/Castle/Learning/Body/Low_Back_Pain_Program_2018.pdf**\t<!-- ID=1KEJF5ttZCCsJqM7pSia7BAGbf-sxgAWB/41;   index_id=0 -->;  **Context Source: My Drive/Castle/Castle/Low_Back_Pain_Program_2018.pdf**\t<!-- ID=1-ACwOCTIT13K2DeceIotn7St2q8raC7S/41;   index_id=0 -->;  **Context Source: My Drive/Castle/Castle/Low_Back_Pain_Program_2018.pdf**\t<!-- ID=10FAzdHpUBIjtodPL-f1EtnIiAXKsSwM7/41;   index_id=0 --><!-- ; thread_id=thread_RVWHBSftNlpsXdyI0H44Gtmu -->"
        if chunk_det.file_path:
            context_srcs.append(f"**Context Source: {chunk_det.file_path}{chunk_det.file_name}**\t<!-- {para_indexes_str}  index_id={chunk_det.faiss_rm_vdb_id} -->")
            # generate context src hrefs
            context_srcs_links.append(ContextSource(chunk_det.file_path, chunk_det.file_name, chunk_det.file_type.generate_link(chunk_det.doc_storage_type, chunk_det.file_id)))
        else:
            context_srcs.append(f"**Context Source: {chunk_det.file_name}**\t<!-- {para_indexes_str}  index_id={chunk_det.faiss_rm_vdb_id} -->")
    
    return context_srcs, context_srcs_links

def _get_filelist_using_retr_and_rerank(faiss_rms:List[faiss_rm.FaissRM], documents_list:List[Dict[str,str]], index_map_list:List[Tuple[str,str]],
                                         index_type, tracebuf:List[str], filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]],
                                         chat_config:ChatConfiguration, last_msg:str, number_of_files:int = 10, searchsubdir:str=None):

    reranked_indices:np.ndarray; sorted_summed_scores:List[DocumentChunkDetails]
    reranked_indices, sorted_summed_scores = retrieve_and_rerank_using_faiss(faiss_rms, documents_list, index_map_list, index_type, tracebuf,
                              filekey_to_file_chunks_dict, chat_config, last_msg, searchsubdir)
    
    files_dict:Dict[str,DocumentChunkDetails] = {}
    for i in range(len(reranked_indices)):
        chosen_reranked_index = reranked_indices[i]
        chunk_det:DocumentChunkDetails = sorted_summed_scores[chosen_reranked_index]
    
        if not files_dict.get(chunk_det._get_file_key()):
            files_dict[chunk_det._get_file_key()] = chunk_det
        
        if len(files_dict) >= number_of_files: break
    
    return ",".join([  f"[{val.file_path}/{val.file_name}]({val.file_path}/{val.file_name})" for key, val in files_dict.items() ])
        

def print_file_details(event, faiss_rms:List[faiss_rm.FaissRM], documents_list:List[Dict[str, dict]], last_msg:str, use_ivfadc:bool):
    """ Returns the details of the filename specified in the query.  The format of the query is <filename>|<query>.  Looks up the query in the vector db, and only returns matches from the specified file, including details of the file and the matches in the file (paragraph_num and distance)    """
    last_msg = last_msg.strip()
    fnend = last_msg.find('|')
    if fnend == -1:
        fn = last_msg
        chat_msg = None
    else:
        fn = last_msg[:fnend]
        chat_msg = last_msg[fnend:]

    srp:str = ""
    for i in range(len(documents_list)):
        documents = documents_list[i]
        faiss_rm_vdb = faiss_rms[i]
        finfo = None
        for fi in documents.values():
            if fi['filename'] == fn:
                finfo = fi
                break
        if finfo:
            srp += f"{finfo['filename']}: fileid={finfo['fileid']}, path={finfo['path']}, mtime={finfo['mtime']}, mimetype={finfo['mimetype']}, num_paragraphs={len(finfo['paragraphs'])}"
            if chat_msg:
                index_map = faiss_rm_vdb.get_index_map()
                distances, indices = faiss_rm_vdb(chat_msg, k=len(index_map), index_type='ivfadc' if use_ivfadc else 'flat' )
                for itr in range(len(indices[0])):
                    ind = indices[0][itr]
                    # documents is {fileid: finfo}; index_map is [(fileid, paragraph_index)];  the index into this list corresponds to the index of the embedding vector in the faiss index
                    im = index_map[ind] 
                    if im[0] != finfo['fileid']:
                        continue
                    else:
                        fmtxt = faiss_rm_vdb.format_paragraph(finfo['paragraphs'][im[1]])
                        srp += f"\n\ndistance={distances[0][itr]}, paragraph_num={im[1]}, paragraph={fmtxt}"
            srp += "\n"
    if not srp: srp = "File not found in index"

    res = {}
    res['id'] = event['requestContext']['requestId']
    res['object'] = 'chat.completion.chunk'
    res['created'] = int(time.time_ns()/1000000)
    res['model'] = 'gpt-3.5-turbo-0125'
    res['choices'] = [
        {
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": f"\n\n{srp}"
                }
        }
    ]
    res_str = json.dumps(res)
    return {
        'statusCode': 200,
        'body': f"data:{res_str}",
        'headers': {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache, no-store, must-revalidate, private',
            'Pragma': 'no-cache',
            'Expires': '0'
        },
    }

g_cross_encoder = None
def new_chat(event, body, faiss_rms:List[faiss_rm.FaissRM], documents_list:List[Dict[str, dict]],
                index_map_list:List[List[Tuple[str,str]]], index_type:str = 'flat', sample_source=None,
                searchsubdir=None):
    """
    documents is a dict like {fileid: finfo}; 
    index_map is a list of tuples: [(fileid, paragraph_index)];  the index into this list corresponds to the index of the embedding vector in the faiss index
    """
    print(f"new_chat: entered")
    # {'messages': [{'role': 'user', 'content': 'Is there a memo?'}, {'role': 'assistant', 'content': '\n\nTO: All Developers\nFROM: John Smith\nDATE: 1st January 2020\nSUBJECT: A new PDF Parsing tool\n\nThere is a new PDF parsing tool available, called py-pdf-parser - you should all check it out! I think it could really help you extract that data we need from those PDFs.  \n**Context Source: simple_memo.pdf**\t<!-- ID=15-CEM_cX.../0 -->'}, {'role': 'user', 'content': 'can you repeat that?'}], 'model': 'gpt-3.5-turbo', 'stream': True, 'temperature': 1, 'top_p': 0.7}
    messages = body['messages']
    last_msg = messages[len(messages) - 1]['content']
    print(f"new_chat: Last Message = {last_msg}")
    
    tracebuf = ['**Begin Trace**']; filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]] = {}
    chat_config:ChatConfiguration; last_msg:str
    chat_config, last_msg = _debug_flags(last_msg, tracebuf)

    if chat_config.file_details:
        return print_file_details(event, faiss_rms, documents_list, last_msg, chat_config.use_ivfadc)
    if chat_config.dbg_set_searchsubdir:
        return do_set_searchsubdir(last_msg.strip())

    # string response??
    srp:str = ""; thread_id:str 
    srp, thread_id, run_usage = retrieve_using_openai_assistant(faiss_rms, documents_list, index_map_list, index_type, tracebuf,
                                    filekey_to_file_chunks_dict, None, chat_config, last_msg, searchsubdir)
    if not srp:
        return respond({"error_msg": "Error. retrieve using assistant failed"}, status=500)
    if run_usage:
        pct=int((float(run_usage.prompt_tokens)/float(MAX_TOKEN_LIMIT))*100.0)
        srp = srp +f"  \n**Tokens:** prompt={run_usage.prompt_tokens}({pct}% of {MAX_TOKEN_LIMIT}), completion={run_usage.completion_tokens}"

    context_srcs:List[str]; context_srcs_links:List[ContextSource]
    context_srcs, context_srcs_links = _generate_context_sources(filekey_to_file_chunks_dict)
    
    if chat_config.print_trace:
        tstr = ""
        for tt in tracebuf:
            tstr += f"  \n{tt}"
        srp = srp +tstr + f"  \n{';  '.join(context_srcs)}" + "<!-- ; thread_id=" + thread_id + " -->"
    else:
        srp = srp +f"  \n{';  '.join(context_srcs)}" + "<!-- ; thread_id=" + thread_id + " -->"

    print(f"new_chat: srp={srp}")
    res = {}
    res['id'] = event['requestContext']['requestId']
    res['object'] = 'chat.completion.chunk'
    res['created'] = int(time.time_ns()/1000000)
    res['model'] = 'gpt-3.5-turbo-0125'
    res['choices'] = [
        {
            "index": 0,
            # "finish_reason": "stop",
            "delta": {
                "role": "assistant",
                "content": f"\n\n{srp}"
                },
            "context_sources": json.loads(jsons.dumps(context_srcs_links))
        }
    ]
    if sample_source:
        res['choices'][0]['sample_source'] = sample_source

    res_str = json.dumps(res)
    print(f"new_chat: Returning {res_str}")
    return {
        'statusCode': 200,
        'body': f"data:{res_str}",
        'headers': {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache, no-store, must-revalidate, private',
            'Pragma': 'no-cache',
            'Expires': '0'
        },
    }
    return respond(None, res=res)

def get_filenames(faiss_rm_vdb):
    rv = []
    fls = faiss_rm_vdb.get_documents()
    for fileid, finfo in fls.items():
        rv.append(f"{finfo['path']}{finfo['filename']}" if 'path' in finfo else f"/{finfo['filename']}")
    return rv

def chat_completions(event, context):
    operation = event['requestContext']['http']['method']
    if (operation != 'POST'):
        return respond({"error_msg": "Error. http operation {operation} not supported"}, status=400)
    try:
        service_conf = get_service_conf()
    except Exception as ex:
        print(f"Caught {ex} while getting service_conf")
        return respond({"error_msg": f"Caught {ex} while getting service_conf"}, status=403)

    if 'bucket' not in service_conf or 'prefix' not in service_conf:
        print(f"Error. bucket and prefix not specified in service conf")
        return respond({"error_msg": "Error. bucket and prefix not specified in service_conf"}, status=403)
    bucket = service_conf['bucket']['S']
    prefix = service_conf['prefix']['S'].strip().strip('/')
    print(f"Index Location: s3://{bucket}/{prefix}")

    start_time:datetime.datetime = datetime.datetime.now()
    set_start_time(start_time)

    rv = check_cookie(event, False)
    email = rv['google']
    if not email:
        print(f"chat_completions: check_cookie did not return email. Sending 403")
        return respond({"status": "Unauthorized: please login to google auth"}, 403, None)

    if not 'body' in event:
        return respond({"error_msg": "Error. body not present"}, status=400)

    s3client = boto3.client('s3')

    body = json.loads(event['body'])
    print(f"body={body}")

    sample_source=None
    faiss_rm_vdbs:List[faiss_rm.FaissRM] = []
    for index,doc_storage_type in ( [None, faiss_rm.DocStorageType.GoogleDrive], ['dropbox', faiss_rm.DocStorageType.DropBox] ):
        faiss_rm_vdb:faiss_rm.FaissRM = init_vdb(email, s3client, bucket, prefix, doc_storage_type, build_faiss_indexes=False, sub_prefix=index)
        if faiss_rm_vdb: faiss_rm_vdbs.append(faiss_rm_vdb)
    if not len(faiss_rm_vdbs) or not faiss_rm_vdbs[0]:
        print(f"chat_completions: No full indexes found. Looking for sample index")
        faiss_rm_vdb:faiss_rm.FaissRM = init_vdb(email, s3client, bucket, prefix,
                                    faiss_rm.DocStorageType.Sample, build_faiss_indexes=False,
                                    sub_prefix="sample")
        if faiss_rm_vdb:
            print(f"chat_completions: Found and loaded sample index for {email}")
            faiss_rm_vdbs.append(faiss_rm_vdb)
        else:
            print(f"chat_completions: No sample index found for {email}. Creating sample index")
            ddbclient = boto3.client('dynamodb')
            if lock_sample_dir(email, ddbclient):
                print(f"chat_completions: Acquired sample index lock for {email}. Now creating..")
                try:
                    if create_sample_index(email, start_time, s3client, bucket, prefix):
                        faiss_rm_vdb:faiss_rm.FaissRM = init_vdb(email, s3client, bucket, prefix,
                                                                faiss_rm.DocStorageType.Sample,
                                                                build_faiss_indexes=False,
                                                                sub_prefix="sample")
                        if faiss_rm_vdb:
                            faiss_rm_vdbs.append(faiss_rm_vdb)
                        else:
                            print(f"chat_completions: Error in create sample index for {email}")
                            return respond({"error_msg": f"Error in create sample index for {email}"}, status=403)
                    else:
                        print(f"chat_completions: Unable to create sample index for {email}")
                        return respond({"error_msg": f"Unable to create sample index for {email}"}, status=403)
                except Exception as ex:
                    print(f"chat_completions: Caught {ex} creating sample index for {email}")
                    return respond({"error_msg": f"Caught {ex} creating sample index for {email}"}, status=403)
                finally:
                    unlock_sample_dir(email, ddbclient)
            else:
                print(f"chat_completions: sample index for user {email} is being created by another lambda instance.")
                return respond({"error_msg": f"No document index found and sample index being created by another lambda. Please wait and try later.."}, status=503)
        sample_source = get_filenames(faiss_rm_vdbs[0])

    documents_list:List[Dict[str, dict]] = []
    index_map_list:List[List[Tuple[str,str]]] = []
    for faiss_rm_vdb in faiss_rm_vdbs:
        documents_list.append(faiss_rm_vdb.get_documents())
        index_map_list.append(faiss_rm_vdb.get_index_map())

    messages = body['messages']
    if len(messages) == 1:
        return new_chat(event, body, faiss_rm_vdbs, documents_list, index_map_list, sample_source=sample_source,
                                              searchsubdir=rv['searchsubdir'] if 'searchsubdir' in rv else None)
    else:
        return ongoing_chat(event, body, faiss_rm_vdbs, documents_list, index_map_list, sample_source=sample_source,
                                              searchsubdir=rv['searchsubdir'] if 'searchsubdir' in rv else None)
        
if __name__ != '__main1__': traceback_with_variables.global_print_exc()
