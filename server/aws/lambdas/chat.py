import io
import json
import os
import sys
import traceback
import tempfile
import uuid
import time
import base64
import zlib
import datetime
from urllib.parse import unquote
import numpy as np
from utils import respond, get_service_conf, check_cookie, set_start_time, prtime
from index_utils import init_vdb
import boto3
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
from fetch_tool_prompts import fetch_tool_prompts
from text_utils import format_paragraph

MAX_TOKEN_LIMIT=2048
MAX_PRE_AND_POST_TOKEN_LIMIT=256
MAX_VDB_RESULTS=1024
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

def calc_tokens(context):
    global encoding_model
    return len(encoding_model.encode(context))

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
    XLSX = 5
    HTML = 6
    GH_ISSUES_ZIP = 7

    @classmethod
    def fromString(clz, doc_type_str:str):
        str_to_type_dict:Dict[str, Self] = { 'pptx':clz.PPTX, 'docx':clz.DOCX, 'pdf':clz.PDF,
                                              'txt':clz.TXT, 'xlsx': clz.XLSX, 'html': clz.HTML,
                                              'gh-issues.zip': clz.GH_ISSUES_ZIP }
        
        if not str_to_type_dict.get(doc_type_str):
            raise Exception(f"Unknown document type:  {doc_type_str}")
            
        return str_to_type_dict.get(doc_type_str)

    def file_ext(self) -> str:
        doc_type_to_ext:Dict[Self, str] = {
            self.__class__.DOCX:'docx',
            self.__class__.PPTX:'pptx',
            self.__class__.XLSX:'xlsx',
            self.__class__.PDF:'pdf',
            self.__class__.TXT:'txt',
            self.__class__.HTML:'html',
            self.__class__.GH_ISSUES_ZIP:'gh-issues.zip'
        }
        if not doc_type_to_ext.get(self):
            raise Exception(f"file_ext(): Unknown document type:  self={self}")
        return doc_type_to_ext[self]

    def generate_link(self, doc_storage_type:faiss_rm.DocStorageType, file_id:str, para_dict=None) -> str:
        if doc_storage_type == faiss_rm.DocStorageType.GoogleDrive:
            # If word document (ends with doc or docx), use the link  https://docs.google.com/document/d/<file_id>
            # if a pdf file (ends with pdf), use the link https://drive.google.com/file/d/<file_id>
            # if pptx file (ends with ppt or pptx) https://docs.google.com/presentation/d/<file_id>
            doc_type_to_link:Dict[Self, str] = {
                self.__class__.DOCX:'https://docs.google.com/document/d/{file_id}',
                self.__class__.PPTX:'https://docs.google.com/presentation/d/{file_id}',
                self.__class__.XLSX:'https://docs.google.com/spreadsheets/d/{file_id}',
                self.__class__.PDF:'https://drive.google.com/file/d/{file_id}',
                self.__class__.TXT:'https://drive.google.com/file/d/{file_id}',
                self.__class__.HTML:'https://drive.google.com/file/d/{file_id}',
                self.__class__.GH_ISSUES_ZIP:para_dict['html_url'] if para_dict and 'html_url' in para_dict else 'https://drive.google.com/file/d/{file_id}'
            }
            if not doc_type_to_link.get(self):
                raise Exception(f"generate_link(): Unknown document type:  self={self}; doc_storage_type={doc_storage_type}; file_id={file_id}")
            return doc_type_to_link[self].format(file_id=file_id)
        elif doc_storage_type == faiss_rm.DocStorageType.Sample:
            # If word document (ends with doc or docx), use the link  https://docs.google.com/document/d/<file_id>
            # if a pdf file (ends with pdf), use the link https://drive.google.com/file/d/<file_id>
            # if pptx file (ends with ppt or pptx) https://docs.google.com/presentation/d/<file_id>
            doc_type_to_link:Dict[Self, str] = {
                self.__class__.DOCX:'https://docs.google.com/document/d/{file_id}',
                self.__class__.PPTX:'https://docs.google.com/presentation/d/{file_id}',
                self.__class__.XLSX:'https://docs.google.com/spreadsheets/d/{file_id}',
                self.__class__.PDF:'https://drive.google.com/file/d/{file_id}',
                self.__class__.TXT:'https://drive.google.com/file/d/{file_id}',
                self.__class__.HTML:'https://drive.google.com/file/d/{file_id}',
                self.__class__.GH_ISSUES_ZIP:para_dict['html_url'] if para_dict and 'html_url' in para_dict else 'https://drive.google.com/file/d/{file_id}'
            }
            if not doc_type_to_link.get(self):
                raise Exception(f"generate_link(): Unknown document type:  self={self}; doc_storage_type={doc_storage_type}; file_id={file_id}")
            return doc_type_to_link[self].format(file_id=file_id)
        elif doc_storage_type == faiss_rm.DocStorageType.DropBox:
            doc_type_to_link:Dict[Self, str] = { # XXX wrong links for dropbox
                self.__class__.DOCX:'https://docs.google.com/document/d/{file_id}',
                self.__class__.PPTX:'https://docs.google.com/presentation/d/{file_id}',
                self.__class__.XLSX:'https://docs.google.com/spreadsheets/d/{file_id}',
                self.__class__.PDF:'https://drive.google.com/file/d/{file_id}',
                self.__class__.TXT:'https://drive.google.com/file/d/{file_id}',
                self.__class__.HTML:'https://drive.google.com/file/d/{file_id}',
                self.__class__.GH_ISSUES_ZIP:'https://drive.google.com/file/d/{file_id}'
            }
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
    file_details:bool
    retreiver_strategy:RetrieverStrategyEnum

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
        
def ongoing_chat(event, body, chat_config, tracebuf, last_msg, faiss_rms:List[faiss_rm.FaissRM], documents_list:List[Dict[str, dict]],
                    index_map_list:List[List[Tuple[str, str]]], index_type:str = 'flat',
                    searchsubdir=None, toolprompts=None):
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

    filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]] = {};
    srp, thread_id, run_usage = retrieve_using_openai_assistant(faiss_rms,documents_list, index_map_list, index_type,
                        tracebuf, filekey_to_file_chunks_dict, thread_id, chat_config, last_msg, toolprompts)
    if not srp:
        return respond({"error_msg": "Error. retrieve using assistant failed"}, status=500)
    if run_usage:
        pct=(float(run_usage.prompt_tokens)/float(MAX_TOKEN_LIMIT))*100.0
        srp = srp +f"  \n**Tokens:** prompt={run_usage.prompt_tokens}({pct}% of {MAX_TOKEN_LIMIT}), completion={run_usage.completion_tokens}"
    
    context_srcs_links:List[ContextSource]
    context_srcs_links = _generate_context_sources(filekey_to_file_chunks_dict)
    srp = srp +f"  \n<!-- ; thread_id={thread_id} -->"
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
    if searchsubdir:
        res['choices'][0]['searchsubdir'] = searchsubdir
    if chat_config.print_trace:
        res['choices'][0]['tracebuf'] = tracebuf
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

def replace_tools_in_assistant(new_tools):
    global assistant
    assistant = client.beta.assistants.create(
        instructions="You are a helpful assistant. Use the provided functions to access confidential and private information and answer questions or provide details of the mentioned subject.",
        model=ASSISTANTS_MODEL,
        tools=new_tools
    )


def _debug_flags(query:str, tracebuf:List[str]) -> Tuple[ChatConfiguration, str]:
    print_trace, use_ivfadc, file_details, retriever_stratgey = \
                (False, False, False, RetrieverStrategyEnum.PreAndPostChunkStrategy)
    idx = 0
    for idx in range(len(query)):
        # '+': print_trace
        # '@': use ivfadc index
        # '^': print_trace with info about choice of context
        # '!': print details of file
        c = query[idx]
        if c not in ['+','@','#','$', '^', '!', '/', '~']: break
        
        if c == '+': print_trace = True
        if c == '@': use_ivfadc = not use_ivfadc
        if c == '!': file_details = True
        if c == '/': retriever_stratgey = RetrieverStrategyEnum.FullDocStrategy
    
    # strip the debug flags from the question
    if idx == len(query) - 1:
        last_msg = ""
    else:
        last_msg = query[idx:]
    chat_config = ChatConfiguration(print_trace, use_ivfadc, file_details, retriever_stratgey)
    logmsg = f"**{prtime()}: Debug Flags**: chat_config={chat_config}, last_={last_msg}"
    print(logmsg)

    return (chat_config, last_msg)

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
                    context:str = _get_context(faiss_rms, documents_list, index_map_list, index_type, tracebuf,
                                                filekey_to_file_chunks_dict, chat_config, tool_arg_question, True, False, searchsubdir=searchsubdir)
                    print(f"{prtime()}: Tool output: context={context}")
                    tracebuf.append(f"{prtime()}: Tool output: context={context[:64]}...")
                    tool_outputs.append({
                        "tool_call_id": tool.id,
                        "output": context
                    })
                elif tool.function.name == "search_question_in_db_return_more" or tool.function.name == 'search_question_in_db_return_more.controls':
                    tool_arg_question = args_dict.get('question')
                    context:str = _get_context(faiss_rms, documents_list, index_map_list, index_type, tracebuf,
                                                filekey_to_file_chunks_dict, chat_config, tool_arg_question, False, True, searchsubdir=searchsubdir)
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

def _get_key(finfo):
    if 'slides' in finfo:
        return 'slides'
    elif 'paragraphs' in finfo:
        return 'paragraphs'
    elif 'rows' in finfo:
        return 'rows'
    else:
        return None

def _gen_context(context_chunk_range_list:List[DocumentChunkRange], handle_overlaps:bool = True) -> Tuple[dict, str]:
    """returns the tupe (error_dict, context)"""
    
    print(f"_gen_context(): context_chunk_range_list={context_chunk_range_list}")
    if handle_overlaps and len(context_chunk_range_list) > 1:
        context_chunk_range_list = DocumentChunkRange.process_overlaps(context_chunk_range_list)
        print(f"_gen_context(): context_chunk_range_list after overlapping merge={context_chunk_range_list}")
    
    # generate the context
    new_context:str = ''    
    for chunk_range in context_chunk_range_list:
        chunk_det = chunk_range.doc_chunk
        fparagraphs = []
        finfo = chunk_range.doc_chunk.file_info
        key = _get_key(finfo)
        if not key:
            emsg = f"ERROR! Could not get key in document for {finfo}"
            print(emsg)
            return respond({"error_msg": emsg}, status=500), None
        for idx in range(chunk_range.start_para_id, chunk_range.end_para_id+1):
            formatted_para:str = format_paragraph(finfo[key][idx])
            fparagraphs.append(formatted_para)
            print(f"Context: Included chunk from file_name={chunk_det.file_name} para_id={idx} faiss_rm_vdb_id={chunk_det.faiss_rm_vdb_id}")
        prelude = f"Name of the file is {chunk_det.file_name}"
        new_context = new_context + "\n" + prelude + "\n" + ". ".join(fparagraphs)
        
    return None, new_context

def extract_named_entities(text):
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": 
f"Extract any named entities present in the sentence. Return a JSON object in this format: {{ 'entities': ['entity1', 'entity2'] }}, without any additional text or explanation. Particularly, do not include text before or after the parseable JSON: {text}"}
        ]
    )
    content = completion.choices[0].message.content
    print(f"extract_named_entities: chatgpt returned content {content}")
    try:
        js = json.loads(content)
    except Exception as ex:
        print(f"extract_named_entities: caught {ex}")
        return None
    print(f"extract_named_entities: js= {js}")
    if 'entities' in js and js['entities']:
        return js['entities']
    else:
        return None

def extract_main_theme(text):
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"Extract the main topic as a single word in the following sentence and return the result as a single word: {text}"}
        ]
    )
    retval = completion.choices[0].message.content
    print(f"extract_main_theme: chatgpt returned {retval}")
    return retval

def _retrieve_rerank(faiss_rms:List[faiss_rm.FaissRM], documents_list:List[Dict[str,str]], index_map_list:List[Tuple[str,str]],
                                    index_type, tracebuf:List[str], filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]],
                                    chat_config:ChatConfiguration, last_msg:str, searchsubdir:str=None) -> Tuple[np.ndarray, List[DocumentChunkDetails]]:
    """
    documents is a dict like {fileid: finfo}; 
    index_map is a list of tuples: [(fileid, paragraph_index)];  the index into this list corresponds to the index of the embedding vector in the faiss index
    
    returns the reranked indices (index into list of documentChunkDetails) and the list of DocumentChunkDetails  """
    use_ivfadc:bool; retreiver_strategy:RetrieverStrategyEnum
    use_ivfadc, retreiver_strategy = ( chat_config.use_ivfadc, chat_config.retreiver_strategy)
    
    queries = [last_msg]
    print(f"queries={queries}")
    tracebuf.append(f"{prtime()} Tool call:search_question_in_db: Entered. Queries:")
    tracebuf.extend(queries)

    named_entities = extract_named_entities(last_msg)
    main_theme = extract_main_theme(last_msg)
    print(f"named_entities={named_entities}, main_theme={main_theme}")
    tracebuf.append(f"{prtime()} named_entities={named_entities}, main theme={main_theme}")

    sorted_summed_scores:List[DocumentChunkDetails] = []
    for i in range(len(faiss_rms)):
        faiss_rm_vdb = faiss_rms[i]
        documents = documents_list[i]
        index_map = index_map_list[i]
        
        # dict of { index_in_faiss:[distance1, distance2] }
        passage_scores_dict:Dict[int, List] = {}
        for qind in range(len(queries)):
            qr = queries[qind]
            distances, indices_in_faiss = faiss_rm_vdb(qr, k=MAX_VDB_RESULTS, index_type='ivfadc' if use_ivfadc else 'flat',
                                                    named_entities=named_entities, main_theme=main_theme)
            for idx in range(len(indices_in_faiss[0])):
                ind_in_faiss = indices_in_faiss[0][idx]
                finfo = documents[index_map[ind_in_faiss][0]]
                if searchsubdir:
                    if 'path' in finfo:
                        if finfo['path'].startswith(searchsubdir):
                            print(f"_retrieve_rerank: searchsubdir={searchsubdir}. Accepting {finfo['filename']} path={finfo['path']}, para={index_map[ind_in_faiss][1]}")
                            # the first query in queries[] is the actual user chat text. we give that twice the weight
                            dist = distances[0][idx] if qind == 0 else distances[0][idx]/2.0
                            if ind_in_faiss in passage_scores_dict:
                                passage_scores_dict[ind_in_faiss].append(dist)
                            else:
                                passage_scores_dict[ind_in_faiss] = [dist]
                        else:
                            print(f"_retrieve_rerank: searchsubdir={searchsubdir}. Rejecting for path mismatch {finfo['filename']} path={finfo['path']}, para={index_map[ind_in_faiss][1]}")
                    else:
                        print(f"_retrieve_rerank: searchsubdir={searchsubdir}. Rejecting {finfo['filename']}, para={index_map[ind_in_faiss][1]} since no path")
                else:
                    print(f"_retrieve_rerank: No searchsubdir. Accepting {finfo['filename']} path={finfo['path'] if 'path' in finfo else 'unavailable'}, para={index_map[ind_in_faiss][1]}")
                    # the first query in queries[] is the actual user chat text. we give that twice the weight
                    dist = distances[0][idx] if qind == 0 else distances[0][idx]/2.0
                    if ind_in_faiss in passage_scores_dict:
                        passage_scores_dict[ind_in_faiss].append(dist)
                    else:
                        passage_scores_dict[ind_in_faiss] = [dist]

        if len(passage_scores_dict.items()) == 0:
            print(f"_retrieve_rerank: No entries in passage_scores!!")
            return False, None

        # faiss returns METRIC_INNER_PRODUCT - larger number means better match
        # sum the passage scores

        summed_scores:List[DocumentChunkDetails] = [] # array of (summed_score, index_in_faiss)
        for index_in_faiss, scores in passage_scores_dict.items():
            fileid = index_map[index_in_faiss][0]
            para_index = index_map[index_in_faiss][1]
            finfo = documents[fileid]
            key = _get_key(finfo)
            if not key:
                print(f"Error. Skipping since we could not determine key for {finfo}")
                continue
            summed_scores.append(DocumentChunkDetails(index_in_faiss,
                                                    faiss_rm_vdb,
                                                    i,
                                                    faiss_rm_vdb.get_doc_storage_type(),
                                                    sum(scores), 
                                                    DocumentType.fromString(finfo['filetype']),
                                                    finfo.get('path') if finfo.get('path') else None,
                                                    finfo['filename'],
                                                    fileid,
                                                    finfo,
                                                    para_index,
                                                    finfo[key][para_index],
                                                      ))
        sorted_summed_scores.extend( summed_scores )
    
    sorted_summed_scores = sorted(sorted_summed_scores, key=lambda x: x.distance, reverse=True)

    # Note that these three arrays are aligned: using the same index in these 3 arrays retrieves corresponding elements: reranker_map (array of faiss_indexes), reranker_input (array of (query, formatted para)) and cross_scores (array of cross encoder scores)
    reranker_map = [] # array of index_in_faiss
    reranker_input = [] # array of (query, formatted_para)
    for idx in range(min(len(sorted_summed_scores), MAX_VDB_RESULTS)):
        curr_chunk:DocumentChunkDetails = sorted_summed_scores[idx]
        curr_chunk.retr_sorted_idx = idx
        index_in_faiss = curr_chunk.index_in_faiss
        curr_chunk.para_dict = curr_chunk.faiss_rm_vdb.get_paragraph(index_in_faiss)
        # force an empty formatted_paragraph from format_paragraph() below, by using an empty dict
        if not curr_chunk.para_dict: curr_chunk.para_dict = {}
        curr_chunk.para_text_formatted = f"Name of the file is {curr_chunk.file_name}\n" + format_paragraph(curr_chunk.para_dict)
        reranker_input.append([last_msg, curr_chunk.para_text_formatted])

    global g_cross_encoder
    # https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-2-v2
    if not g_cross_encoder: g_cross_encoder = CrossEncoder('/var/task/cross-encoder/ms-marco-MiniLM-L-6-v2') if os.path.isdir('/var/task/cross-encoder/ms-marco-MiniLM-L-6-v2') else CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    # _retrieve_rerank: cross_scores=[-10.700319  -11.405142   -3.5650876  -8.041701   -9.972779   -9.609493 -10.653023   -6.8494396  -7.601103  -11.405787  -10.690331  -10.050377 ...
    # Note that these three arrays are aligned: using the same index in these 3 arrays retrieves corresponding elements: reranker_map (array of faiss_indexes), reranker_input (array of (query, formatted para)) and cross_scores (array of cross encoder scores)
    # 
    # Negative Scores for cross-encoder/ms-marco-MiniLM-L-6-v2 #1058: https://github.com/UKPLab/sentence-transformers/issues/1058
    cross_scores:np.ndarray = g_cross_encoder.predict(reranker_input)
    print(f"_retrieve_rerank: cross_scores={cross_scores}")
    # Returns the indices into the given cross_scores array, that would sort the given cross_scores array.
    # Perform an indirect sort along the given axis using the algorithm specified by the kind keyword. It returns an array of indices of the same shape as a that index data along the given axis in sorted order.
    reranked_indices = np.argsort(cross_scores)[::-1]
    print(f"_retrieve_rerank: reranked_indices={reranked_indices}")
    cross_sorted_scores:List[DocumentChunkDetails] = []
    for ind in range(len(reranked_indices)):
        chk=sorted_summed_scores[reranked_indices[ind]]
        chk.cross_encoder_score = cross_scores[reranked_indices[ind]]
        cross_sorted_scores.append(chk)
    _lg(tracebuf, f"{prtime()}_retrieve_rerank: cross_sorted_scores:")
    for ind in range(len(cross_sorted_scores)):
        chk = cross_sorted_scores[ind]
        _lg(tracebuf, f"  reranker_sorted_idx={ind}, {chk.file_info['path']}{chk.file_info['filename']},para={chk.para_id}: cross_encoder_score={chk.cross_encoder_score}, index_in_faiss={chk.index_in_faiss}, distance={chk.distance}")
    return True, cross_sorted_scores

def _calc_cross_sorted_diffs(cross_sorted_scores):
    rv = []
    for ind in range(1, len(cross_sorted_scores)):
        prev_chunk = cross_sorted_scores[ind-1]
        chunk = cross_sorted_scores[ind]
        dif = prev_chunk.cross_encoder_score - chunk.cross_encoder_score
        rv.append((ind, dif))
    return rv

def _truncate_cross_sorted_scores(cross_sorted_scores, most_relevant):
    print(f"_truncate_cross_sorted_scores: Entered. most_relevant={most_relevant}")
    csd=_calc_cross_sorted_diffs(cross_sorted_scores)
    print(f"_truncate_cross_sorted_scores: csd={csd}")
    sorted_csd = sorted(csd, key=lambda x: x[1], reverse=True)
    print(f"_truncate_cross_sorted_scores: sorted_csd={sorted_csd}. Trunc point={sorted_csd[0][0]}")
    if most_relevant:
        rv = cross_sorted_scores[:sorted_csd[0][0]]
    else:
        rv = cross_sorted_scores[sorted_csd[0][0]:]
    print(f"_truncate_cross_sorted_scores: after truncating. truncated cross_sorted_scores={rv}")
    return rv

def _get_context(faiss_rms:List[faiss_rm.FaissRM], documents_list:List[Dict[str,str]], index_map_list:List[Tuple[str,str]],
                                    index_type, tracebuf:List[str], filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]],
                                    chat_config:ChatConfiguration, last_msg:str, most_relevant_only:bool, least_relevant_only:bool, searchsubdir:str=None):
    """
    documents is a dict like {fileid: finfo}; 
    index_map is a list of tuples: [(fileid, paragraph_index)];  the index into this list corresponds to the index of the embedding vector in the faiss index
    
    the context to be sent to the LLM.  Does a similarity search in faiss to fetch the context
    """
    cross_sorted_scores:List[DocumentChunkDetails]
    status, cross_sorted_scores = _retrieve_rerank(faiss_rms, documents_list, index_map_list, index_type, tracebuf,
                            filekey_to_file_chunks_dict, chat_config, last_msg, searchsubdir)
    if not status:
        print(f"_get_context: no context from _retrieve_rerank")
        return "No context found"

    cross_sorted_scores = cross_sorted_scores[:16]
    # if most_relevant_only and least_relevant_only are both False, return all
    if len(cross_sorted_scores) > 1:
        if most_relevant_only:
            cross_sorted_scores = _truncate_cross_sorted_scores(cross_sorted_scores, True)
        elif least_relevant_only:
            cross_sorted_scores = _truncate_cross_sorted_scores(cross_sorted_scores, False)
    _lg(tracebuf, f"{prtime()}_get_context: length of truncated cross_sorted_scores = {len(cross_sorted_scores)}")

    context:str = ''
    all_docs_token_count = 0
    max_token_limit:int = MAX_TOKEN_LIMIT
    max_pre_and_post_token_limit = MAX_PRE_AND_POST_TOKEN_LIMIT
    context_chunk_range_list:List[DocumentChunkRange] = []
    for i in range(len(cross_sorted_scores)):
        chunk_det:DocumentChunkDetails = cross_sorted_scores[i]
        chunk_range:DocumentChunkRange = DocumentChunkRange(chunk_det)
        index_in_faiss = chunk_det.index_in_faiss
        fileid, para_index = chunk_det.file_id, chunk_det.para_id
        finfo = chunk_det.file_info
        _lg(tracebuf, f"{prtime()}_get_context: Processing {finfo['path']}{finfo['filename']},para={chunk_det.para_id}")

        key = _get_key(finfo)
        if not key:
            emsg = f"ERROR! Could not get key in document for {finfo}"
            print(emsg)
            return respond({"error_msg": emsg}, status=500)
        
        context_chunk_range_list.append(chunk_range)

        if chunk_det.file_type == DocumentType.GH_ISSUES_ZIP:
            chunk_range.start_para_id = chunk_det.para_id
            chunk_range.end_para_id = chunk_det.para_id
            _lg(tracebuf, f"{prtime()}_get_context: gh issues zip file. Not adding previous or next paragraphs for context")
            break
        
        if chat_config.retreiver_strategy == RetrieverStrategyEnum.FullDocStrategy:
            fparagraphs = []
            for para in finfo[key]:
                fparagraphs.append(format_paragraph(para))
            prelude = f"Name of the file is {chunk_det.file_name}"
            if len(context) + len(". ".join(fparagraphs)) > max_token_limit*3:  # each token on average is 3 bytes..
                # if the document is too long, just use the top hit paragraph and some subseq paras
                paras:List[str]; start_para_idx:int; end_para_idx:int
                paras, start_para_idx, end_para_idx = chunk_det.faiss_rm_vdb.get_paragraphs(index_in_faiss, 8)
                print(f"all paras in the file={chunk_det.file_name} > {max_token_limit} tokens in vdb_id={chunk_det.faiss_rm_vdb_id}: so retricting to paragraph number = {chunk_det.para_id} and max 7 more: start_para_idx={start_para_idx}; end_para_idx={end_para_idx}")
                if not paras:
                    emsg = f"ERROR! Could not get paragraph for context {chunk_det}"
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
                if not finfo[key][idx]: # Paragraphs/Slides/rows can be sparse
                    break
                formatted_para:str = format_paragraph(finfo[key][idx])
                fparagraphs.insert(0,formatted_para)
                chunk_range.start_para_id = idx
                tiktoken_count = calc_tokens(formatted_para)
                token_count += tiktoken_count
                all_docs_token_count += tiktoken_count
                if token_count >= max_pre_and_post_token_limit or all_docs_token_count >= max_token_limit: break
            _lg(tracebuf, f"{prtime()}_get_context: including prior chunks upto {idx} for {chunk_det.file_name} hit para_number={chunk_det.para_id}")

            token_count:int = 0
            chunk_range.end_para_id = chunk_det.para_id
            # if there are chunks after the current para_id
            if not (chunk_det.para_id + 1) == len(finfo[key]):
                for idx in range(chunk_det.para_id + 1, len(finfo[key])):
                    if not finfo[key][idx]: # Paragraphs/Slides/rows can be sparse
                        break
                    formatted_para:str = format_paragraph(finfo[key][idx])
                    fparagraphs.append(formatted_para)
                    chunk_range.end_para_id = idx
                    
                    tiktoken_count = calc_tokens(formatted_para)
                    token_count += tiktoken_count
                    all_docs_token_count += tiktoken_count
                    if token_count >= max_pre_and_post_token_limit or all_docs_token_count >= max_token_limit: break
                _lg(tracebuf, f"{prtime()}_get_context: including posterior chunks upto {idx} for {chunk_det.file_name} hit para_number={chunk_det.para_id}")
            
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
    file_id:str
    para_id:str
    file_extn:str
    
def _generate_context_sources(filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]]) -> Tuple[List[str],List[ContextSource]] :
    context_srcs_links:List[str] = []
    csdict = {}
    for file_key, chunks in filekey_to_file_chunks_dict.items():
        for chunk_det in chunks:
            if chunk_det.file_id not in csdict:
                csdict[chunk_det.file_id] = chunk_det # Only one context source for each file
                para_dict = chunk_det.para_dict
                if chunk_det.file_path:
                    context_srcs_links.append(ContextSource(chunk_det.file_path, chunk_det.file_name,
                                            chunk_det.file_type.generate_link(chunk_det.doc_storage_type, chunk_det.file_id, para_dict),
                                            chunk_det.file_id, str(chunk_det.para_id), chunk_det.file_type.file_ext()))
                else:
                    context_srcs_links.append(ContextSource("", chunk_det.file_name,
                                            chunk_det.file_type.generate_link(chunk_det.doc_storage_type, chunk_det.file_id, para_dict),
                                            chunk_det.file_id, str(chunk_det.para_id), chunk_det.file_type.file_ext()))
    return context_srcs_links

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
                        fmtxt = format_paragraph(finfo['paragraphs'][im[1]])
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
def new_chat(event, body, chat_config, tracebuf, last_msg, faiss_rms:List[faiss_rm.FaissRM], documents_list:List[Dict[str, dict]],
                index_map_list:List[List[Tuple[str,str]]], index_type:str = 'flat',
                searchsubdir=None, toolprompts=None):
    """
    documents is a dict like {fileid: finfo}; 
    index_map is a list of tuples: [(fileid, paragraph_index)];  the index into this list corresponds to the index of the embedding vector in the faiss index
    """
    print(f"new_chat: entered")
    filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]] = {}

    if chat_config.file_details:
        return print_file_details(event, faiss_rms, documents_list, last_msg, chat_config.use_ivfadc)

    # string response??
    srp:str = ""; thread_id:str 
    srp, thread_id, run_usage = retrieve_using_openai_assistant(faiss_rms, documents_list, index_map_list, index_type, tracebuf,
                                    filekey_to_file_chunks_dict, None, chat_config, last_msg, searchsubdir, toolprompts)
    if not srp:
        return respond({"error_msg": "Error. retrieve using assistant failed"}, status=500)
    if run_usage:
        pct=int((float(run_usage.prompt_tokens)/float(MAX_TOKEN_LIMIT))*100.0)
        srp = srp +f"  \n**Tokens:** prompt={run_usage.prompt_tokens}({pct}% of {MAX_TOKEN_LIMIT}), completion={run_usage.completion_tokens}"

    context_srcs_links:List[ContextSource]
    context_srcs_links = _generate_context_sources(filekey_to_file_chunks_dict)
    
    srp = srp +f"  \n<!-- ; thread_id={thread_id} -->"

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
    if searchsubdir:
        res['choices'][0]['searchsubdir'] = searchsubdir
    if chat_config.print_trace:
        res['choices'][0]['tracebuf'] = tracebuf

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

    if 'YOJA_USER' in os.environ:
        rv = {}
        email = os.environ['YOJA_USER']
        print(f"chat_completions: YOJA_USER specified email={email}. Bypassing cookie..")
    else:
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

    last_msg:str = body['messages'][-1]['content']
    tracebuf = [f"{prtime()} Begin Trace"];
    chat_config, last_msg = _debug_flags(last_msg, tracebuf)

    searchsubdir = None
    faiss_rms:List[faiss_rm.FaissRM] = []
    if 'searchsubdir' in rv:
        ss1 = rv['searchsubdir']
        if ss1.startswith('[gdrive]'):
            searchsubdir = ss1[8:].lstrip('/').rstrip('/')
            faiss_rm_vdb:faiss_rm.FaissRM = init_vdb(email, s3client, bucket, prefix,
                                                    faiss_rm.DocStorageType.GoogleDrive,
                                                    chat_config=chat_config, tracebuf=tracebuf,
                                                    build_faiss_indexes=False, sub_prefix=None)
            if faiss_rm_vdb: faiss_rms.append(faiss_rm_vdb)
        elif ss1.startswith('[dropbox]'):
            searchsubdir = ss1[9:].lstrip('/').rstrip('/')
            faiss_rm_vdb:faiss_rm.FaissRM = init_vdb(email, s3client, bucket, prefix,
                                                    faiss_rm.DocStorageType.DropBox,
                                                    chat_config=chat_config, tracebuf=tracebuf,
                                                    build_faiss_indexes=False, sub_prefix='dropbox')
            if faiss_rm_vdb: faiss_rms.append(faiss_rm_vdb)
        else:
            print(f"Error. searchsubdir does not start with [gdrive] or [dropbox]. Ignoring")
    else:
        for index,doc_storage_type in ( [None, faiss_rm.DocStorageType.GoogleDrive], ['dropbox', faiss_rm.DocStorageType.DropBox] ):
            faiss_rm_vdb:faiss_rm.FaissRM = init_vdb(email, s3client, bucket, prefix,
                                                    doc_storage_type,
                                                    chat_config=chat_config, tracebuf=tracebuf,
                                                    build_faiss_indexes=False, sub_prefix=index)
            if faiss_rm_vdb: faiss_rms.append(faiss_rm_vdb)
    if not len(faiss_rms) or not faiss_rms[0]:
        print(f"chat_completions: index not available.")
        return respond({"error_msg": f"Index creation in progress. Please wait and try later.."}, status=507)

    documents_list:List[Dict[str, dict]] = []
    index_map_list:List[List[Tuple[str,str]]] = []
    for faiss_rm_vdb in faiss_rms:
        documents_list.append(faiss_rm_vdb.get_documents())
        index_map_list.append(faiss_rm_vdb.get_index_map())

    toolprompts=None
    msgs = []
    for msg in body['messages']:
        if msg['content'].strip()[0] == '%':
            print(f"chat_completions: Applying tool prompts sheet {msg['content'][1:]}")
            toolprompts = fetch_tool_prompts(email, msg['content'])
        else:
            msgs.append(msg)
    body['messages'] = msgs
    print(f"chat_completions: finished pre-processing. messages={body['messages']}")
    if len(body['messages']) == 1:
        return new_chat(event, body, chat_config, tracebuf, last_msg, faiss_rms, documents_list, index_map_list,
                        searchsubdir=searchsubdir,
                        toolprompts=toolprompts)
    else:
        return ongoing_chat(event, body, chat_config, tracebuf, last_msg, faiss_rms, documents_list, index_map_list,
                            searchsubdir=searchsubdir,
                            toolprompts=toolprompts)

#
# to run this locally:
# $ cd <yoja>/server/aws
# $ (cd lambdas; docker build -t yoja-img .)
# $ docker run -v /tmp:/host-tmp --interactive --tty --entrypoint /bin/bash yoja-img
# Once in the container:
# # cp /host-tmp/credentials .
# # OAUTH_CLIENT_ID='123456789012-abcdefghijklmonpqrstuvwxyzabcdef.apps.googleusercontent.com' OAUTH_CLIENT_SECRET='GOCSPX-abcdefgh_abdefghijklmnop-qab' OAUTH_REDIRECT_URI='https://chat.example.ai/rest/entrypoint/oauth2cb' AWS_PROFILE=example AWS_DEFAULT_REGION=us-east-1 PERIOIDIC_PROCESS_FILES_TIME_LIMIT=2400 USERS_TABLE=yoja-users SERVICECONF_TABLE=yoja-ServiceConf LAMBDA_VERSION=dummy  AWS_SHARED_CREDENTIALS_FILE=./credentials OPENAI_API_KEY=sk-ABCDEFGHIJKLMONPQRSTUVWXYZabcedfghijklmnopqestuv python chat.py <email> <chat_msg>
#
if __name__=="__main__":
    if len(sys.argv) < 3:
        print(f"Usage: chat.py user_email chat_msg")
        sys.exit(255)
    os.environ['YOJA_USER'] = sys.argv[1]
    event = {'requestContext': {'requestId': 'abc', 'http': {'method': 'POST', 'path': '/rest/v1/chat/completions'}}}
    event['body'] = json.dumps({'messages': [{'content': sys.argv[2]}]})
    res = chat_completions(event, None)
    sys.exit(0)
