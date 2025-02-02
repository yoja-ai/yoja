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
from index_utils import init_vdb, generate_context_sources, ContextSource
import boto3
from openai import OpenAI
import traceback_with_variables
import cryptography.fernet 
# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from typing import Tuple, List, Dict, Any, Self
import faiss_rm
import re
import dataclasses
import enum
import jsons
from searchsubdir import do_set_searchsubdir
from fetch_tool_prompts import fetch_tool_prompts
from text_utils import format_paragraph
from documentchunk import DocumentType, DocumentChunkDetails, DocumentChunkRange
from chatconfig import ChatConfiguration, RetrieverStrategyEnum
from openai_client import chat_using_openai_assistant
from ollama_client import chat_using_ollama_assistant
from yoja_retrieve import YojaIndex, get_max_token_limit
from agentic import agentic_new_chat, agentic_ongoing_chat

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

def ongoing_chat(event, body, chat_config, tracebuf, yoja_index, searchsubdir=None, toolprompts=None):
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
    srp, thread_id, run_usage = agentic_ongoing_chat(yoja_index, tracebuf,
                                                    filekey_to_file_chunks_dict, thread_id,
                                                    chat_config, body['messages'],
                                                    searchsubdir=searchsubdir, toolprompts=toolprompts)
    if not srp:
        return respond({"error_msg": "Error. retrieve using assistant failed"}, status=500)
    if run_usage:
        pct=(float(run_usage.prompt_tokens)/float(get_max_token_limit()))*100.0
        srp = srp +f"  \n**Tokens:** prompt={run_usage.prompt_tokens}({pct}% of {get_max_token_limit()}), completion={run_usage.completion_tokens}"
    
    context_srcs_links:List[ContextSource]
    context_srcs_links = generate_context_sources(filekey_to_file_chunks_dict)
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

def _debug_flags(messages) -> Tuple[ChatConfiguration, str]:
    print_trace, use_ivfadc, retriever_stratgey = \
                (False, False, RetrieverStrategyEnum.PreAndPostChunkStrategy)
    last_msg:str = messages[-1]['content']
    idx = 0
    for idx in range(len(messages[-1]['content'])):
        # '+': print_trace
        # '@': use ivfadc index
        # '^': print_trace with info about choice of context
        c = messages[-1]['content'][idx]
        if c not in ['+','@','#','$', '^', '!', '/', '~']: break
        
        if c == '+': print_trace = True
        if c == '@': use_ivfadc = not use_ivfadc
        if c == '/': retriever_stratgey = RetrieverStrategyEnum.PreAndPostChunkStrategy
    
    # strip the debug flags from the question
    if idx == len(messages[-1]['content']) - 1:
        messages[-1]['content'] = ""
    else:
        messages[-1]['content'] = messages[-1]['content'][idx:]
    chat_config = ChatConfiguration(print_trace, use_ivfadc, retriever_stratgey)
    logmsg = f"**{prtime()}: Debug Flags**: chat_config={chat_config}, last_={messages[-1]['content']}"
    print(logmsg)

    return chat_config

def new_chat(event, body, chat_config, tracebuf, yoja_index, searchsubdir=None, toolprompts=None):
    """
    documents is a dict like {fileid: finfo}; 
    index_map is a list of tuples: [(fileid, paragraph_index)];  the index into this list corresponds to the index of the embedding vector in the faiss index
    """
    print(f"new_chat: entered")
    filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]] = {}

    # string response??
    srp:str = ""; thread_id:str 
    srp, thread_id, run_usage = agentic_new_chat(yoja_index, tracebuf, filekey_to_file_chunks_dict,
                                            None, chat_config, body['messages'], searchsubdir, toolprompts)
    if not srp:
        return respond({"error_msg": "Error. chat using llm chat function failed"}, status=500)
    if run_usage:
        pct=int((float(run_usage.prompt_tokens)/float(get_max_token_limit()))*100.0)
        srp = srp +f"  \n**Tokens:** prompt={run_usage.prompt_tokens}({pct}% of {get_max_token_limit()}), completion={run_usage.completion_tokens}"

    context_srcs_links:List[ContextSource]
    context_srcs_links = generate_context_sources(filekey_to_file_chunks_dict)
    
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

    if not 'body' in event:
        return respond({"error_msg": "Error. body not present"}, status=400)
    body = json.loads(event['body'])
    print(f"body={body}")

    bucket = None
    prefix = None
    index_dir = None
    docs_dir = None
    if 'INDEX_DIR' in os.environ:
        index_dir = os.environ['INDEX_DIR']
    if 'DOCS_DIR' in os.environ:
        docs_dir = os.environ['DOCS_DIR']
    if index_dir:
        print(f"Index Location: Local directory {index_dir}")
        os.environ['YOJA_USER'] = 'notused'
        s3client = None
    else:
        if 'bucket' not in service_conf or 'prefix' not in service_conf:
            print(f"Error. index_dir not specified in env, and bucket/prefix not specified in service conf")
            return respond({"error_msg": "Error. index_dir not specified in env, and bucket/prefix not specified in service_conf"}, status=403)
        else:
            bucket = service_conf['bucket']['S']
            prefix = service_conf['prefix']['S'].strip().strip('/')
            print(f"Index Location: s3://{bucket}/{prefix}")
        s3client = boto3.client('s3')

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

    tracebuf = []
    chat_config = _debug_flags(body['messages'])

    searchsubdir = None
    faiss_rms:List[faiss_rm.FaissRM] = []
    if 'searchsubdir' in rv:
        ss1 = rv['searchsubdir']
        if ss1.startswith('[gdrive]'):
            searchsubdir = ss1[8:].lstrip('/').rstrip('/')
            faiss_rm_vdb:faiss_rm.FaissRM = init_vdb(email, index_dir, s3client, bucket, prefix,
                                                    faiss_rm.DocStorageType.GoogleDrive,
                                                    chat_config=chat_config, tracebuf=tracebuf,
                                                    build_faiss_indexes=False, sub_prefix=None)
            if faiss_rm_vdb: faiss_rms.append(faiss_rm_vdb)
        elif ss1.startswith('[dropbox]'):
            searchsubdir = ss1[9:].lstrip('/').rstrip('/')
            faiss_rm_vdb:faiss_rm.FaissRM = init_vdb(email, index_dir, s3client, bucket, prefix,
                                                    faiss_rm.DocStorageType.DropBox,
                                                    chat_config=chat_config, tracebuf=tracebuf,
                                                    build_faiss_indexes=False, sub_prefix='dropbox')
            if faiss_rm_vdb: faiss_rms.append(faiss_rm_vdb)
        elif ss1.startswith('[local]'):
            searchsubdir = ss1[7:].lstrip('/').rstrip('/')
            faiss_rm_vdb:faiss_rm.FaissRM = init_vdb(email, index_dir, s3client, bucket, prefix,
                                                    faiss_rm.DocStorageType.Local,
                                                    chat_config=chat_config, tracebuf=tracebuf,
                                                    build_faiss_indexes=False, sub_prefix=None)
            if faiss_rm_vdb: faiss_rms.append(faiss_rm_vdb)
        else:
            print(f"Error. searchsubdir does not start with [gdrive], [dropbox] or [local]. Ignoring")
    else:
        for index,doc_storage_type in ( [None, faiss_rm.DocStorageType.GoogleDrive],
                                        ['dropbox', faiss_rm.DocStorageType.DropBox],
                                        [None, faiss_rm.DocStorageType.Local] ):
            faiss_rm_vdb:faiss_rm.FaissRM = init_vdb(email, index_dir, s3client, bucket, prefix,
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
    index_type:str = 'flat'
    yoja_index = YojaIndex(faiss_rms, documents_list, index_map_list, index_type)
    if len(body['messages']) == 1:
        return new_chat(event, body, chat_config, tracebuf, yoja_index,
                        searchsubdir=searchsubdir, toolprompts=toolprompts)
    else:
        return ongoing_chat(event, body, chat_config, tracebuf, yoja_index,
                            searchsubdir=searchsubdir, toolprompts=toolprompts)

#
# to run this locally:
# $ cd <yoja>/server/aws
# $ (cd lambdas; docker build -t yoja-img .)
# $ docker run -v /tmp:/host-tmp --interactive --tty --entrypoint /bin/bash yoja-img
# Once in the container:
# # cp /host-tmp/credentials .
# # OAUTH_CLIENT_ID='123456789012-abcdefghijklmonpqrstuvwxyzabcdef.apps.googleusercontent.com' OAUTH_CLIENT_SECRET='GOCSPX-abcdefgh_abdefghijklmnop-qab' OAUTH_REDIRECT_URI='https://chat.example.ai/rest/entrypoint/oauth2cb' AWS_PROFILE=example AWS_DEFAULT_REGION=us-east-1 PERIOIDIC_PROCESS_FILES_TIME_LIMIT=2400 USERS_TABLE=yoja-users SERVICECONF_TABLE=yoja-ServiceConf LAMBDA_VERSION=dummy  AWS_SHARED_CREDENTIALS_FILE=./credentials OPENAI_API_KEY=sk-ABCDEFGHIJKLMONPQRSTUVWXYZabcedfghijklmnopqestuv python chat.py <email> <chat_msg>
#
# if you want to run this to index a local file system directory, run as follows:
# Step 1: Build the docker container using (cd lambdas; docker build -t yoja-img .)
# Step 2: Start the docker container: docker run -v /home/jagane/tmp:/host-tmp --interactive --tty --entrypoint /bin/bash yoja-img
# Step 3: In the container, run: OPENAI_API_KEY='ABCDEFGHIJLMNOPQRSTUVabcdefghijklmnopqestuvwxyz' python chat.py /host-tmp/index 'how do I descale my coffee maker?'
# Alt Step 3: In the container, run (192.168.1.100 is the IP where ollama is listening): OLLAMA_HOST='192.168.1.100' python chat.py /host-tmp/index 'how do I descale my coffee maker?'
#
if __name__=="__main__":
    if len(sys.argv) == 2:
        event = {'requestContext': {'requestId': 'abc', 'http': {'method': 'POST', 'path': '/rest/v1/chat/completions'}}}
        os.environ['YOJA_USER'] = 'notused'
        event['body'] = json.dumps({'messages': [{'content': sys.argv[1]}]})
        res = chat_completions(event, None)
        sys.exit(0)
    elif len(sys.argv) == 3:
        event = {'requestContext': {'requestId': 'abc', 'http': {'method': 'POST', 'path': '/rest/v1/chat/completions'}}}
        os.environ['YOJA_USER'] = sys.argv[1]
        event['body'] = json.dumps({'messages': [{'content': sys.argv[2]}]})
        res = chat_completions(event, None)
        sys.exit(0)
    else:
        print(f"Usage 1(run chat with local index): chat.py chat_msg")
        print(f"Usage 2(run chat with gdrive index): chat.py user_email chat_msg")
        sys.exit(255)
