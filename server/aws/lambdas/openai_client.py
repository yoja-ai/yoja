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

#ASSISTANTS_MODEL="gpt-4"
ASSISTANTS_MODEL="gpt-4-1106-preview"

MAX_TOKEN_LIMIT=2048
MAX_PRE_AND_POST_TOKEN_LIMIT=256

MAX_VDB_RESULTS=1024

g_cross_encoder = None
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
def get_openai_max_token_limit():
    return MAX_TOKEN_LIMIT

def _lg(tracebuf, lgstr):
    print(lgstr)
    tracebuf.append(lgstr)

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

def _calc_tokens(context):
    global encoding_model
    return len(encoding_model.encode(context))

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
                tiktoken_count = _calc_tokens(formatted_para)
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
                    
                    tiktoken_count = _calc_tokens(formatted_para)
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

    named_entities = _extract_named_entities(last_msg)
    main_theme = _extract_main_theme(last_msg)
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


