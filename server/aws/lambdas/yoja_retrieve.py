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
from text_utils import format_paragraph
from sentence_transformers.cross_encoder import CrossEncoder

MAX_TOKEN_LIMIT=2048
MAX_PRE_AND_POST_TOKEN_LIMIT=256
MAX_VDB_RESULTS=1024

g_cross_encoder = None

def get_max_token_limit():
    return MAX_TOKEN_LIMIT

def _lg(tracebuf, lgstr):
    print(lgstr)
    tracebuf.append(lgstr)

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

def _calc_cross_sorted_diffs(cross_sorted_scores):
    rv = []
    for ind in range(1, len(cross_sorted_scores)):
        prev_chunk = cross_sorted_scores[ind-1]
        chunk = cross_sorted_scores[ind]
        dif = prev_chunk.cross_encoder_score - chunk.cross_encoder_score
        rv.append((ind, dif))
    return rv

def _retrieve_rerank(faiss_rms:List[faiss_rm.FaissRM], documents_list:List[Dict[str,str]], index_map_list:List[Tuple[str,str]],
                                    index_type, tracebuf:List[str], filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]],
                                    chat_config:ChatConfiguration, last_msg:str, searchsubdir:str=None,
                                    extract_main_theme=None, extract_named_entities=None) -> Tuple[np.ndarray, List[DocumentChunkDetails]]:
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

    named_entities = extract_named_entities(last_msg) if extract_named_entities else None
    main_theme = extract_main_theme(last_msg) if extract_main_theme else None
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

class YojaIndex:
    def __init__(self, faiss_rms, documents_list, index_map_list, index_type):
        self.faiss_rms = faiss_rms
        self.documents_list = documents_list
        self.index_map_list = index_map_list
        self.index_type = index_type

def get_context(yoja_index, tracebuf:List[str], filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]],
                chat_config:ChatConfiguration, last_msg:str, most_relevant_only:bool,
                least_relevant_only:bool, searchsubdir:str=None, calc_tokens=None,
                extract_main_theme=None, extract_named_entities=None):
    """
    documents is a dict like {fileid: finfo}; 
    index_map is a list of tuples: [(fileid, paragraph_index)];  the index into this list corresponds to the index of the embedding vector in the faiss index
    
    the context to be sent to the LLM.  Does a similarity search in faiss to fetch the context
    """
    faiss_rms = yoja_index.faiss_rms
    documents_list = yoja_index.documents_list
    index_map_list = yoja_index.index_map_list
    index_type = yoja_index.index_type

    cross_sorted_scores:List[DocumentChunkDetails]
    status, cross_sorted_scores = _retrieve_rerank(faiss_rms, documents_list, index_map_list, index_type, tracebuf,
                            filekey_to_file_chunks_dict, chat_config, last_msg, searchsubdir, extract_main_theme, extract_named_entities)
    if not status:
        print(f"get_context: no context from _retrieve_rerank")
        return "No context found"

    cross_sorted_scores = cross_sorted_scores[:16]
    # if most_relevant_only and least_relevant_only are both False, return all
    if len(cross_sorted_scores) > 1:
        if most_relevant_only:
            cross_sorted_scores = _truncate_cross_sorted_scores(cross_sorted_scores, True)
        elif least_relevant_only:
            cross_sorted_scores = _truncate_cross_sorted_scores(cross_sorted_scores, False)
    _lg(tracebuf, f"{prtime()}get_context: length of truncated cross_sorted_scores = {len(cross_sorted_scores)}")

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
        _lg(tracebuf, f"{prtime()}get_context: Processing {finfo['path']}{finfo['filename']},para={chunk_det.para_id}")

        key = _get_key(finfo)
        if not key:
            emsg = f"ERROR! Could not get key in document for {finfo}"
            print(emsg)
            return respond({"error_msg": emsg}, status=500)
        
        context_chunk_range_list.append(chunk_range)

        if chunk_det.file_type == DocumentType.GH_ISSUES_ZIP:
            chunk_range.start_para_id = chunk_det.para_id
            chunk_range.end_para_id = chunk_det.para_id
            _lg(tracebuf, f"{prtime()}get_context: gh issues zip file. Not adding previous or next paragraphs for context")
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
            _lg(tracebuf, f"{prtime()}get_context: including prior chunks upto {idx} for {chunk_det.file_name} hit para_number={chunk_det.para_id}")

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
                _lg(tracebuf, f"{prtime()}get_context: including posterior chunks upto {idx} for {chunk_det.file_name} hit para_number={chunk_det.para_id}")
            
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

def get_filelist_using_retr_and_rerank(yoja_index, tracebuf:List[str], filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]],
                                         chat_config:ChatConfiguration, last_msg:str, number_of_files:int = 10, searchsubdir:str=None):
    faiss_rms = yoja_index.faiss_rms
    documents_list = yoja_index.documents_list
    index_map_list = yoja_index.index_map_list
    index_type = yoja_index.index_type

    cross_sorted_scores:List[DocumentChunkDetails]
    status, cross_sorted_scores = _retrieve_rerank(faiss_rms, documents_list, index_map_list, index_type, tracebuf,
                            filekey_to_file_chunks_dict, chat_config, last_msg, searchsubdir)
    if not status:
        print(f"get_filelist_using_retr_and_rerank: no context from _retrieve_rerank")
        return "No context files found"
    files_dict:Dict[str,DocumentChunkDetails] = {}
    for chunk_det in cross_sorted_scores:
        if not files_dict.get(chunk_det._get_file_key()):
            files_dict[chunk_det._get_file_key()] = chunk_det
            msg = f"{prtime()}: adding {chunk_det.file_path}{chunk_det.file_name} to listing"
            print(msg)
            tracebuf.append(msg)
        if len(files_dict) >= number_of_files: break
    return ",".join([  f"[{val.file_path}/{val.file_name}]({val.file_path}/{val.file_name})" for key, val in files_dict.items() ])

