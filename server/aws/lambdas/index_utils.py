import io
import json
import os
import traceback
import tempfile
import uuid
import time
import base64
import zlib
from urllib.parse import unquote
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import boto3
import sys
import base64
from botocore.exceptions import ClientError
from utils import refresh_user
from typing import Dict, List, Tuple, Any, Optional, Union
import traceback_with_variables
from dataclasses import dataclass
import datetime
import subprocess

from googleapiclient.http import MediaIoBaseDownload
from google.api_core.datetime_helpers import to_rfc3339, from_rfc3339
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.oauth2.credentials 
import google.auth.transport.requests
import googleapiclient.discovery

import docx
from pptx import Presentation

from distilbert_dotprod import MsmarcoDistilbertBaseDotProdV3
import pickle
from faiss_rm import FaissRM

import llama_index
import llama_index.core
import llama_index.readers.file
import llama_index.core.node_parser

if os.path.isdir('/var/task/sentence-transformers/msmarco-distilbert-base-dot-prod-v3'):
    vectorizer = MsmarcoDistilbertBaseDotProdV3(
            tokenizer_name_or_path='/var/task/sentence-transformers/msmarco-distilbert-base-dot-prod-v3',
            model_name_or_path='/var/task/sentence-transformers/msmarco-distilbert-base-dot-prod-v3'
        )
else:
    vectorizer = MsmarcoDistilbertBaseDotProdV3()
    
def download_gdrive_file(service, file_id, filename) -> io.BytesIO:
  """Downloads a file
  Args:
      file_id: ID of the file to download
  Returns : IO object with location.

  Load pre-authorized user credentials from the environment.
  TODO(developer) - See https://developers.google.com/identity
  for guides on implementing OAuth2 for the application.
  """
  try:
    # pylint: disable=maybe-no-member
    request = service.files().get_media(fileId=file_id)
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    done = False
    while done is False:
      status, done = downloader.next_chunk()
      print(f"filename={filename}, fileid={file_id}, download {int(status.progress() * 100)} %")
  except HttpError as error:
    print(f"An error occurred: {error}")
    file = None
  return file

def export_gdrive_file(service, file_id, mimetype) -> io.BytesIO:
  try:
    # pylint: disable=maybe-no-member
    request = service.files().export_media(fileId=file_id, mimeType=mimetype)
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    done = False
    while done is False:
      status, done = downloader.next_chunk()
      print(f"fileid={file_id}, download {int(status.progress() * 100)} %")
  except HttpError as error:
    print(f"An error occurred: {error}")
    file = None
  return file

def download_files_index(s3client, bucket, prefix, download_faiss_index):
    """ download the jsonl file that has a line for each file in the google drive; download it to /tmp/files_index.jsonl   """
    try:
        files_index_jsonl = "/tmp/files_index.jsonl"
        # index1/raj@yoja.ai/files_index.jsonl
        print(f"Downloading {files_index_jsonl}")
        s3client.download_file(bucket, f"{prefix}/{os.path.basename(files_index_jsonl)}", files_index_jsonl)
    except Exception as ex:
        print(f"Caught {ex} while downloading {files_index_jsonl}")
        return False

    if download_faiss_index:
        try:
            faiss_index_flat_fname = "/tmp/faiss_index_flat"
            print(f"Downloading {faiss_index_flat_fname}")
            # index1/raj@yoja.ai/faiss_index_flat
            s3client.download_file(bucket, f"{prefix}/{os.path.basename(faiss_index_flat_fname)}", faiss_index_flat_fname)
        except Exception as ex:
            print(f"Caught {ex} while downloading {faiss_index_flat_fname}")
            return False

        try:
            faiss_index_ivfadc_fname = "/tmp/faiss_index_ivfadc"
            print(f"Downloading {faiss_index_ivfadc_fname}")
            # index1/raj@yoja.ai/faiss_index_ivfadc
            s3client.download_file(bucket, f"{prefix}/{os.path.basename(faiss_index_ivfadc_fname)}", faiss_index_ivfadc_fname)
        except Exception as ex:
            print(f"Caught {ex} while downloading {faiss_index_ivfadc_fname}")
            return False
    
    return True

embeddings_file_cache = {}
processed_embeddings_files = []

def populate_embeddings_file_cache(embeddings_file_uri, s3client, index_bucket, index_object):
    print(f"populate_embeddings_file_cache: Entered for file {embeddings_file_uri}")
    try:
        s3client.download_file(index_bucket, index_object, "/tmp/tmp_embedding_file.jsonl")
    except Exception as ex:
        print(f"Caught {ex} while downloading {embeddings_file_uri}")
        return False
    try:
        with open("/tmp/tmp_embedding_file.jsonl", 'r') as rfp:
            for line in rfp:
                ff = json.loads(line)
                ff['mtime'] = from_rfc3339(ff['mtime'])
                fileid = ff['fileid']
                if fileid in embeddings_file_cache:
                    # multiple files in s3 may have embeddings for different versions of
                    # the same fileid. update cache only if this version is newer
                    if ff['mtime'] > embeddings_file_cache[fileid]['mtime']:
                        embeddings_file_cache[fileid] = ff
                else:
                    embeddings_file_cache[fileid] = ff
        processed_embeddings_files.append(embeddings_file_uri)
        return True
    except Exception as ex:
        print(f"Caught {ex} while parsing tmp_embeddings_file.jsonl")
        return False

def init_vdb(email, s3client, bucket, prefix, build_faiss_indexes=True) -> FaissRM :
    """ initializes a faiss vector db with the embeddings specified in bucket/prefix/files_index.jsonl.  Downloads the index from S3.  Returns a FaissRM instance which encapsulates vectorDB, metadata, documents.  Or None, if index not found in S3"""
    print(f"init_vdb: Entered. email={email}, index=s3://{bucket}/{prefix}")
    user_prefix = f"{prefix}/{email}"
    # each line in files_index.jsonl has the structure below.  In fls, it is stored as { file_id: <dict_from_each_line_of_jsonl_file>}
    # {'1S8cnVQqarbVMOnOWcixbqX_7DSMzZL3gXVbURrFNSPM': {'filename': 'Multimodal', 'fileid': '1S8cnVQqarbVMOnOWcixbqX_7DSMzZL3gXVbURrFNSPM', 'mtime': datetime.datetime(2024, 3, 4, 16, 27, 1, 169000, tzinfo=datetime.timezone.utc), 'index_bucket':'yoja-index-2', 'index_object':'index1/raj@yoja.ai/data/embeddings-1712657862202462825.jsonl'}, ... }
    fls = {}
    # if the ask is to not build the faiss indexes, then try downloading it..
    if download_files_index(s3client, bucket, user_prefix, not build_faiss_indexes):
        with open("/tmp/files_index.jsonl", "r") as rfp:
            for line in rfp:
                ff = json.loads(line)
                ff['mtime'] = from_rfc3339(ff['mtime'])
                fls[ff['fileid']] = ff
    else:
        print(f"init_vdb: Failed to download files_index.jsonl from s3://{bucket}/{user_prefix}")
        return None

    embeddings = []
    index_map = [] # list of (fileid, paragraph_index)
    # see further above for structure of fls..
    for fileid, finfo in fls.items():
        if 'slides' in finfo:
            key = 'slides'
        elif 'paragraphs' in finfo:
            key = 'paragraphs'
        else:
            continue
        for para_index in range(len(finfo[key])):
            para = finfo[key][para_index]
            embeddings.append(pickle.loads(base64.b64decode(para['embedding'].strip()))[0])
            index_map.append((fileid, para_index))
    return FaissRM(fls, index_map, embeddings, vectorizer, k=100, flat_index_fname=None if build_faiss_indexes else '/tmp/faiss_index_flat', ivfadc_index_fname=None if build_faiss_indexes else '/tmp/faiss_index_ivfadc')

def calc_file_lists(service, s3_index, gdrive_listing, folder_details) -> Tuple[dict, dict, dict]:
    """ returns a tuple of (unmodified:dict, needs_embedding:dict, deleted:dict).  Each dict has the format { fileid: {filename:abc, fileid:xyz, mtime:mno}}"""
    unmodified = {}
    deleted = {}
    # {'1S8cnVQqarbVMOnOWcixbqX_7DSMzZL3gXVbURrFNSPM': {'filename': 'Multimodal', 'fileid': '1S8cnVQqarbVMOnOWcixbqX_7DSMzZL3gXVbURrFNSPM', 'mtime': datetime.datetime(2024, 3, 4, 16, 27, 1, 169000, tzinfo=datetime.timezone.utc), 'index_bucket':'yoja-index-2', 'index_object':'index1/raj@yoja.ai/data/embeddings-1712657862202462825.jsonl'}, ... }
    needs_embedding = {}
    for fileid, gdrive_entry in gdrive_listing.items():
        if fileid in s3_index and s3_index[fileid]['mtime'] == gdrive_entry['mtime']:
            if 'partial' in s3_index[fileid]:
                needs_embedding[fileid] = s3_index[fileid]
            else:
                unmodified[fileid] = s3_index[fileid]
        else:
            path = ""
            entry = gdrive_entry
            while True:
                if not 'parents' in entry:
                    break
                folder_id = entry['parents'][0]
                if folder_id not in folder_details:
                    # try shared drives
                    try:
                        shared_drive_name = service.files().get(fileId=folder_id).execute()['name']
                        folder_details[folder_id] = {'filename': shared_drive_name}
                    except Exception as ex:
                        print(f"calc_file_lists: Exception {ex} while getting shared drive name for id {folder_id}")
                if folder_id in folder_details:
                    dir_entry = folder_details[folder_id]
                    path = dir_entry['filename'] + '/' + path
                    if 'parents' in dir_entry:
                        entry = dir_entry
                        continue
                    else:
                        break
                else:
                    print(f"WARNING: Could not map folder id {folder_id} to folder_name for filename {entry['filename']}, path={path}")
                    break
            gdrive_entry['path'] = path
            needs_embedding[fileid] = gdrive_entry
    
    # detect deleted files
    for fileid in s3_index:
        if not ( unmodified.get(fileid) or needs_embedding.get(fileid) ): 
            deleted[fileid] = s3_index[fileid]
            
    return unmodified, needs_embedding, deleted

def get_s3_index(s3client, bucket, prefix) -> Dict[str, dict]:
    """ download the jsonl that has a line for each file in the google drive; Return the dict {'1S8cnVQqarbVMOnOWcixbqX_7DSMzZL3gXVbURrFNSPM': {'filename': 'Multimodal', 'fileid': '1S8cnVQqarbVMOnOWcixbqX_7DSMzZL3gXVbURrFNSPM', 'mtime': datetime.datetime(2024, 3, 4, 16, 27, 1, 169000, tzinfo=datetime.timezone.utc), 'index_bucket':'yoja-index-2', 'index_object':'index1/raj@yoja.ai/data/embeddings-1712657862202462825.jsonl'}, paragraphs:[{embedding:xxxxx, paragraph_text|aaaa:yyyy}], slides:[{embedding:xxxx, title:abc, text=xyz}] }   """
    rv = {}
    if download_files_index(s3client, bucket, prefix, False):
        with open("/tmp/files_index.jsonl", "r") as rfp:
            for line in rfp:
                ff = json.loads(line)
                ff['mtime'] = from_rfc3339(ff['mtime'])
                rv[ff['fileid']] = ff
    else:
        print(f"get_s3_index: Failed to download files_index.jsonl from s3://{bucket}/{prefix}")
    return rv

def read_docx(filename:str, fileid:str, file_io:io.BytesIO, mtime:datetime.datetime, start_time, time_limit, prev_paras) -> Dict[str, Union[str,Dict[str, str]]]:
    doc_dct={"filename": filename, "fileid": fileid, "mtime": mtime, "paragraphs": prev_paras} # {'filename': 'module2_fall-prevention.docx', 'fileid': '11ehUfwX2Hn85qaPEcP7p-6UnYRE7IbWt', 'mtime': datetime.datetime(2024, 4, 10, 7, 49, 21, 574000, tzinfo=datetime.timezone.utc), 'paragraphs': [{'paragraph_text': 'Module 2: How To Manage Change', 'embedding': 'gASVCBsAAAAAAA...GVhLg=='}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, ...]}
    prev_len = len(prev_paras)
    doc = docx.Document(file_io)
    prelude = f"The filename is {filename} and the paragraphs are:"
    prelude_token_len = vectorizer.get_token_count(prelude)
    chunk_len = prelude_token_len
    chunk_paras = []
    for parag in doc.paragraphs:
        para = parag.text
        if para:
            para_len = vectorizer.get_token_count(para)
            if para_len + chunk_len >= 512:
                if prev_len > len(doc_dct['paragraphs']): # skip previously processed chunks
                    chunk_len = prelude_token_len
                    chunk_paras = []
                    continue
                chunk = '.'.join(chunk_paras)
                para_dct = {'paragraph_text': chunk} # {'paragraph_text': 'Module 2: How To Manage Change', 'embedding': 'gASVCBsAAAAAAA...GVhLg=='}
                try:
                    embedding = vectorizer([f"{prelude}{chunk}"])
                    eem = base64.b64encode(pickle.dumps(embedding)).decode('ascii')
                    para_dct['embedding'] = eem
                except Exception as ex:
                    print(f"Exception {ex} while creating para embedding")
                    traceback.print_exc()
                doc_dct['paragraphs'].append(para_dct)
                chunk_len = prelude_token_len
                chunk_paras = []
                now = datetime.datetime.now()
                if now - start_time > datetime.timedelta(minutes=time_limit):
                    doc_dct['partial'] = "true"
                    print(f"read_docx: More than {time_limit} minutes have passed when reading files from google drive. Breaking..")
                    break
            else:
                chunk_paras.append(para)
                chunk_len += para_len
    if chunk_len > prelude_token_len and len(chunk_paras) > 0:
        if prev_len <= len(doc_dct['paragraphs']): # skip previously processed chunks
            chunk = '.'.join(chunk_paras)
            para_dct = {} # {'paragraph_text': 'Module 2: How To Manage Change', 'embedding': 'gASVCBsAAAAAAA...GVhLg=='}
            para_dct['paragraph_text'] = chunk
            try:
                embedding = vectorizer([f"{prelude}{chunk}"])
                eem = base64.b64encode(pickle.dumps(embedding)).decode('ascii')
                para_dct['embedding'] = eem
            except Exception as ex:
                print(f"Exception {ex} while creating residual para embedding")
                traceback.print_exc()
            doc_dct['paragraphs'].append(para_dct)
    return doc_dct

def read_pptx(filename, fileid, file_io, mtime:datetime.datetime, start_time, time_limit, prev_slides) -> Dict[str, Union[str, Dict[str,str]]]:
    prs = Presentation(file_io)
    ppt={"filename": filename, "fileid": fileid, "mtime": mtime, "slides": prev_slides}  # {'filename': 'S6O4HowToSewAButton.pptx', 'fileid': '11anl03mkvhqmeOGEhX50JYnG8jwZZO8z', 'mtime': datetime.datetime(2024, 4, 10, 7, 42, 6, 750000, tzinfo=datetime.timezone.utc), 'slides': [{...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}]}
    ind = 0
    for slide in prs.slides:
        title=None
        if hasattr(slide, 'shapes') and hasattr(slide.shapes, 'title') \
                                and hasattr(slide.shapes.title, 'text'):
            title=slide.shapes.title.text
        txt=[]
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    if run.text != title:
                        txt.append(run.text)
        slide_text = ','.join(txt)
        if title:
            slide_dct = {"title": title, "text": slide_text}  # {'title': 'References', 'text': 'http://www.wikihow.com/Sew-a-Button,Video Clips,http://www.youtube.com/watch?v=Gg0pfdIRBgw,http://www.youtube.com/watch?v=hrSs_DiJ-ZA,[[Image:Sew_button_1.jpg|thumb|description]] ', 'embedding': 'gASVCBsAAAAAAABdlF2UKEe/wtm+YAAAAEc/...'}
            chu = f"The title of this slide is {title} and the content of the slide is {slide_text}"
        else:
            slide_dct = {"text": slide_text} # {'text': 'http://www.wikihow.com/Sew-a-Button,Video Clips,http://www.youtube.com/watch?v=Gg0pfdIRBgw,http://www.youtube.com/watch?v=hrSs_DiJ-ZA,[[Image:Sew_button_1.jpg|thumb|description]] ', 'embedding': 'gASVCBsAAAAAAABdlF2UKEe/wtm+YAAAAEc/...'}
            chu = f"The content of the slide is {slide_text}"
        if ind >= len(prev_slides): # skip past prev_slides
            embedding = vectorizer([chu])
            try:
                eem = base64.b64encode(pickle.dumps(embedding)).decode('ascii')
                slide_dct['embedding'] = eem
            except Exception as ex:
                print(f"Exception {ex} while creating slide embedding")
                traceback.print_exc()
            ppt['slides'].append(slide_dct)
            now = datetime.datetime.now()
            if now - start_time > datetime.timedelta(minutes=time_limit):
                ppt['partial'] = "true"
                print(f"read_pptx: More than {time_limit} minutes have passed when reading files from google drive. Breaking..")
                break
        ind += 1
    return ppt

def _read_pdf(filename:str, fileid:str, bio:io.BytesIO, mtime:datetime.datetime, start_time, time_limit, prev_paras) -> Dict[str, Any]:
    # https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/
    pdf_reader:llama_index.readers.file.PyMuPDFReader = llama_index.readers.file.PyMuPDFReader()
    # write the bytes out to a file
    with tempfile.NamedTemporaryFile('wb') as tfile:
        bio.seek(0)
        tfile.write(bio.read()); tfile.flush(); tfile.seek(0)
        # error: fitz.EmptyFileError: Cannot open empty file: filename='/tmp/tmpt6o3l__m'.
        docs:List[llama_index.core.Document] = pdf_reader.load(file_path=tfile.name)
    
    sent_split:llama_index.core.node_parser.SentenceSplitter = llama_index.core.node_parser.SentenceSplitter(chunk_size=512)
    # 
    chunks:List[str] = []
    for doc in docs:
        chunks.extend(sent_split.split_text(doc.text))

    # {'filename': 'module2_fall-prevention.docx', 'fileid': '11ehUfwX2Hn85qaPEcP7p-6UnYRE7IbWt', 'mtime': datetime.datetime(2024, 4, 10, 7, 49, 21, 574000, tzinfo=datetime.timezone.utc), 'paragraphs': [{...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, ...]}    
    doc_dct={"filename": filename, "fileid": fileid, "mtime": mtime, "paragraphs": prev_paras} 
    for ind in range(len(prev_paras), len(chunks)):
        chunk = chunks[ind]
        para_dct = {'paragraph_text':chunk} # {'paragraph_text': 'Module 2: How To Manage Change', 'embedding': 'gASVCBsAAAAAAA...GVhLg=='}
        try:
            embedding = vectorizer([f"The name of the file is {filename} and the paragraph is {chunk}"])
            eem = base64.b64encode(pickle.dumps(embedding)).decode('ascii')
            para_dct['embedding'] = eem
            doc_dct['paragraphs'].append(para_dct)
            now = datetime.datetime.now()
            if now - start_time > datetime.timedelta(minutes=time_limit):
                doc_dct['partial'] = "true"
                print(f"_read_pdf: More than {time_limit} minutes have passed when reading files from google drive. Breaking..")
                break
        except Exception as ex:
            print(f"Exception {ex} while creating para embedding")
            traceback.print_exc()
    print(f"_read_pdf: fn={filename}. returning. num paras={len(doc_dct['paragraphs'])}")
    return doc_dct

def process_docx(file_item, filename, fileid, bio, start_time, time_limit):
    if 'partial' in file_item and 'paragraphs' in file_item:
        del file_item['partial']
        prev_paras = file_item['paragraphs']
        print(f"process_docx: fn={filename}. found partial. len(prev_paras)={len(prev_paras)}")
    else:
        prev_paras = []
        print(f"process_docx: fn={filename}. did not find partial")
    doc_dict = read_docx(filename, fileid, bio, file_item['mtime'], start_time, time_limit, prev_paras)
    file_item['paragraphs'] = doc_dict['paragraphs']
    file_item['filetype'] = 'docx'
    if 'partial' in doc_dict:
        file_item['partial'] = 'true'

def process_pptx(file_item, filename, fileid, bio, start_time, time_limit):
    if 'partial' in file_item and 'slides' in file_item:
        del file_item['partial']
        prev_slides = file_item['slides']
        print(f"process_pptx: fn={filename}. found partial. len(prev_slides)={len(prev_slides)}")
    else:
        prev_slides = []
        print(f"process_pptx: fn={filename}. did not find partial")
    ppt = read_pptx(filename, fileid, bio, file_item['mtime'], start_time, time_limit, prev_slides)
    file_item['slides'] = ppt['slides']
    file_item['filetype'] = 'pptx'

def process_files(service, needs_embedding, s3client, bucket, prefix) -> Dict[str, Dict[str, Any]] : 
    """ processs the files in google drive. uses folder_id for the user, if specified. returns a dict:  { fileid: {filename, fileid, mtime} }
    'service': google drive service ; 'needs_embedding':  the list of files as a dict that needs to be embedded; s3client, bucket, prefix: location of the vector db/index file 
    Note: will stop processing files after PERIOIDIC_PROCESS_FILES_TIME_LIMIT minutes (env variable)
    """
    start_time = datetime.datetime.now()
    time_limit:int = int(os.getenv("PERIOIDIC_PROCESS_FILES_TIME_LIMIT", 9))
    done_embedding = {} # {'1S8cnVQqarbVMOnOWcixbqX_7DSMzZL3gXVbURrFNSPM': {'filename': 'Multimodal', 'fileid': '1S8cnVQqarbVMOnOWcixbqX_7DSMzZL3gXVbURrFNSPM', 'mtime': datetime.datetime(2024, 3, 4, 16, 27, 1, 169000, tzinfo=datetime.timezone.utc), 'index_bucket':'yoja-index-2', 'index_object':'index1/raj@yoja.ai/data/embeddings-1712657862202462825.jsonl'}, ... }
    for fileid, file_item in needs_embedding.items():
        # {'filename': 'Multimodal', 'fileid': '1S8cnVQqarbVMOnOWcixbqX_7DSMzZL3gXVbURrFNSPM', 'mtime': datetime.datetime(2024, 3, 4, 16, 27, 1, 169000, tzinfo=datetime.timezone.utc)}
        filename = file_item['filename']
        path = file_item['path'] if 'path' in file_item else ''
        mimetype = file_item['mimetype']
        # handle errors like these: a docx file that was deleted (in Trash) but is still visible in google drive api: raise BadZipFile("File is not a zip file")
        try:
            if filename.lower().endswith('.pptx'):
                bio:io.BytesIO = download_gdrive_file(service, fileid, filename)
                process_pptx(file_item, filename, fileid, bio, start_time, time_limit)
                done_embedding[fileid] = file_item
            elif filename.lower().endswith('.docx'):
                bio:io.BytesIO = download_gdrive_file(service, fileid, filename)
                process_docx(file_item, filename, fileid, bio, start_time, time_limit)
                done_embedding[fileid] = file_item
            elif filename.lower().endswith('.doc'):
                bio:io.BytesIO = download_gdrive_file(service, fileid, filename)
                tfd, tfn = tempfile.mkstemp(suffix=".doc", dir="/tmp")
                with os.fdopen(tfd, "wb") as wfp:
                    wfp.write(bio.getvalue())
                bio.close()
                rv = subprocess.run(['/opt/libreoffice7.6/program/soffice', '--headless', '--convert-to', 'docx', tfn, '--outdir', '/tmp'],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for line in rv.stdout.splitlines():
                    print(line.decode('utf-8'))
                os.remove(tfn)
                if rv.returncode == 0:
                    print(f"Successfully converted {filename} to docx. temp file is {tfn}, Proceeding with embedding generation")
                    with open(f"{tfn}x", 'rb') as fp:
                        process_docx(file_item, filename, fileid, fp, start_time, time_limit)
                        done_embedding[fileid] = file_item
                    os.remove(f'{tfn}x')
                else:
                    print(f"Failed to convert {filename} to docx. Return value {rv.returncode}. Not generating embeddings")
            elif filename.lower().endswith('.ppt'):
                bio:io.BytesIO = download_gdrive_file(service, fileid, filename)
                tfd, tfn = tempfile.mkstemp(suffix=".ppt", dir="/tmp")
                with os.fdopen(tfd, "wb") as wfp:
                    wfp.write(bio.getvalue())
                bio.close()
                rv = subprocess.run(['/opt/libreoffice7.6/program/soffice', '--headless', '--convert-to', 'pptx', tfn, '--outdir', '/tmp'],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for line in rv.stdout.splitlines():
                    print(line.decode('utf-8'))
                os.remove(tfn)
                if rv.returncode == 0:
                    print(f"Successfully converted {filename} to pptx. temp file is {tfn}, Proceeding with embedding generation")
                    with open(f"{tfn}x", 'rb') as fp:
                        process_pptx(file_item, filename, fileid, fp, start_time, time_limit)
                        done_embedding[fileid] = file_item
                    os.remove(f'{tfn}x')
                else:
                    print(f"Failed to convert {filename} to pptx. Return value {rv.returncode}. Not generating embeddings")
            elif filename.lower().endswith('.pdf'):
                bio:io.BytesIO = download_gdrive_file(service, fileid, filename)
                if 'partial' in file_item and 'paragraphs' in file_item:
                    del file_item['partial']
                    prev_paras = file_item['paragraphs']
                    print(f"process_files: fn={filename}. found partial. len(prev_paras)={len(prev_paras)}")
                else:
                    prev_paras = []
                    print(f"process_files: fn={filename}. did not find partial")
                pdf_dict:Dict[str,Any] = _read_pdf(filename, fileid, bio, file_item['mtime'], start_time, time_limit, prev_paras)
                file_item['paragraphs'] = pdf_dict['paragraphs']
                file_item['filetype'] = 'pdf'
                if 'partial' in pdf_dict:
                    file_item['partial'] = 'true'
                done_embedding[fileid] = file_item                    
            elif mimetype == 'application/vnd.google-apps.presentation':
                bio:io.BytesIO = export_gdrive_file(service, fileid,
                                        'application/vnd.openxmlformats-officedocument.presentationml.presentation')
                if not bio:
                    print(f"process_files: Unable to export {filename} to pptx format")
                else:
                    tfd, tfn = tempfile.mkstemp(suffix=".pptx", dir="/tmp")
                    with os.fdopen(tfd, "wb") as wfp:
                        wfp.write(bio.getvalue())
                    bio.close()
                    with open(f"{tfn}", 'rb') as fp:
                        ppt = read_pptx(filename, fileid, fp, file_item['mtime'])
                        file_item['slides'] = ppt['slides']
                        file_item['filetype'] = 'pptx'
                        done_embedding[fileid] = file_item
                    os.remove(f'{tfn}')
            elif mimetype == 'application/vnd.google-apps.document':
                bio:io.BytesIO = export_gdrive_file(service, fileid,
                                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document')
                if not bio:
                    print(f"process_files: Unable to export {filename} to docx format")
                else:
                    tfd, tfn = tempfile.mkstemp(suffix=".docx", dir="/tmp")
                    with os.fdopen(tfd, "wb") as wfp:
                        wfp.write(bio.getvalue())
                    bio.close()
                    with open(f"{tfn}", 'rb') as fp:
                        doc_dct = read_docx(filename, fileid, fp, file_item['mtime'])
                        file_item['paragraphs'] = doc_dct['paragraphs']
                        file_item['filetype'] = 'docx'
                        done_embedding[fileid] = file_item
                    os.remove(f'{tfn}')
            else:
                print(f"process_files: skipping unknown file type {filename}")
        except Exception as e:
            print(f"process_files(): skipping filename={filename} with fileid={fileid} due to exception={e}")
            traceback_with_variables.print_exc()
        now = datetime.datetime.now()
        if now - start_time > datetime.timedelta(minutes=time_limit):
            print(f"process_files: More than {time_limit} minutes have passed when reading files from google drive. Breaking..")
            break
                
    return done_embedding

def update_index_for_user(item:dict, s3client, bucket:str, prefix:str, only_create_index:bool=False):
    """ only_create_index: only create the index if it doesn't exist; do not update existing index; used when called from 'chat' since we don't want to update the index from chat """
    print(f'update_index_for_user: Entered. {item}')
    email:str = item['email']['S']
    # index1/xyz@abc.com
    user_prefix = f"{prefix}/{email}"
    folder_id = item['folder_id']['S'] if item.get('folder_id') else None
    try:
        # user_prefix = 'index1/raj@yoja.ai' 
        s3_index = get_s3_index(s3client, bucket, user_prefix)
        
        # if index already exists and ask is to not update it (only create if not found), then return.
        if s3_index and only_create_index: 
            print(f"Not updating index for {user_prefix} since index already exists and only_create_index={only_create_index} ")
            return
        
        try:
            creds:google.oauth2.credentials.Credentials = refresh_user(item)
        except Exception as ex:
            print(f"update_index_for_user: credentials not valid. not processing user {item['email']['S']}")
            return
        
        # {'1S8cnVQqarbVMOnOWcixbqX_7DSMzZL3gXVbURrFNSPM': {'filename': 'Multimodal', 'fileid': '1S8cnVQqarbVMOnOWcixbqX_7DSMzZL3gXVbURrFNSPM', 'mtime': datetime.datetime(2024, 3, 4, 16, 27, 1, 169000, tzinfo=datetime.timezone.utc)}, '1SZLHxQO0CIkRADeb0yCjuTtNh-uNy8CLT7HR4kuvZks': {'filename': 'Q&A', 'fileid': '1SZLHxQO0CIkRADeb0yCjuTtNh-uNy8CLT7HR4kuvZks', 'mtime': datetime.datetime(2024, 3, 4, 16, 26, 51, 962000, tzinfo=datetime.timezone.utc)}, '19HNNCPaO-NGGCMHqTUtRvYOtFe-vG61ltTUMbQgHJ9Y': {'filename': 'Extraction', 'fileid': '19HNNCPaO-NGGCMHqTUtRvYOtFe-vG61ltTUMbQgHJ9Y', 'mtime': datetime.datetime(2024, 3, 4, 16, 26, 36, 440000, tzinfo=datetime.timezone.utc)}, '11cZc_vN-5NUBoW3XZYKUjfwEarmOPiOKbN5Xvc1U3pA': {'filename': 'Chatbots', 'fileid': '11cZc_vN-5NUBoW3XZYKUjfwEarmOPiOKbN5Xvc1U3pA', 'mtime': datetime.datetime(2024, 3, 4, 16, 26, 27, 281000, tzinfo=datetime.timezone.utc)}, '158NNowMCrbTFd_28OnLBwO6GZJ50NwaBa9seHQsVaWY': {'filename': 'Agents', 'fileid': '158NNowMCrbTFd_28OnLBwO6GZJ50NwaBa9seHQsVaWY', 'mtime': datetime.datetime(2024, 3, 4, 16, 26, 16, 596000, tzinfo=datetime.timezone.utc)}}
        gdrive_listing = {}
        service:googleapiclient.discovery.Resource = build("drive", "v3", credentials=creds)
        pageToken = None
        kwargs = {}
        folder_details = {}
        try:
            driveid = service.files().get(fileId='root').execute()['id']
            folder_details[driveid] = {'filename': 'My Drive'}
        except Exception as ex:
            print(f"update_index_for_user: Exception {ex} while getting driveid")
        # folder_id is specified, then only index the folder's contents
        if folder_id: kwargs['q'] = "'" + folder_id + "' in parents"
        while True:
            # Metadata for files: https://developers.google.com/drive/api/reference/rest/v3/files
            results = (
                service.files()
                .list(pageSize=100, fields="nextPageToken, files(id, name, size, modifiedTime, mimeType, parents)", **kwargs)
                .execute()
            )
            # [{'id': '1S8cnVQqarbVMOnOWcixbqX_7DSMzZL3gXVbURrFNSPM', 'name': 'Multimodal', 'modifiedTime': '2024-03-04T16:27:01.169Z'}, {'id': '1SZLHxQO0CIkRADeb0yCjuTtNh-uNy8CLT7HR4kuvZks', 'name': 'Q&A', 'modifiedTime': '2024-03-04T16:26:51.962Z'}, {'id': '19HNNCPaO-NGGCMHqTUtRvYOtFe-vG61ltTUMbQgHJ9Y', 'name': 'Extraction', 'modifiedTime': '2024-03-04T16:26:36.440Z'}, {'id': '11cZc_vN-5NUBoW3XZYKUjfwEarmOPiOKbN5Xvc1U3pA', 'name': 'Chatbots', 'modifiedTime': '2024-03-04T16:26:27.281Z'}, {'id': '158NNowMCrbTFd_28OnLBwO6GZJ50NwaBa9seHQsVaWY', 'name': 'Agents', 'modifiedTime': '2024-03-04T16:26:16.596Z'}]
            items = results.get("files", [])
            if not items:
                print("No more files found.")
                return
            for item in items:
                filename = item['name']
                fileid = item['id']
                if 'size' in item:
                    size = int(item['size'])
                else:
                    size = 0
                mtime = from_rfc3339(item['modifiedTime'])
                mimetype = item['mimeType']
                if mimetype == 'application/vnd.google-apps.folder':
                    if 'parents' in item:
                        folder_details[fileid] = {'filename': filename, 'parents': item['parents']}
                else:
                    if 'parents' in item:
                        gdrive_listing[fileid] = {"filename": filename, "fileid": fileid, "mtime": mtime, 'mimetype': mimetype, 'size': size, 'parents': item['parents']}
                    else:
                        gdrive_listing[fileid] = {"filename": filename, "fileid": fileid, "mtime": mtime, 'mimetype': mimetype, 'size': size}
            pageToken = results.get("nextPageToken", None)
            kwargs['pageToken']=pageToken
            if not pageToken:
                break

        unmodified:dict; needs_embedding:dict;  deleted_files: dict
        unmodified, needs_embedding, deleted_files = calc_file_lists(service, s3_index, gdrive_listing, folder_details)
        print(f"Number of unmodified={len(unmodified)}; modified or added={len(needs_embedding)}; deleted={len(deleted_files)}; ")

        # remove deleted files from the index
        for fileid in deleted_files: s3_index.pop(fileid)

        done_embedding = process_files(service, needs_embedding, s3client, bucket, user_prefix)
        add_or_updated_num = len(done_embedding.items())
        deleted_num = len(deleted_files)
        if add_or_updated_num > 0 or deleted_num > 0:
            print(f"update_index_for_user: Updating files_index.jsonl with new embeddings for {add_or_updated_num} files and removing deleted files={deleted_num}")
            # consolidate unmodified and done_embedding
            for fileid, file_item in done_embedding.items():
                unmodified[fileid] = file_item
            unsorted_all_files = []
            for fileid, file_item in unmodified.items():
                unsorted_all_files.append(file_item)
            sorted_all_files = sorted(unsorted_all_files, key=lambda x: x['mtime'], reverse=True)
            with open("/tmp/files_index.jsonl", "w") as wfp:
                for file_item in sorted_all_files:
                    file_item['mtime'] = to_rfc3339(file_item['mtime'])
                    wfp.write(json.dumps(file_item) + "\n")
            files_index_jsonl = '/tmp/files_index.jsonl'
            print(f"Uploading  {files_index_jsonl}  Size of file={os.path.getsize(files_index_jsonl)}")
            s3client.upload_file(files_index_jsonl, bucket, f"{user_prefix}/{os.path.basename(files_index_jsonl)}")
            
            # now build the index.
            faiss_rm:FaissRM = init_vdb(email, s3client, bucket, prefix)
            
            # save the created index
            faiss_flat_fname = '/tmp/faiss_index_flat'
            faiss.write_index(faiss_rm.get_index_flat(), faiss_flat_fname)
            print(f"Uploading faiss flat index {faiss_flat_fname}  Size of index={os.path.getsize(faiss_flat_fname)}")
            s3client.upload_file(faiss_flat_fname, bucket, f"{user_prefix}/{os.path.basename(faiss_flat_fname)}")
            os.remove(faiss_flat_fname)
                
            # save the created index
            faiss_ivfadc_fname = '/tmp/faiss_index_ivfadc'
            faiss.write_index(faiss_rm.get_index_ivfadc(), faiss_ivfadc_fname)
            print(f"Uploading faiss ivfadc index {faiss_ivfadc_fname}.  Size of index={os.path.getsize(faiss_ivfadc_fname)}")
            s3client.upload_file(faiss_ivfadc_fname, bucket, f"{user_prefix}/{os.path.basename(faiss_ivfadc_fname)}")
            os.remove(faiss_ivfadc_fname)
        else:
            print(f"update_index_for_user: No new embeddings or deleted files. Not updating files_index.jsonl")
    except HttpError as error:
        print(f"HttpError occurred: {error}")
        traceback.print_exc()
    except Exception as ex:
        print(f"Exception occurred: {ex}")
        traceback.print_exc()

if __name__ == '__main__':
    files = [ '/home/dev/simple_memo.pdf', '/home/dev/tp_link_deco_e4_wifi_mesh_datasheet_Deco_E4_EU_US_3.pdf' ]
    with open(files[1], "rb") as f:
        bio = io.BytesIO(f.read())
    print(_read_pdf("simple_memo.pdf", "file_id_for_simple_memo.pdf", bio, datetime.datetime.now() ))
