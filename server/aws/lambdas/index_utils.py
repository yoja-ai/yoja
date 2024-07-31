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
from utils import refresh_user_google, refresh_user_dropbox, lambda_timelimit_exceeded, lambda_time_left_seconds, set_start_time
from typing import Dict, List, Tuple, Any, Optional, Union
import traceback_with_variables
from dataclasses import dataclass
import datetime
from datetime import timezone
import subprocess
import dropbox
import jsons
import gzip
from pdf import read_pdf
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object

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
from faiss_rm import FaissRM, DocStorageType

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
    
def lock_user(item, client):
    email=item['email']['S']
    print(f"lock_user: Entered. Trying to lock for email={email}")
    gdrive_next_page_token = None
    dropbox_next_page_token = None
    try:
        now = time.time()
        now_s = datetime.datetime.fromtimestamp(now).strftime('%Y-%m-%d %I:%M:%S')
        if not 'lambda_end_time' in item:
            print(f"lock_user: no lambda_end_time in ddb. now={now}/{now_s}")
        else:
            l_e_t=int(item['lambda_end_time']['N'])
            l_e_t_s = datetime.datetime.fromtimestamp(l_e_t).strftime('%Y-%m-%d %I:%M:%S')
            print(f"lock_user: lambda_end_time in ddb={l_e_t}/{l_e_t_s}, now={now}/{now_s}")
        response = client.update_item(
            TableName=os.environ['USERS_TABLE'],
            Key={'email': {'S': email}},
            UpdateExpression="set #lm = :st",
            ConditionExpression=f"attribute_not_exists(#lm) OR #lm < :nw",
            ExpressionAttributeNames={'#lm': 'lambda_end_time'},
            ExpressionAttributeValues={':nw': {'N': str(int(now))}, ':st': {'N': str(int(now)+(15*60))} },
            ReturnValues="ALL_NEW"
        )
        if 'Attributes' in response:
            if 'gdrive_next_page_token' in response['Attributes']:
                gdrive_next_page_token = response['Attributes']['gdrive_next_page_token']['S']
            if 'dropbox_next_page_token' in response['Attributes']:
                dropbox_next_page_token = response['Attributes']['dropbox_next_page_token']['S']
        print(f"lock_user: conditional check success. No other instance of lambda is active for user {email}. "
                f"gdrive_next_page_token={gdrive_next_page_token}, dropbox_next_page_token={dropbox_next_page_token}")
        return gdrive_next_page_token, dropbox_next_page_token, True
    except ClientError as e:
        if e.response['Error']['Code'] == "ConditionalCheckFailedException":
            print(f"lock_user: conditional check failed. {e.response['Error']['Message']}. Another instance of lambda is active for {email}")
            return gdrive_next_page_token, dropbox_next_page_token, False
        else:
            raise

def unlock_user(item, client, gdrive_next_page_token, dropbox_next_page_token):
    email=item['email']['S']
    print(f"unlock_user: Entered. email={email}")
    try:
        if gdrive_next_page_token and dropbox_next_page_token:
            ue="SET gdrive_next_page_token = :pt, dropbox_next_page_token = :db"
            eav={':pt': {'S': gdrive_next_page_token}, ':db': {'S': dropbox_next_page_token}}
            response = client.update_item(
                TableName=os.environ['USERS_TABLE'],
                Key={'email': {'S': email}},
                UpdateExpression=ue, ExpressionAttributeValues=eav,
                ReturnValues="UPDATED_NEW"
            )
        elif gdrive_next_page_token and not dropbox_next_page_token:
            ue="SET gdrive_next_page_token = :pt REMOVE dropbox_next_page_token"
            eav={':pt': {'S': gdrive_next_page_token}}
            response = client.update_item(
                TableName=os.environ['USERS_TABLE'],
                Key={'email': {'S': email}},
                UpdateExpression=ue, ExpressionAttributeValues=eav,
                ReturnValues="UPDATED_NEW"
            )
        elif not gdrive_next_page_token and dropbox_next_page_token:
            ue="REMOVE gdrive_next_page_token SET dropbox_next_page_token = :db"
            eav={':db': {'S': dropbox_next_page_token}}
            response = client.update_item(
                TableName=os.environ['USERS_TABLE'],
                Key={'email': {'S': email}},
                UpdateExpression=ue, ExpressionAttributeValues=eav,
                ReturnValues="UPDATED_NEW"
            )
        else:
            ue = "REMOVE gdrive_next_page_token, dropbox_next_page_token"
            response = client.update_item(
                TableName=os.environ['USERS_TABLE'],
                Key={'email': {'S': email}},
                UpdateExpression=ue,
                ReturnValues="UPDATED_NEW"
            )
    except ClientError as e:
        print(f"unlock_user: Error {e.response['Error']['Message']} while unlocking")
        return False
    return True

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
    print(f"download_gdrive_file: An error occurred: {error}")
    return None
  file.seek(0, os.SEEK_SET)
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

@dataclass
class IndexMetadata:
    jsonl_last_modified:float  = 0 # time.time()
    index_flat_last_modified:float = 0
    index_ivfadc_last_modified: float = 0
    
    def is_vdb_index_stale(self):
        return self.jsonl_last_modified > self.index_flat_last_modified or self.jsonl_last_modified > self.index_ivfadc_last_modified

FILES_INDEX_JSONL = "/tmp/files_index.jsonl"    
FILES_INDEX_JSONL_GZ = "/tmp/files_index.jsonl.gz"
INDEX_METADATA_JSON = "/tmp/index_metadata.json"
FAISS_INDEX_FLAT = "/tmp/faiss_index_flat"
FAISS_INDEX_IVFADC = "/tmp/faiss_index_ivfadc"
def download_files_index(s3client, bucket, prefix, download_faiss_index) -> bool:
    """ download the jsonl file that has a line for each file in the google drive; download it to /tmp/files_index.jsonl   
        download_faiss_index:  download the built faiss indexes to local storage (in addition to above jsonl file)
        return True if all files can be downloaded.  Return False if any of the files cannot be downloaded.  
        Note that files_index.jsonl.gz may exist but faiss_index_flat and faiss_index_ivfadc may not exist since the latter is sometimes built only after all the document embeddings are generated. """
    try:
        files_index_jsonl_gz = FILES_INDEX_JSONL_GZ
        # index1/raj@yoja.ai/files_index.jsonl.gz
        print(f"Downloading {files_index_jsonl_gz} from s3://{bucket}/{prefix}")
        s3client.download_file(bucket, f"{prefix}/{os.path.basename(files_index_jsonl_gz)}", files_index_jsonl_gz)
    except Exception as ex:
        print(f"Caught {ex} while downloading {files_index_jsonl_gz} from s3://{bucket}/{prefix}")
        return False

    try:
        index_metadata_fname = INDEX_METADATA_JSON
        print(f"Downloading {index_metadata_fname} from s3://{bucket}/{prefix}")
        s3client.download_file(bucket, f"{prefix}/{os.path.basename(index_metadata_fname)}", index_metadata_fname)
    except Exception as ex:
        print(f"Caught {ex} while downloading {index_metadata_fname}.  Creating an empty {index_metadata_fname}")
        # create an empty metadata file
        _create_index_metadata_json_local(IndexMetadata())

    if download_faiss_index:
        try:
            faiss_index_flat_fname = FAISS_INDEX_FLAT
            print(f"Downloading {faiss_index_flat_fname} from s3://{bucket}/{prefix}")
            # index1/raj@yoja.ai/faiss_index_flat
            s3client.download_file(bucket, f"{prefix}/{os.path.basename(faiss_index_flat_fname)}", faiss_index_flat_fname)
        except Exception as ex:
            print(f"Caught {ex} while downloading {faiss_index_flat_fname} from s3://{bucket}/{prefix}")
            return False

        try:
            faiss_index_ivfadc_fname = FAISS_INDEX_IVFADC
            print(f"Downloading {faiss_index_ivfadc_fname} from s3://{bucket}/{prefix}")
            # index1/raj@yoja.ai/faiss_index_ivfadc
            s3client.download_file(bucket, f"{prefix}/{os.path.basename(faiss_index_ivfadc_fname)}", faiss_index_ivfadc_fname)
        except Exception as ex:
            print(f"Caught {ex} while downloading {faiss_index_ivfadc_fname} from s3://{bucket}/{prefix}")
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

def init_vdb(email, s3client, bucket, prefix, doc_storage_type:DocStorageType, build_faiss_indexes=True, sub_prefix=None) -> FaissRM :
    """ initializes a faiss vector db with the embeddings specified in bucket/prefix/files_index.jsonl.  Downloads the index from S3.  Returns a FaissRM instance which encapsulates vectorDB, metadata, documents.  Or None, if index not found in S3
    sub_prefix: specify subfolder under which index must be downloaded from.  If not specified, ignored.
    """
    print(f"init_vdb: Entered. email={email}, index=s3://{bucket}/{prefix}; sub_prefix={sub_prefix}")
    user_prefix = f"{prefix}/{email}" + f"{'/' + sub_prefix if sub_prefix else ''}"
    fls = {}
    # if the ask is to not build the faiss indexes, then try downloading it..
    if download_files_index(s3client, bucket, user_prefix, not build_faiss_indexes):
        with gzip.open(FILES_INDEX_JSONL_GZ, "r") as rfp:
            for line in rfp:
                ff = json.loads(line)
                ff['mtime'] = from_rfc3339(ff['mtime'])
                fls[ff['fileid']] = ff
    else:
        print(f"init_vdb: Failed to download files_index.jsonl from s3://{bucket}/{user_prefix}")
        return None
    print(f"init_vdb: finished reading files_index.jsonl.gz. Entries in fls dict={len(fls.items())}")

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
    print(f"init_vdb: finished loading embeddings/index_map. Entries in embeddings={len(embeddings)}")

    return FaissRM(fls, index_map, embeddings, vectorizer, doc_storage_type, k=100, flat_index_fname=None if build_faiss_indexes else FAISS_INDEX_FLAT, ivfadc_index_fname=None if build_faiss_indexes else FAISS_INDEX_IVFADC)

def _calc_path(service, entry, folder_details):
    # calculate path by walking up parents
    path = ""
    while True:
        if not 'parents' in entry:
            break
        folder_id = entry['parents'][0]
        if folder_id not in folder_details:
            try:
                res = service.files().get(fileId=folder_id, fields='name,parents').execute()
                if 'parents' in res:
                    folder_details[folder_id] = {'filename': res['name'], 'parents': res['parents']}
                else:
                    folder_details[folder_id] = {'filename': res['name']}
            except Exception as ex:
                print(f"_calc_path: Exception {ex} while getting shared drive name for id {folder_id}")
        if folder_id in folder_details:
            dir_entry = folder_details[folder_id]
            path = dir_entry['filename'] + '/' + path
            if 'parents' in dir_entry:
                entry = dir_entry
                continue
            else:
                break
        else:
            print(f"_calc_path: WARNING: Could not map folder id {folder_id} to folder_name for filename {entry['filename']}, path={path}")
            break
    return path

def calc_file_lists(service, s3_index, gdrive_listing, folder_details) -> Tuple[dict, dict, dict]:
    """ returns a tuple of (unmodified:dict, needs_embedding:dict, deleted:dict).  Each dict has the format { fileid: {filename:abc, fileid:xyz, mtime:mno}}"""
    unmodified = {}
    deleted = {}
    needs_embedding = {}
    for fileid, gdrive_entry in gdrive_listing.items():
        if fileid in s3_index and s3_index[fileid]['mtime'] == gdrive_entry['mtime']:
            if 'partial' in s3_index[fileid]:
                needs_embedding[fileid] = s3_index[fileid]
            else:
                unmodified[fileid] = s3_index[fileid]
        else:
            path=_calc_path(service, gdrive_entry, folder_details)
            gdrive_entry['path'] = path
            needs_embedding[fileid] = gdrive_entry
    
    # detect deleted files
    for fileid in s3_index:
        if not ( unmodified.get(fileid) or needs_embedding.get(fileid) ): 
            deleted[fileid] = s3_index[fileid]
            
    return unmodified, needs_embedding, deleted

def get_s3_index(s3client, bucket, prefix) -> Dict[str, dict]:
    rv = {}
    if download_files_index(s3client, bucket, prefix, False):
        with gzip.open(FILES_INDEX_JSONL_GZ, "rb") as rfp:
            for line in rfp:
                ff = json.loads(line)
                ff['mtime'] = from_rfc3339(ff['mtime'])
                rv[ff['fileid']] = ff
    else:
        print(f"get_s3_index: Failed to download files_index.jsonl from s3://{bucket}/{prefix}")
    return rv


def read_docx(filename:str, fileid:str, file_io:io.BytesIO, mtime:datetime.datetime, prev_paras) -> Dict[str, Union[str,Dict[str, str]]]:
    doc_dct={"filename": filename, "fileid": fileid, "mtime": mtime, "paragraphs": prev_paras}
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
                if lambda_timelimit_exceeded():
                    doc_dct['partial'] = "true"
                    print(f"read_docx: Lambda timelimit exceeded when reading docx file. Breaking..")
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

def read_pptx(filename, fileid, file_io, mtime:datetime.datetime, prev_slides) -> Dict[str, Union[str, Dict[str,str]]]:
    prs = Presentation(file_io)
    ppt={"filename": filename, "fileid": fileid, "mtime": mtime, "slides": prev_slides}
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
            slide_dct = {"title": title, "text": slide_text}
            chu = f"The title of this slide is {title} and the content of the slide is {slide_text}"
        else:
            slide_dct = {"text": slide_text}
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
            if lambda_timelimit_exceeded():
                ppt['partial'] = "true"
                print(f"read_pptx: Lambda timelimit exceeded reading pptx file. Breaking..")
                break
        ind += 1
    return ppt

def process_docx(file_item, filename, fileid, bio):
    if 'partial' in file_item and 'paragraphs' in file_item:
        del file_item['partial']
        prev_paras = file_item['paragraphs']
        print(f"process_docx: fn={filename}. found partial. len(prev_paras)={len(prev_paras)}")
    else:
        prev_paras = []
        print(f"process_docx: fn={filename}. did not find partial")
    doc_dict = read_docx(filename, fileid, bio, file_item['mtime'], prev_paras)
    file_item['paragraphs'] = doc_dict['paragraphs']
    file_item['filetype'] = 'docx'
    if 'partial' in doc_dict:
        file_item['partial'] = 'true'

def process_pptx(file_item, filename, fileid, bio):
    if 'partial' in file_item and 'slides' in file_item:
        del file_item['partial']
        prev_slides = file_item['slides']
        print(f"process_pptx: fn={filename}. found partial. len(prev_slides)={len(prev_slides)}")
    else:
        prev_slides = []
        print(f"process_pptx: fn={filename}. did not find partial")
    ppt = read_pptx(filename, fileid, bio, file_item['mtime'], prev_slides)
    file_item['slides'] = ppt['slides']
    file_item['filetype'] = 'pptx'

class StorageReader:
    def read(self, fileid, filename, mimetype):
        pass

class GoogleDriveReader(StorageReader):
    def __init__(self, service):
        self._service = service
    def read(self, fileid, filename, mimetype):
        print(f"GoogleDriveReader.read: Entered. fileid={fileid}, filename={filename}, mimetype={mimetype}")
        if mimetype == 'application/vnd.google-apps.presentation':
            print(f"GoogleDriveReader.read: fileid={fileid} calling export1")
            return export_gdrive_file(self._service, fileid,
                        'application/vnd.openxmlformats-officedocument.presentationml.presentation')
        elif mimetype == 'application/vnd.google-apps.document':
            print(f"GoogleDriveReader.read: fileid={fileid} calling export2")
            return export_gdrive_file(self._service, fileid,
                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document')
        else:
            print(f"GoogleDriveReader.read: fileid={fileid} calling download")
            return download_gdrive_file(self._service, fileid, filename)

class DropboxReader(StorageReader):
    def __init__(self, token):
        self._token = token
    def read(self, fileid, filename, mimetype):
        print(f"DropboxReader.read: Entered. fileid={fileid}, filename={filename}")
        url = 'https://content.dropboxapi.com/2/files/download'
        headers = {'Authorization': f"Bearer {self._token}", 'Dropbox-API-Arg': json.dumps({'path': fileid})}
        r = requests.get(url, stream=True, headers=headers)
        file = io.BytesIO()
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk:
                file.write(chunk)
        file.seek(0, os.SEEK_SET)
        return file

def process_files(storage_reader:StorageReader, needs_embedding, s3client, bucket, prefix) -> Tuple[Dict[str, Dict[str, Any]], bool] : 
    """ processs the files in google drive. uses folder_id for the user, if specified. returns a dict:  { fileid: {filename, fileid, mtime} }
    'service': google drive service ; 'needs_embedding':  the list of files as a dict that needs to be embedded; s3client, bucket, prefix: location of the vector db/index file 
    Note: will stop processing files after PERIOIDIC_PROCESS_FILES_TIME_LIMIT minutes (env variable)
    """
    time_limit_exceeded = False
    done_embedding = {}
    for fileid, file_item in needs_embedding.items():
        filename = file_item['filename']
        path = file_item['path'] if 'path' in file_item else ''
        mimetype = file_item['mimetype'] if 'mimetype' in file_item else ''
        
        # handle errors like these: a docx file that was deleted (in Trash) but is still visible in google drive api: raise BadZipFile("File is not a zip file")
        try:
            if 'size' in file_item and int(file_item['size']) == 0: 
                print(f"skipping google drive with file size == 0: {file_item}")
                continue
            
            if filename.lower().endswith('.pptx'):
                bio:io.BytesIO = storage_reader.read(fileid, filename, mimetype)
                process_pptx(file_item, filename, fileid, bio)
                done_embedding[fileid] = file_item
            elif filename.lower().endswith('.docx'):
                bio:io.BytesIO = storage_reader.read(fileid, filename, mimetype)
                process_docx(file_item, filename, fileid, bio)
                done_embedding[fileid] = file_item
            elif filename.lower().endswith('.doc'):
                bio:io.BytesIO = storage_reader.read(fileid, filename, mimetype)
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
                        process_docx(file_item, filename, fileid, fp)
                        done_embedding[fileid] = file_item
                    os.remove(f'{tfn}x')
                else:
                    print(f"Failed to convert {filename} to docx. Return value {rv.returncode}. Not generating embeddings")
            elif filename.lower().endswith('.ppt'):
                bio:io.BytesIO = storage_reader.read(fileid, filename, mimetype)
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
                        process_pptx(file_item, filename, fileid, fp)
                        done_embedding[fileid] = file_item
                    os.remove(f'{tfn}x')
                else:
                    print(f"Failed to convert {filename} to pptx. Return value {rv.returncode}. Not generating embeddings")
            elif filename.lower().endswith('.pdf'):
                bio:io.BytesIO = storage_reader.read(fileid, filename, mimetype)
                if 'partial' in file_item and 'paragraphs' in file_item:
                    del file_item['partial']
                    prev_paras = file_item['paragraphs']
                    print(f"process_files: fn={filename}. found partial. len(prev_paras)={len(prev_paras)}")
                else:
                    prev_paras = []
                    print(f"process_files: fn={filename}. did not find partial")
                pdf_dict:Dict[str,Any] = read_pdf(filename, fileid, bio, file_item['mtime'], vectorizer, prev_paras)
                file_item['paragraphs'] = pdf_dict['paragraphs']
                file_item['filetype'] = 'pdf'
                if 'partial' in pdf_dict:
                    file_item['partial'] = 'true'
                done_embedding[fileid] = file_item                    
            elif mimetype == 'application/vnd.google-apps.presentation':
                bio:io.BytesIO = storage_reader.read(fileid, filename, mimetype)
                if not bio:
                    print(f"process_files: Unable to export {filename} to pptx format")
                else:
                    tfd, tfn = tempfile.mkstemp(suffix=".pptx", dir="/tmp")
                    with os.fdopen(tfd, "wb") as wfp:
                        wfp.write(bio.getvalue())
                    bio.close()
                    with open(f"{tfn}", 'rb') as fp:
                        process_pptx(file_item, filename, fileid, fp)
                        done_embedding[fileid] = file_item
                    os.remove(f'{tfn}')
            elif mimetype == 'application/vnd.google-apps.document':
                bio:io.BytesIO = storage_reader.read(fileid, filename, mimetype)
                if not bio:
                    print(f"process_files: Unable to export {filename} to docx format")
                else:
                    tfd, tfn = tempfile.mkstemp(suffix=".docx", dir="/tmp")
                    with os.fdopen(tfd, "wb") as wfp:
                        wfp.write(bio.getvalue())
                    bio.close()
                    with open(f"{tfn}", 'rb') as fp:
                        process_docx(file_item, filename, fileid, fp)
                        done_embedding[fileid] = file_item
                    os.remove(f'{tfn}')
            else:
                print(f"process_files: skipping unknown file type {filename}")
        except Exception as e:
            print(f"process_files(): skipping filename={filename} with fileid={fileid} due to exception={e}")
            traceback_with_variables.print_exc()
        if lambda_timelimit_exceeded():
            print(f"process_files: lambda_timelimit_exceeded. Breaking..")
            time_limit_exceeded = True
            break
                
    return done_embedding, time_limit_exceeded

def _create_index_metadata_json_local(in_index_metadata:IndexMetadata):
    with open(INDEX_METADATA_JSON, "w") as f:
         json.dump(jsons.dump(in_index_metadata), f)
         print(f"{INDEX_METADATA_JSON} after create = {in_index_metadata}")

def _update_index_metadata_json_local(in_index_metadata:IndexMetadata):
    # update the metadata file
    with open(INDEX_METADATA_JSON, "r") as f:
        index_metadata = jsons.load(json.load(f),IndexMetadata)
        print(f"{INDEX_METADATA_JSON} before update = {index_metadata}")

    if in_index_metadata.jsonl_last_modified: index_metadata.jsonl_last_modified = in_index_metadata.jsonl_last_modified
    if in_index_metadata.index_flat_last_modified: index_metadata.index_flat_last_modified = in_index_metadata.index_flat_last_modified
    if in_index_metadata.index_ivfadc_last_modified: index_metadata.index_ivfadc_last_modified = in_index_metadata.index_ivfadc_last_modified
    
    with open(INDEX_METADATA_JSON, "w") as f:
         json.dump(jsons.dump(index_metadata), f)
         print(f"{INDEX_METADATA_JSON} after  update = {index_metadata}")

def _read_index_metadata_json_local() -> IndexMetadata:
    index_metadata:IndexMetadata = None
    with open(INDEX_METADATA_JSON, "r") as f:
        index_metadata = jsons.load(json.load(f),IndexMetadata)
        print(f"read {INDEX_METADATA_JSON} = {index_metadata}")
    return index_metadata
    
def update_files_index_jsonl(done_embedding, unmodified, bucket, user_prefix, s3client):
    print(f"update_files_index_jsonl: Entered dest=s3://{bucket}/{user_prefix} . new embeddings={len(done_embedding.items())}, unmodified={len(unmodified.items())}")
    # consolidate unmodified and done_embedding
    for fileid, file_item in done_embedding.items():
        unmodified[fileid] = file_item
    unsorted_all_files = []
    for fileid, file_item in unmodified.items():
        unsorted_all_files.append(file_item)
    sorted_all_files = sorted(unsorted_all_files, key=lambda x: x['mtime'], reverse=True)
    files_index_jsonl_gz = FILES_INDEX_JSONL_GZ
    with gzip.open(files_index_jsonl_gz, "wb") as wfp:
        for file_item in sorted_all_files:
            file_item['mtime'] = to_rfc3339(file_item['mtime'])
            wfp.write((json.dumps(file_item) + "\n").encode())
    
    print(f"Uploading  {files_index_jsonl_gz}  Size of file={os.path.getsize(files_index_jsonl_gz)}")
    s3client.upload_file(files_index_jsonl_gz, bucket, f"{user_prefix}/{os.path.basename(files_index_jsonl_gz)}")
    os.remove(FILES_INDEX_JSONL_GZ)
    # remove stale files_index.jsonl
    if os.path.exists(FILES_INDEX_JSONL): os.remove(FILES_INDEX_JSONL)
    
    # create an empty index_metadata_json if it doesn't already exist: this is needed when the index is first being created and none of the files files_index.jsonl.gz, index_metadata.json, faiss_index_flat and faiss_index_ivfadc exist.
    if not os.path.exists(INDEX_METADATA_JSON):
        _create_index_metadata_json_local(IndexMetadata(jsonl_last_modified=time.time()))
    else:
        # update index_metadata_json
        _update_index_metadata_json_local(IndexMetadata(jsonl_last_modified=time.time()))
    print(f"Uploading  {INDEX_METADATA_JSON}  Size of file={os.path.getsize(INDEX_METADATA_JSON)}")
    s3client.upload_file(INDEX_METADATA_JSON, bucket, f"{user_prefix}/{os.path.basename(INDEX_METADATA_JSON)}")

def _delete_faiss_index(email:str, s3client:S3Client, bucket:str, prefix:str, sub_prefix:str, user_prefix:str ):
    """  user_prefix == prefix + email + sub_prefix """
    for faiss_index_fname in (FAISS_INDEX_FLAT, FAISS_INDEX_IVFADC):
        faiss_s3_key = f"{user_prefix}/{os.path.basename(faiss_index_fname)}"
        if os.path.exists(faiss_index_fname): 
            print(f"Deleting faiss index {faiss_index_fname} in local filesystem")
            os.remove(faiss_index_fname)
        try:
            s3client.head_object(Bucket=bucket, Key=faiss_s3_key)
            print(f"Deleting faiss index {faiss_index_fname} in S3 key={faiss_s3_key}")
            s3client.delete_object(Bucket=bucket, Key=faiss_s3_key)
        except Exception as e:
            pass
    
def build_and_save_faiss(email, s3client, bucket, prefix, sub_prefix, user_prefix, doc_storage_type:DocStorageType):
    """ Note that user_prefix == prefix + email + sub_prefix """
    # now build the index.
    faiss_rm:FaissRM = init_vdb(email, s3client, bucket, prefix, sub_prefix=sub_prefix, doc_storage_type=doc_storage_type)
    
    # save the created index
    faiss_flat_fname = FAISS_INDEX_FLAT
    faiss.write_index(faiss_rm.get_index_flat(), faiss_flat_fname)
    print(f"Uploading faiss flat index {faiss_flat_fname}  Size of index={os.path.getsize(faiss_flat_fname)}")
    s3client.upload_file(faiss_flat_fname, bucket, f"{user_prefix}/{os.path.basename(faiss_flat_fname)}")
    os.remove(faiss_flat_fname)
    
    # update index_metadata_json    
    _update_index_metadata_json_local(IndexMetadata(index_flat_last_modified=time.time()))
    print(f"Uploading  {INDEX_METADATA_JSON}  Size of file={os.path.getsize(INDEX_METADATA_JSON)}")
    s3client.upload_file(INDEX_METADATA_JSON, bucket, f"{user_prefix}/{os.path.basename(INDEX_METADATA_JSON)}")
    
    # save the created index
    faiss_ivfadc_fname = FAISS_INDEX_IVFADC
    faiss.write_index(faiss_rm.get_index_ivfadc(), faiss_ivfadc_fname)
    print(f"Uploading faiss ivfadc index {faiss_ivfadc_fname}.  Size of index={os.path.getsize(faiss_ivfadc_fname)}")
    s3client.upload_file(faiss_ivfadc_fname, bucket, f"{user_prefix}/{os.path.basename(faiss_ivfadc_fname)}")
    os.remove(faiss_ivfadc_fname)

    # update index_metadata_json
    _update_index_metadata_json_local(IndexMetadata(index_ivfadc_last_modified=time.time()))
    print(f"Uploading  {INDEX_METADATA_JSON}  Size of file={os.path.getsize(INDEX_METADATA_JSON)}")
    s3client.upload_file(INDEX_METADATA_JSON, bucket, f"{user_prefix}/{os.path.basename(INDEX_METADATA_JSON)}")

def _build_and_save_faiss_if_needed(email:str, s3client, bucket:str, prefix:str, sub_prefix:str, user_prefix:str, doc_storage_type:DocStorageType ):
    """  user_prefix == prefix + email + sub_prefix 
    """
    
    # at this point, we must have the index_metadata.json file: it must have been downloaded above (and is unmodified as no files need to be embedded) or it was modified above (as files needed to be embedded)
    index_metadata:IndexMetadata = _read_index_metadata_json_local()
    # if faiss index needs to be updated and we have at least 5 minutes
    if index_metadata.is_vdb_index_stale():
        time_left:float = lambda_time_left_seconds()
        if time_left > 300:
            print(f"updating faiss index for s3://{bucket}/{user_prefix} as time left={time_left} > 300 seconds")
            # we update the index only when we have 1 to 100 embeddings that need to be updated..
            build_and_save_faiss(email, s3client, bucket, prefix, sub_prefix, user_prefix, doc_storage_type)
        else:
            print(f"Not updating faiss index for s3://{bucket}/{user_prefix} as time left={time_left} < 300 seconds.  Deleting stale faiss indexes if needed..")
            # delete the existing faiss indexes since they are not in sync with files_index.jsonl.gz
            _delete_faiss_index(email, s3client, bucket, prefix, sub_prefix, user_prefix)
            
    else:
        print(f"Not updating faiss index for s3://{bucket}/{user_prefix} as it isn't stale: {index_metadata}")

# Returns entries, dropbox_next_page_token
def _get_dropbox_listing(dbx, dropbox_next_page_token):
    print(f"_get_dropbox_listing: Entered dropbox_next_page_token={dropbox_next_page_token}")
    try: 
        all_entries = []
        cursor = dropbox_next_page_token
        has_more = True
        while has_more:
            if not cursor:
                res = dbx.files_list_folder("", recursive=True)
            else:
                res = dbx.files_list_folder_continue(cursor=cursor)
            all_entries.extend(res.entries)
            has_more = res.has_more
            cursor = res.cursor
        return all_entries, cursor
    except Exception as e: 
        print(str(e)) 
        traceback.print_exc()
        return None

# returns unmodified, needs_embedding
def _calc_filelists_dropbox(entries, s3_index):
    needs_embedding = {}
    num_deleted_files = 0
    for entry in entries:
        if isinstance(entry, dropbox.files.FolderMetadata):
            print(f"_calc_filelists_dropbox: type=DIR id={entry.id} path_display={entry.path_display}, name={entry.name}")
        elif isinstance(entry, dropbox.files.FileMetadata):
            print(f"_calc_filelists_dropbox: type=FILE id={entry.id} path_display={entry.path_display}, name={entry.name}, mtime={entry.server_modified}")
            mtime = entry.server_modified.replace(tzinfo=timezone.utc)
            needs_embedding[entry.id] = {'filename': entry.name,
                                        'fileid': entry.id,
                                        'path': entry.path_display,
                                        'mtime': mtime}
        elif isinstance(entry, dropbox.files.DeletedMetadata):
            print(f"_calc_filelists_dropbox: type=DELETED path_display={entry.path_display}, name={entry.name}")
            to_delete = []
            for fileid, s3entry in s3_index.items():
                if s3entry['path'].startswith(entry.path_display):
                    print(f"_calc_filelists_dropbox: DELETED Yes! deleted={entry.path_display}. s3entry={s3entry['path']}")
                    to_delete.append(fileid)
                else:
                    print(f"_calc_filelists_dropbox: DELETED No! deleted={entry.path_display}. s3entry={s3entry['path']}")
            for td in to_delete:
                del s3_index[td]
                if td in needs_embedding:
                    del needs_embedding[td]
            num_deleted_files += len(to_delete)
        else:
            print(f"_calc_filelists_dropbox: Unknown entry type {entry}. Ignoring and continuing..")
    # if there is any partial entry in s3_index, move it to needs_embedding
    to_move = []
    for fid, entry in s3_index.items():
        if 'partial' in entry:
            to_move.append(fid)
    for tm in to_move:
        needs_embedding[tm] = s3_index[tm]
        del s3_index[tm]
    print(f"_calc_filelists_dropbox: return. s3_index={s3_index}, needs_embedding={needs_embedding}, num_deleted_files={num_deleted_files}")
    return s3_index, needs_embedding, num_deleted_files

def update_index_for_user_dropbox(item:dict, s3client, bucket:str, prefix:str, only_create_index:bool=False, dropbox_next_page_token=None):
    print(f'update_index_for_user_dropbox: Entered. {item}')
    email:str = item['email']['S']
    sub_prefix = "dropbox"
    user_prefix = f"{prefix}/{email}/{sub_prefix}"
    try:
        s3_index = get_s3_index(s3client, bucket, user_prefix)
        # if index already exists and ask is to not update it (only create if not found), then return.
        if s3_index and only_create_index: 
            print(f"update_index_for_user_dropbox: Not updating index for {user_prefix} since index already exists and only_create_index={only_create_index} ")
            return
        access_token = refresh_user_dropbox(item)
    except HttpError as error:
        print(f"HttpError occurred: {error}")
        traceback.print_exc()
        return
    except Exception as ex:
        print(f"Exception occurred: {ex}")
        traceback.print_exc()
        return
    if not access_token:
        print(f"Error. Failed to create access token. Not updating dropbox index")
        return

    try: 
        dbx = dropbox.Dropbox(access_token) 
        entries, dropbox_next_page_token = _get_dropbox_listing(dbx, dropbox_next_page_token)
        unmodified, needs_embedding, num_deleted_files = _calc_filelists_dropbox(entries, s3_index)
    except Exception as ex:
        print(f"Exception occurred: {ex}")
        traceback.print_exc()

    # Move items in 'partial' state from umodified to needs_embedding
    to_del=[]
    for fid, entry in unmodified.items():
        if 'partial' in entry:
            pth = entry['path'] if 'path' in entry else ''
            print(f"update.dropbox: fileid={fid}, path={pth}, filename={entry['filename']} is in partial state. Moving from unmodified to needs_embedding")
            needs_embedding[fid] = entry
            to_del.append(fid)
    for td in to_del:
        del unmodified[td]

    done_embedding, time_limit_exceeded = process_files(DropboxReader(access_token), needs_embedding, s3client, bucket, user_prefix)
    if len(done_embedding.items()) > 0 or num_deleted_files > 0:
        update_files_index_jsonl(done_embedding, unmodified, bucket, user_prefix, s3client)
    else:
        print(f"update_index_for_user_dropbox: No new embeddings or deleted files. Not updating files_index.jsonl")
        
    # at this point, we must have the index_metadata.json file: it must have been downloaded above (and is unmodified as no files need to be embedded) or it was modified above (as files needed to be embedded)    
    _build_and_save_faiss_if_needed(email, s3client, bucket, prefix, sub_prefix, user_prefix, DocStorageType.DropBox )
    return dropbox_next_page_token
    
FOLDER = 'application/vnd.google-apps.folder'

def iterfiles(service:googleapiclient.discovery.Resource, name=None, *, is_folder=None, parent=None,
              order_by='folder,name,createdTime'):
    q = []
    if name is not None:
        q.append("name = '{}'".format(name.replace("'", "\\'")))
    if is_folder is not None:
        q.append("mimeType {} '{}'".format('=' if is_folder else '!=', FOLDER))
    if parent is not None:
        q.append("'{}' in parents".format(parent.replace("'", "\\'")))

    params = {'pageToken': None, 'orderBy': order_by, "fields":"nextPageToken, files(id, name, size, modifiedTime, mimeType, parents)"}
    
    if q:
        params['q'] = ' and '.join(q)

    while True:
        response = service.files().list(**params).execute()
        for f in response['files']:
            yield f
        try:
            params['pageToken'] = response['nextPageToken']
        except KeyError:
            return


def walk(service:googleapiclient.discovery.Resource, top='root', *, by_name: bool = False):
    if by_name:
        top, = iterfiles(name=top, is_folder=True)
    else:
        top = service.files().get(fileId=top).execute()
        if top['mimeType'] != FOLDER:
            raise ValueError(f'not a folder: {top!r}')

    stack = [((top['name'],), top)]
    while stack:
        path, top = stack.pop()

        dirs, files = is_file = [], []
        for f in iterfiles(service, parent=top['id']):
            is_file[f['mimeType'] != FOLDER].append(f)

        yield path, top, dirs, files

        if dirs:
            stack.extend((path + (d['name'],), d) for d in reversed(dirs))

def _process_gdrive_items(gdrive_listing, folder_details, items):
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
    return fileid

def _get_start_page_token(item):
    resp = None
    try:
        headers={"Authorization": f"Bearer {item['access_token']['S']}"}
        params={'supportsAllDrives': True}
        print(f"_get_start_page_token: startPageToken. hdrs={json.dumps(headers)}, params={json.dumps(params)}")
        resp = requests.get('https://www.googleapis.com/drive/v3/changes/startPageToken', headers=headers, params=params)
        resp.raise_for_status()
        respj = resp.json()
        print(f"_get_start_page_token: got {respj} from changes.getStartPageToken")
        return respj['startPageToken']
    except Exception as ex:
        if resp:
            print(f"_get_start_page_token: In changes.getStartPageToken for user {item['email']['S']}, caught {ex}. Response={resp.content}")
        else:
            print(f"_get_start_page_token: In changes.getStartPageToken for user {item['email']['S']}, caught {ex}")
        return None

def _get_gdrive_listing_incremental(service, item, gdrive_next_page_token, folder_id):
    gdrive_listing = {}
    folder_details = {}
    deleted_files = {}

    new_start_page_token = None
    next_token = gdrive_next_page_token
    try:
        driveid = service.files().get(fileId='root').execute()['id']
        folder_details[driveid] = {'filename': 'My Drive'}
    except Exception as ex:
        print(f"_get_gdrive_listing_incremental: Exception {ex} while getting driveid")
        return gdrive_listing, deleted_files, new_start_page_token
        return gdrive_listing, deleted_files, new_start_page_token if new_start_page_token else next_token

    escape = 0
    while not new_start_page_token:
        escape += 1
        if escape > 10:
            break
        resp = None
        try:
            headers = {"Authorization": f"Bearer {item['access_token']['S']}", "Content-Type": "application/json"}
            params={'includeCorpusRemovals': True,
                'includeItemsFromAllDrives': True,
                'includeRemoved': True,
                'pageToken': next_token,
                'pageSize': 100,
                'restrictToMyDrive': False,
                'spaces': 'drive',
                'supportsAllDrives': True}
            print(f"_get_gdrive_listing_incremental: hdrs={json.dumps(headers)}, params={json.dumps(params)}")
            resp = requests.get('https://www.googleapis.com/drive/v3/changes', headers=headers, params=params)
            resp.raise_for_status()
            respj = resp.json()
            print(f"respj={respj}")
            for chg in respj['changes']:
                if chg['removed']:
                    if chg['changeType'] == 'file':
                        fileid = chg['fileId']
                        mtime = from_rfc3339(chg['time'])
                        deleted_files[fileid] = {"fileid": fileid, "mtime": mtime}
                else:
                    if chg['changeType'] == 'file':
                        filename = chg['file']['name']
                        fileid = chg['fileId']
                        mtime = from_rfc3339(chg['time'])
                        mimetype = chg['file']['mimeType']
                        file_details = service.files().get(fileId=fileid, fields='size,parents').execute()
                        if mimetype == 'application/vnd.google-apps.folder':
                            if 'parents' in file_details:
                                folder_details[fileid] = {'filename': filename, 'parents': file_details['parents']}
                        else:
                            size = file_details['size'] if 'size' in file_details else 0
                            if size: # ignore if size is not available
                                if 'parents' in file_details:
                                    entry = {"filename": filename, "fileid": fileid, "mtime": mtime,
                                                            'mimetype': mimetype, 'size': size, 'parents': file_details['parents']}
                                else:
                                    entry = {"filename": filename, "fileid": fileid, "mtime": mtime,
                                                            'mimetype': mimetype, 'size': size}
                                entry['path']=_calc_path(service, entry, folder_details)
                                gdrive_listing[fileid] = entry
            if 'newStartPageToken' in respj:
                new_start_page_token = respj['newStartPageToken']
            else:
                next_token = respj['nextPageToken']
        except Exception as ex:
            if resp:
                print(f"_get_gdrive_listing_incremental: caught {ex}")
                print(f"_get_gdrive_listing_incremental: status={resp.status_code}")
                print(f"_get_gdrive_listing_incremental: content={str(resp.content)}")
                print(f"_get_gdrive_listing_incremental: headers={str(resp.headers)}")
            else:
                print(f"_get_gdrive_listing_incremental: caught {ex}")
    return gdrive_listing, deleted_files, new_start_page_token if new_start_page_token else next_token

def _get_gdrive_listing_full(service, item, folder_id):
    gdrive_listing = {}
    pageToken = None
    kwargs = {}
    folder_details = {}
    gdrive_next_page_token = None
    try:
        driveid = service.files().get(fileId='root').execute()['id']
        folder_details[driveid] = {'filename': 'My Drive'}
    except Exception as ex:
        print(f"_get_gdrive_listing_full: Exception {ex} while getting driveid")
        return gdrive_listing, folder_details, gdrive_next_page_token
    
    # folder_id is specified, then only index the folder's contents
    #if folder_id: kwargs['q'] = "'" + folder_id + "' in parents"
    if folder_id:
        _get_gdrive_rooted_at_folder_id(service, folder_id, gdrive_listing, folder_details)
    else:
        total_items = 0
        while True:
            # Metadata for files: https://developers.google.com/drive/api/reference/rest/v3/files
            results = (
                service.files()
                .list(pageSize=100, fields="nextPageToken, files(id, name, size, modifiedTime, mimeType, parents)", **kwargs)
                .execute()
            )
            items = results.get("files", [])
            total_items += len(items)
            print(f"Fetched {total_items} items from google drive rooted at id={driveid}..")
            # print(f"files = {[ item['name'] for item in items]}")
            
            _process_gdrive_items(gdrive_listing, folder_details, items)
            
            pageToken = results.get("nextPageToken", None)
            kwargs['pageToken']=pageToken
            if not pageToken:
                gdrive_next_page_token = _get_start_page_token(item)
                print(f"_get_gdrive_listing_full: no pageToken. next_page_token={gdrive_next_page_token}")
                break
    return gdrive_listing, folder_details, gdrive_next_page_token

def _get_gdrive_rooted_at_folder_id(service:googleapiclient.discovery.Resource, folder_id:str, gdrive_listing:dict, folder_details:dict):
    total_items = 0
    for path, root, dirs, files in walk(service, top=folder_id, by_name=False):
        total_items += (len(dirs) + len(files))
        print(f'_get_gdrive_rooted_at_folder_id(): total_items={total_items}', '/'.join(path), f'{len(dirs):d} {len(files):d}', sep='\t')
        _process_gdrive_items(gdrive_listing, folder_details, files + dirs)
    
def update_index_for_user_gdrive(item:dict, s3client, bucket:str, prefix:str, only_create_index:bool=False, gdrive_next_page_token:str=None):
    """ only_create_index: only create the index if it doesn't exist; do not update existing index; used when called from 'chat' since we don't want to update the index from chat """
    print(f'update_index_for_user_gdrive: Entered. {item}, gdrive_next_page_token={gdrive_next_page_token}')
    email:str = item['email']['S']
    # index1/xyz@abc.com
    user_prefix = f"{prefix}/{email}"
    folder_id = item['folder_id']['S'] if item.get('folder_id') else None
    try:
        # user_prefix = 'index1/raj@yoja.ai' 
        s3_index:Dict[str, dict] = get_s3_index(s3client, bucket, user_prefix)
        
        # if index already exists and ask is to not update it (only create if not found), then return.
        if s3_index and only_create_index: 
            print(f"Not updating index for {user_prefix} since index already exists and only_create_index={only_create_index} ")
            return
        
        try:
            creds:google.oauth2.credentials.Credentials = refresh_user_google(item)
        except Exception as ex:
            traceback.print_exc()
            print(f"update_index_for_user_gdrive: credentials not valid. not processing user {item['email']['S']}. Exception={ex}")
            return
        
        unmodified:dict; needs_embedding:dict;  deleted_files: dict
        service:googleapiclient.discovery.Resource = build("drive", "v3", credentials=creds)
        if not gdrive_next_page_token or gdrive_next_page_token == "1":
            gdrive_listing, folder_details, gdrive_next_page_token = _get_gdrive_listing_full(service, item, folder_id)
            unmodified, needs_embedding, deleted_files = calc_file_lists(service, s3_index, gdrive_listing, folder_details)
            print(f"gdrive full_listing. Number of unmodified={len(unmodified)}; modified or added={len(needs_embedding)}; deleted={len(deleted_files)}; ")
        else:
            needs_embedding, deleted_files, gdrive_next_page_token = _get_gdrive_listing_incremental(service, item, gdrive_next_page_token, folder_id)
            unmodified = s3_index
        # remove deleted files from the index
        for fileid in deleted_files:
            if fileid in unmodified:
                print(f"update_index_for_user_gdrive: removing deleted fileid {fileid} from s3_index")
                unmodified.pop(fileid)
            else:
                print(f"update_index_for_user_gdrive: deleted fileid {fileid} not in unmodified index")

        # Move items in 'partial' state from umodified to needs_embedding
        to_del=[]
        for fid, entry in unmodified.items():
            if 'partial' in entry:
                pth = entry['path'] if 'path' in entry else ''
                print(f"update.gdrive: fileid={fid}, path={pth}, filename={entry['filename']} is in partial state. Moving from unmodified to needs_embedding")
                needs_embedding[fid] = entry
                to_del.append(fid)
        for td in to_del:
            del unmodified[td]

        done_embedding, time_limit_exceeded = process_files(GoogleDriveReader(service), needs_embedding, s3client, bucket, user_prefix)
        if len(done_embedding.items()) > 0 or len(deleted_files) > 0:
            update_files_index_jsonl(done_embedding, unmodified, bucket, user_prefix, s3client)
        else:
            print(f"update_index_for_user_gdrive: No new embeddings or deleted files. Not updating files_index.jsonl")
        if time_limit_exceeded:
            print("update_index_for_user_gdrive: time limit exceeded. Forcing full listing next time..")
            gdrive_next_page_token = None # force full scan next time
        
        # at this point, we must have the index_metadata.json file: it must have been downloaded above (and is unmodified as no files need to be embedded) or it was modified above (as files needed to be embedded)
        _build_and_save_faiss_if_needed(email, s3client, bucket, prefix, None, user_prefix, DocStorageType.GoogleDrive)
            
    except HttpError as error:
        print(f"HttpError occurred: {error}")
        traceback.print_exc()
    except Exception as ex:
        print(f"Exception occurred: {ex}")
        traceback.print_exc()
    return gdrive_next_page_token

def update_index_for_user(item:dict, s3client, bucket:str, prefix:str, start_time:datetime.datetime,
                            only_create_index:bool=False, gdrive_next_page_token:str=None, dropbox_next_page_token:str=None):
    set_start_time(start_time)
    if not lambda_timelimit_exceeded():
        gdrive_next_page_token = update_index_for_user_gdrive(item, s3client, bucket, prefix, only_create_index, gdrive_next_page_token)
    if not lambda_timelimit_exceeded():
        dropbox_next_page_token = update_index_for_user_dropbox(item, s3client, bucket, prefix, only_create_index, dropbox_next_page_token)
    return gdrive_next_page_token, dropbox_next_page_token

if __name__=="__main__":
    with open(sys.argv[1], 'rb') as f:
        bio = io.BytesIO(f.read())
    rv = read_pdf(sys.argv[1], 'abc', bio, datetime.datetime.now(), vectorizer, [])
    rv['mtime'] = to_rfc3339(rv['mtime'])
    print(json.dumps(rv, indent=4))
    sys.exit(0)
