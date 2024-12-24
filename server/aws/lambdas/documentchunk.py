import os
import sys
import io
from typing import Tuple, List, Dict, Any, Self
import dataclasses
import enum
import faiss_rm
import copy

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

