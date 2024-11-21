from typing import Union, Optional
import numpy as np
import faiss
from scipy import spatial
from utils import is_lambda_debug_enabled, prtime
from typing import List, Dict, Any, Tuple
import os
import tempfile
import math
import enum
from text_utils import format_paragraph
import Stemmer
import bm25s

BM25_NUM_HITS=32
DEFAULT_SEMANTIC_NUM_HITS=1024
MAX_COMMON_HITS=16

class DocStorageType(enum.Enum):
    GoogleDrive = 1
    DropBox = 2
    Sample = 3

class FaissRM():
    def __init__(self, documents:Dict[str, dict], index_map:List[Tuple[str,int]],
                    pre_calc_embeddings:List[List[float]], vectorizer,
                    doc_storage_type:DocStorageType, chat_config, tracebuf,
                    k: int = DEFAULT_SEMANTIC_NUM_HITS,
                    flat_index_fname=None, ivfadc_index_fname:str=None,
                    bm25s_corpus_records=None):
        """ documents is a dict like {fileid: finfo}; index_map is a list of tuples: [(fileid, paragraph_index)]; 
        
        The two lists are aligned: index_map, pre_calc_embeddings.  For example, for the 'i'th position, we have the embedding at pre_calc_embeddings[i] and the document chunk for the embedding at index_map[i].  index_map[i] is the tuple (document_id, paragraph_number).  'document_id' can be used to index into 'documents'
        
        documents: each item in this dict is similar to below 
        {'1S8cnV...': {'filename': 'Multimodal', 'fileid': '1S8cnV....', 'mtime': datetime.datetime(2024, 3, 4, 16, 27, 1, 169000, tzinfo=datetime.timezone.utc), 'index_bucket':'yoja-index-2', 'index_object':'index1/raj@.../data/embeddings-1712657862202462825.jsonl'}, ... }
        
        pre_calc_embeddings: list of embeddings.
        
        index_map: the corresponding text chunk for each embedding in 'pre_calc_embeddings' above.  each element is the tuple (fileid, paragraph_index).  This is used to locate te text chunk in 'documents'
        """
        self._vectorizer = vectorizer
        self._index_map = index_map
        self._chat_config = chat_config
        self._tracebuf = tracebuf

        self._bm25s_corpus_records = bm25s_corpus_records
        self._bm25s_retriever = None
        if self._bm25s_corpus_records:
            self._stemmer = Stemmer.Stemmer('english')
            bm25s_corpus_lst = []
            for rc in self._bm25s_corpus_records:
                finfo = documents[rc['fileid']]
                rcrd = f"{finfo['path']} {finfo['filename']} {rc['text']}"
                bm25s_corpus_lst.append(rcrd)
            bm25s_corpus_tokens = bm25s.tokenize(bm25s_corpus_lst, stopwords="en", stemmer=self._stemmer)
            self._bm25s_retriever = bm25s.BM25()
            self._bm25s_retriever.index(bm25s_corpus_tokens)

        if is_lambda_debug_enabled():
            print(f"faiss_rm: Entered. Document chunks=")
            for ch in documents:
                print(f"\t{ch}")

        if not flat_index_fname:
            if len(pre_calc_embeddings) == 0:
                raise Exception(f"faiss_rm: Error. Neither pre-calculated embeddings nor flat_index_fname present")
            else:
                print(f"Computing flat index using embedding vectors")
                embeddings = pre_calc_embeddings
                xb:np.array = np.array(embeddings, dtype=np.float32)
                faiss.normalize_L2(xb)
                # dimension
                d = xb.shape[1]
                print(f"FaissRM: embedding dimension={d}")
                self._faiss_index:faiss.IndexFlatL2 = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
                self._faiss_index.add(xb)
        else:
            print(f"Reading flat index from file {flat_index_fname}")
            self._faiss_index =faiss.read_index(flat_index_fname)
        print(f"faiss_index_flat:  total vectors={self._faiss_index.ntotal}; index_memory={self._get_memory(self._faiss_index)}")
        
        if not ivfadc_index_fname:
            # at least 256*40 == 10240, then setup ivfdac. Note that it seems nlist=256 is the minimum for IVF.  If nlist < 256 is provided, it still uses 256 it seems (based on error thrown).  Note that, from the warning below, we'll need 40 times the number of centroids for training..
            if xb.shape[0] >= 256*40:
                print(f"Computing IVFADC using embedding vectors")
                # for IVF: nlist=256
                # for PQ: m=32; nbits=8
                ivf_nlist:int = 256
                if xb.shape[0] < ivf_nlist: ivf_nlist = 1 if xb.shape[0] == 0 else 2**math.floor(math.log2(xb.shape[0]))
                pq_m = 32  # 768/32 == subvector of size 24 (floats)
                pq_nbits = 8 # 2^8 == 256 centroids

                factory:str = f"IVF{ivf_nlist},PQ{pq_m}x{pq_nbits}"
                print(f"creating IVFADC index using factory={factory}; ivf_nlist={ivf_nlist}, pq_m={pq_m}, pq_nbits={pq_nbits}")
                self._faiss_index_ivf_adc:faiss.IndexIVFPQ = faiss.index_factory(d, factory, faiss.METRIC_INNER_PRODUCT)
                # WARNING clustering 38 points to 32 centroids: please provide at least 1248 training points
                # builtins.RuntimeError: Error in void faiss::Clustering::train_encoded(faiss::idx_t, const uint8_t*, const faiss::Index*, faiss::Index&, const float*) at /project/faiss/faiss/Clustering.cpp:275: Error: 'nx >= k' failed: Number of training points (38) should be at least as large as number of clusters (256)
                self._faiss_index_ivf_adc.train(xb)
                self._faiss_index_ivf_adc.add(xb)
                self._faiss_index_ivf_adc.nprobe = 16
                print(f"faiss_index_ivf_adc:  total vectors={self._faiss_index_ivf_adc.ntotal}; index_memory={self._get_memory(self._faiss_index_ivf_adc)}")
            # if size not met, then just use the flat index.
            else:
                print(f"Not computing IVFADC since not enough embedding vectors: vector count={xb.shape[0]}.  Reusing flat index instead")
                self._faiss_index_ivf_adc = self._faiss_index
        else:
            print(f"Reading ivfadc index from file {ivfadc_index_fname}.  Note that this can be flat index due to not enough embeddings.")
            self._faiss_index_ivf_adc = faiss.read_index(ivfadc_index_fname)
            self._faiss_index_ivf_adc.nprobe = 16
        print(f"faiss_index_ivfadc:  total vectors={self._faiss_index_ivf_adc.ntotal}; index_memory={self._get_memory(self._faiss_index_ivf_adc)}")

        self._documents = documents  # save the input document chunks
        self.k = k
        self.doc_storage_type = doc_storage_type

    def get_doc_storage_type(self) -> DocStorageType:
        return self.doc_storage_type
    
    #@staticmethod
    @classmethod
    def _get_memory(cls, index:faiss.Index) -> int:
        # write index to file
        tmpfile = f'{tempfile.gettempdir()}/temp.index'
        faiss.write_index(index, tmpfile)
        # get file size
        file_size = os.path.getsize(tmpfile)
        # delete saved index
        os.remove(tmpfile)
        return file_size

    def get_documents(self):
        """ documents is a dict like {fileid: finfo}; """
        return self._documents

    def get_index_flat(self):
        return self._faiss_index
    
    def get_index_ivfadc(self):
        return self._faiss_index_ivf_adc
        
    def _dump_raw_results(self, queries, index_list, distance_list) -> None:
        for i in range(len(queries)):
            indices = index_list[i]
            distances = distance_list[i]
            print(f"Query: {queries[i]}")
            for j in range(len(indices)):
                para = self.get_paragraph(indices[j])
                print(f"    Hit {j} = {indices[j]}/{distances[j]}: {format_paragraph(para)}")
        return

    def get_paragraph(self, index_in_faiss):
        fileid, para_index = self._index_map[index_in_faiss]
        finfo = self._documents[fileid]
        if 'slides' in finfo:
            key = 'slides'
        elif 'paragraphs' in finfo:
            key = 'paragraphs'
        elif 'rows' in finfo:
            key = 'rows'
        else:
            return None
        return finfo[key][para_index]

    def get_paragraphs(self, index_in_faiss, num_paragraphs) -> Tuple[List[str],int, int]:
        """ returns the paragraph starting from index_in_faiss and subsequent paragraphs, for a total of num_paragraphs.  returns the tuple (list of paragraph text, start_para_index, end_para_index).  end_para_index is inclusive. """
        fileid, para_index = self._index_map[index_in_faiss]
        finfo = self._documents[fileid]
        if 'slides' in finfo:
            key = 'slides'
        elif 'paragraphs' in finfo:
            key = 'paragraphs'
        elif 'rows' in finfo:
            key = 'rows'
        else:
            return None, None, None
        rv = []
        start_para_index:int = None; end_para_index:int = None
        for ind in range(num_paragraphs):
            if para_index+ind >= 0 and para_index+ind < len(finfo[key]):
                rv.append(format_paragraph(finfo[key][para_index+ind]))
                if start_para_index == None: 
                    start_para_index = para_index + ind
                end_para_index = para_index + ind
                
        return rv, start_para_index, end_para_index

    def get_index_map(self):
        """ index_map is a list of tuples: [(fileid, paragraph_index)];  the index into this list corresponds to the index of the embedding vector in the faiss index """
        return self._index_map

    def _faiss_search(self, emb_npa, k, index_type:str = 'flat'):
        print(f"using index_type={index_type} for faiss")
        if k > len(self._index_map):
            print(f"_faiss_search: reducing k from {k} to the number of entries in this vdb, i.e. {len(self._index_map)}")
            k = len(self._index_map)
        dist_rv = []
        index_rv = []
        index = self._faiss_index_ivf_adc if index_type == 'ivfadc' else self._faiss_index
        # https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexFlat.html
        # 
        distance_list, index_list = index.search(emb_npa, k)
            
        #print(f"_faiss_search: index_type={index.__class__.__module__}.{index.__class__.__name__}: before trunc distance_list={distance_list}")
        #print(f"_faiss_search: index_type={index.__class__.__module__}.{index.__class__.__name__}: before trunc index_list={index_list}")
        lindex = list(list(index_list)[0])
        try:
            trunc_point = lindex.index(-1)
            print(f"_faiss_search: index_type={index_type}: Found -1 at {trunc_point}. Truncating results to {trunc_point} entries")
            return np.array([distance_list[0][:trunc_point]]), np.array([index_list[0][:trunc_point]])
        except ValueError as ve:
            for trunc_point in range(len(lindex)):
                if lindex[trunc_point] >= len(self._index_map):
                    print(f"_faiss_search: index_type={index_type}: Found ind > len(index_map) at {trunc_point}. Truncating results to {trunc_point} entries")
                    return np.array([distance_list[0][:trunc_point]]), np.array([index_list[0][:trunc_point]])
            return distance_list, index_list

    def __call__(self, query: str, k: Optional[int] = None, index_type:str = 'flat', bm25terms=None):
        """Search the faiss index for k or self.k top passages for query.

        Args:
            query str: The query or queries to search for.
            'k': the number of passages to retrieve (nearest neighbor search or ANN)
            'index_type': 'flat' or 'ivfadc'

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        queries = [query]
        queries = [q for q in queries if q]  # Filter empty queries
        embeddings = self._vectorizer(queries)
        emb_npa = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(emb_npa)

        distance_list, index_list = self._faiss_search(emb_npa, k or self.k, index_type)
        msg = f"{prtime()}: faiss search returns {len(distance_list[0])} hits"
        print(msg)
        if self._chat_config and self._chat_config.print_trace: self._tracebuf.append(msg)
        if is_lambda_debug_enabled(): self._dump_raw_results(queries, index_list, distance_list)

        if self._bm25s_retriever and bm25terms:
            # first create one dict of hits per bm25term. Each dict has {'fileid1': {'para1': num_hits, 'para2':num_hits}, 'fileid2': {'para1': num_hits}}
            query_tokens = bm25s.tokenize(bm25terms, stopwords="en", stemmer=self._stemmer)
            results = self._bm25s_retriever.retrieve(query_tokens, self._bm25s_corpus_records, k=BM25_NUM_HITS)
            bm25terms_hits = []
            for bm25_query_ind in range(np.shape(results)[1]):
                bm25_hits = {}
                for hit_ind in range(np.shape(results)[2]):
                    fileid = results[0][bm25_query_ind][hit_ind]['fileid']
                    para = results[0][bm25_query_ind][hit_ind]['para']
                    score = results[1][bm25_query_ind][hit_ind]
                    if not fileid in bm25_hits:
                        bm25_hits[fileid] = {}
                    bm25_hits[fileid][para] = score
                msg = f"{prtime()}: bm25_hits({bm25terms[bm25_query_ind]}):"
                print(msg)
                if self._chat_config and self._chat_config.print_trace: self._tracebuf.append(msg)
                for ky, vl in bm25_hits.items():
                    msg = f"  {self._documents[ky]['path']}{self._documents[ky]['filename']}: {vl}"
                    print(msg)
                    if self._chat_config and self._chat_config.print_trace: self._tracebuf.append(msg)
                bm25terms_hits.append(bm25_hits)
            # next, for each term in the semantic search, filter semantic search results by presence in any of the bm25terms_hits
            common_hits_found = 0
            modified_index_rv = []
            modified_distances_rv = []
            for semantic_query_ind in range(len(index_list)):
                print(f"{prtime()}: begin filtering results of semantic query for {queries[semantic_query_ind]} by results of all bm25 search terms. common_hits_found={common_hits_found}")
                indices = index_list[semantic_query_ind]
                distances = distance_list[semantic_query_ind]
                modified_indices = []
                modified_distances = []
                if common_hits_found < MAX_COMMON_HITS:
                    for sem_ind in range(len(indices)):
                        index = indices[sem_ind]
                        distance = distances[sem_ind]
                        fileid, para_index = self._index_map[index]
                        finfo = self._documents[fileid]
                        for bm25_ind in range(len(bm25terms_hits)):
                            bm25_hits = bm25terms_hits[bm25_ind]
                            if fileid in bm25_hits and para_index in bm25_hits[fileid]:
                                msg = f"{finfo['filename']}, para {para_index} is in both search results. Including"
                                print(msg)
                                if self._chat_config and self._chat_config.print_trace: self._tracebuf.append(msg)
                                modified_indices.append(index)
                                modified_distances.append(distance)
                                common_hits_found += 1
                                break
                            else:
                                print(f"{finfo['filename']}, para {para_index} is in semantic search, but not bm25 term {bm25terms[bm25_ind]}. Excluding")
                        if common_hits_found >= MAX_COMMON_HITS:
                            break
                modified_index_rv.append(np.array(modified_indices))
                modified_distances_rv.append(np.array(modified_distances))
            # finally, if we found common hits return those, otherwise truncate the semantic list and return that
            if common_hits_found > 0:
                print(f"{common_hits_found} common hits found. Returning modified lists")
                return np.array(modified_distances_rv), np.array(modified_index_rv)
            else:
                print(f"No common hits found. Returning truncated lists")
                per_sem_query_hits = int(MAX_COMMON_HITS/len(queries))
                print(f"No common hits found. {per_sem_query_hits} hits per semantic query")
                modified_index_rv = []
                modified_distances_rv = []
                for semantic_query_ind in range(len(index_list)):
                    modified_index_rv.append(np.array(index_list[semantic_query_ind][:per_sem_query_hits]))
                    modified_distances_rv.append(np.array(distance_list[semantic_query_ind][:per_sem_query_hits]))
                return np.array(modified_distances_rv), np.array(modified_index_rv)
        else:
            return distance_list, index_list

if __name__ == '__main__':
    input("Enter the location of the jsonl file: ")
