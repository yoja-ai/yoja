from typing import Union, Optional
import numpy as np
import faiss
from scipy import spatial
from utils import is_lambda_debug_enabled, prtime, lg
from typing import List, Dict, Any, Tuple
import os
import tempfile
import math
import enum
from text_utils import format_paragraph
import Stemmer
import bm25s
import tarfile

DEFAULT_SEMANTIC_NUM_HITS=1024
COMMON_HITS_PER_QUERY=4
SEMANTIC_HITS_PER_QUERY=20

class DocStorageType(enum.Enum):
    GoogleDrive = 1
    DropBox = 2
    Local = 3

    def __str__(self):
        return f"DocStorageType({self.name})"

    def __repr__(self):
        return f"DocStorageType({self.name})"


def extract_tar_file(tar_file_path, extract_to_directory):
    with tarfile.open(tar_file_path, "r") as tar:
        tar.extractall(path=extract_to_directory)

class FaissRM():
    def __init__(self, documents:Dict[str, dict], index_map:List[Tuple[str,int]],
                    pre_calc_embeddings:List[List[float]], vectorizer,
                    doc_storage_type:DocStorageType, chat_config, tracebuf,
                    k: int = DEFAULT_SEMANTIC_NUM_HITS,
                    flat_index_fname=None, ivfadc_index_fname:str=None,
                    bm25s_index_fname=None):
        """ documents is a dict like {fileid: finfo}; index_map is a list of tuples: [(fileid, paragraph_index)]; 
        
        The two lists are aligned: index_map, pre_calc_embeddings.
        For example, for the 'i'th position, we have the embedding at pre_calc_embeddings[i] and the document chunk for the embedding at index_map[i].
        index_map[i] is the tuple (document_id, paragraph_number).
        'document_id' can be used to index into 'documents'
        
        documents: each item in this dict is similar to below 
        {'1S8cnV...': {'filename': 'Multimodal', 'fileid': '1S8cnV....',
                        'mtime': datetime.datetime(2024, 3, 4, 16, 27, 1, 169000, tzinfo=datetime.timezone.utc),
                        'index_bucket':'yoja-index-2', 'index_object':'index1/raj@.../data/embeddings-1712657862202462825.jsonl'},
        ... }
        
        pre_calc_embeddings: list of embeddings.
        
        index_map: the corresponding text chunk for each embedding in 'pre_calc_embeddings' above.  each element is the tuple (fileid, paragraph_index).
        This is used to locate te text chunk in 'documents'
        """
        self._documents = documents
        self.k = k
        self.doc_storage_type = doc_storage_type
        self._vectorizer = vectorizer
        self._index_map = index_map
        self._chat_config = chat_config
        lg(tracebuf, f"{prtime()} FaissRM: Entered")

        self._stemmer = Stemmer.Stemmer('english')
        if bm25s_index_fname:
            tmpdir=tempfile.mkdtemp()
            extract_tar_file(bm25s_index_fname, tmpdir)
            self._bm25s_retriever = bm25s.BM25.load(os.path.join(tmpdir, 'bm25s_index'), load_corpus=False)
            lg(tracebuf, f"{prtime()} FaissRM: loaded pre-created bm25s index {bm25s_index_fname} untarred into {tmpdir}")
        else:
            bm25s_corpus_lst = []
            for ind in range(len(index_map)):
                finfo = documents[index_map[ind][0]]
                para = self.get_paragraph(ind)
                rcrd = f"{finfo['path']} {finfo['filename']} {format_paragraph(para)}"
                bm25s_corpus_lst.append(rcrd)
            bm25s_corpus_tokens = bm25s.tokenize(bm25s_corpus_lst, stopwords="en", stemmer=self._stemmer)
            self._bm25s_retriever = bm25s.BM25()
            self._bm25s_retriever.index(bm25s_corpus_tokens)
            lg(tracebuf, f"{prtime()} FaissRM: bm25s index created")

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
                tracebuf.append(f"{prtime()} FaissRM: flat index computed using pre calc embedding vectors")
        else:
            print(f"Reading flat index from file {flat_index_fname}")
            self._faiss_index =faiss.read_index(flat_index_fname)
            tracebuf.append(f"{prtime()} FaissRM: pre computed flat index loaded")
        print(f"faiss_index_flat:  total vectors={self._faiss_index.ntotal}; index_memory={self._get_memory(self._faiss_index)}")
        
        if not ivfadc_index_fname:
            # at least 256*40 == 10240, then setup ivfdac. Note that it seems nlist=256 is the minimum for IVF.
            # If nlist < 256 is provided, it still uses 256 it seems (based on error thrown).
            # Note that, from the warning below, we'll need 40 times the number of centroids for training..
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
                # builtins.RuntimeError: Error in void faiss::Clustering::train_encoded(faiss::idx_t, const uint8_t*, const faiss::Index*, faiss::Index&, const float*)
                # at /project/faiss/faiss/Clustering.cpp:275: Error: 'nx >= k' failed: Number of training points (38) should be at least as large as number of clusters (256)
                self._faiss_index_ivf_adc.train(xb)
                self._faiss_index_ivf_adc.add(xb)
                self._faiss_index_ivf_adc.nprobe = 16
                print(f"faiss_index_ivf_adc:  total vectors={self._faiss_index_ivf_adc.ntotal}; index_memory={self._get_memory(self._faiss_index_ivf_adc)}")
                tracebuf.append(f"{prtime()} FaissRM: faiss_index_ivf_adc created")
            # if size not met, then just use the flat index.
            else:
                print(f"Not computing IVFADC since insufficient embedding vectors: vector count={xb.shape[0]}.  Reusing flat index instead")
                self._faiss_index_ivf_adc = self._faiss_index
        else:
            print(f"Reading ivfadc index from file {ivfadc_index_fname}.  Note that this can be flat index due to insufficient embeddings.")
            self._faiss_index_ivf_adc = faiss.read_index(ivfadc_index_fname)
            self._faiss_index_ivf_adc.nprobe = 16
            tracebuf.append(f"{prtime()} FaissRM: ivfadc index loaded from {ivfadc_index_fname}")
        print(f"faiss_index_ivfadc:  total vectors={self._faiss_index_ivf_adc.ntotal}; index_memory={self._get_memory(self._faiss_index_ivf_adc)}")

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

    def get_bm25s_retriever(self):
        return self._bm25s_retriever

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

    def __str__(self):
        return f"FaissRM({self.doc_storage_type})"

    def __repr__(self):
        return f"FaissRM(storage={self.doc_storage_type!r})"

    def __call__(self, query: str, bm25_num_hits: int, k: Optional[int] = None, index_type:str = 'flat',
                named_entities=None, main_theme=None, tracebuf=None):
        """Search the faiss index for k or self.k top passages for query.

        Args:
            query str: The query or queries to search for.
            'k': the number of passages to retrieve (nearest neighbor search or ANN)
            'index_type': 'flat' or 'ivfadc'

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        print(f"faiss_rm: Entered. query={query}, bm25_num_hits={bm25_num_hits}, k={k}, index_type={index_type}, named_entities={named_entities}, main_theme={main_theme}")
        queries = [query]
        queries = [q for q in queries if q]  # Filter empty queries
        embeddings = self._vectorizer.encode(queries)
        emb_npa = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(emb_npa)

        distance_list, index_list = self._faiss_search(emb_npa, k or self.k, index_type)
        lg(tracebuf, f"{prtime()}: faiss search returns {len(distance_list[0])} hits")
        if is_lambda_debug_enabled(): self._dump_raw_results(queries, index_list, distance_list)

        if self._bm25s_retriever:
            if named_entities:
                lg(tracebuf, f"{prtime()}: named_entities present = {named_entities}")
                search_entities = named_entities
            else:
                search_entities = []
            if main_theme:
                lg(tracebuf, f"{prtime()}: main theme present = {main_theme}")
                search_entities.append(main_theme)
            print(f"search_entities={search_entities}")
            if not search_entities:
                return distance_list, index_list # No search_entities. Just return the semantic search results
            else:
                search_entity_tokens = bm25s.tokenize(search_entities, stopwords="en", stemmer=self._stemmer)
                results = self._bm25s_retriever.retrieve(search_entity_tokens, k=bm25_num_hits)
                search_entity_hits = {}
                for search_entity_ind in range(np.shape(results)[1]):
                    for hit_ind in range(np.shape(results)[2]):
                        index_in_faiss = results[0][search_entity_ind][hit_ind]
                        fileid = self._index_map[index_in_faiss][0]
                        para = self._index_map[index_in_faiss][1]
                        score = results[1][search_entity_ind][hit_ind]
                        if not index_in_faiss in search_entity_hits:
                            lg(tracebuf, f"  bm25s result({search_entities[search_entity_ind]}): first {self._documents[self._index_map[index_in_faiss][0]]['path']}{self._documents[self._index_map[index_in_faiss][0]]['filename']},para={self._index_map[index_in_faiss][1]}: {score}")
                            search_entity_hits[index_in_faiss] = (index_in_faiss, score)
                        else:
                            lg(tracebuf, f"  bm25s result({search_entities[search_entity_ind]}): add {self._documents[self._index_map[index_in_faiss][0]]['path']}{self._documents[self._index_map[index_in_faiss][0]]['filename']},para={self._index_map[index_in_faiss][1]}: {score}")
                            search_entity_hits[index_in_faiss] = (index_in_faiss, search_entity_hits[index_in_faiss][1] + score)
                if search_entity_hits.values():
                    for ky in search_entity_hits.keys():
                        if ky in index_list:
                            lg(tracebuf, f"  bm25s res also in semantic seach:  {self._documents[self._index_map[ky][0]]['path']}{self._documents[self._index_map[ky][0]]['filename']},para={self._index_map[ky][1]}")
                            search_entity_hits[ky] = (ky, search_entity_hits[ky][1] + 10.0)
                    sorted_search_entity_hits = sorted(list(search_entity_hits.values()), key=lambda x: x[1], reverse=True)
                    sorted_truncated_search_entity_hits = sorted_search_entity_hits[:4]
                    print(f"sorted_truncated_search_entity_hits={sorted_truncated_search_entity_hits}")
                    indices = []
                    distances = []
                    lg(tracebuf, f"{prtime()}: sorted_truncated_search_entity_hits:")
                    for vl in sorted_truncated_search_entity_hits:
                        indices.append(vl[0])
                        distances.append(vl[1])
                        lg(tracebuf, f"  {self._documents[self._index_map[vl[0]][0]]['path']}{self._documents[self._index_map[vl[0]][0]]['filename']},para={self._index_map[vl[0]][1]}: {vl[1]}")
                    return np.array([np.array(distances)]), np.array([np.array(indices)])
                else:
                    # No hits using bm25 for the named entities
                    print(f"No hits using bm25 for the named entities. Returning truncated lists")
                    per_sem_query_hits = int(COMMON_HITS_PER_QUERY/len(queries))
                    print(f"No hits using bm25 for the named entities. {per_sem_query_hits} hits per semantic query")
                    modified_index_rv = []
                    modified_distances_rv = []
                    for semantic_query_ind in range(len(index_list)):
                        modified_index_rv.append(np.array(index_list[semantic_query_ind][:per_sem_query_hits]))
                        modified_distances_rv.append(np.array(distance_list[semantic_query_ind][:per_sem_query_hits]))
                    return np.array(modified_distances_rv), np.array(modified_index_rv)
        else:
            return distance_list, index_list # No _bm25s_retriever. Just return the semantic search results

if __name__ == '__main__':
    input("Enter the location of the jsonl file: ")
