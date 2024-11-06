import os
import io
import sys
import json
import codecs
from pptx import Presentation
from datetime import datetime, timezone, timedelta
import torch
from typing import List, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModel
import base64
import pickle
import boto3
import zipfile
import tempfile

class CustomModel():
    def __init__(self, bucket_name, object_key):
        print(f"CustomModel: Downloading s3://{bucket_name}/{object_key}")
        s3_client = boto3.client('s3')
        temp_dir = tempfile.mkdtemp()
        print(f"CustomModel: temp_dir={temp_dir}")
        s3_client.download_file(bucket_name, object_key, '/tmp/downloaded_file.zip')
        with zipfile.ZipFile('/tmp/downloaded_file.zip', 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        #temp_dir='/home/jagane/tmp/tmp8j4s76yc'
        if torch.cuda.is_available():
            self._device = torch.device('cuda:0')
            self._model = AutoModel.from_pretrained(temp_dir, trust_remote_code=True).cuda()
        elif torch.backends.mps.is_available():
            self._device = torch.device('mps')
            self._model = AutoModel.from_pretrained(temp_dir, trust_remote_code=True).mps()
        else:
            self._device = torch.device('cpu')
            self._model = AutoModel.from_pretrained(temp_dir, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(temp_dir, model_max_length=512)
        print(f"CustomModel: chosen device={self._device}")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _inner_call(self, inp: List) -> np.ndarray:
        encoded_input = self._tokenizer(inp, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self._device)
        with torch.no_grad():
            model_output = self._model(**encoded_input.to(self._device))

        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings.cpu().numpy().tolist()

    def _extract_text_from_examples(self, inp_examples: List) -> List[str]:
        if isinstance(inp_examples[0], str):
            return inp_examples 
        return [" ".join([example[key] for key in example._input_keys]) for example in inp_examples]

    def __call__(self, inp_examples: List) -> np.ndarray:
        text_to_vectorize = self._extract_text_from_examples(inp_examples)
        for ind in range(int((len(text_to_vectorize) + 15)/16)):
            bef = datetime.now()
            ic = self._inner_call(text_to_vectorize[ind:(ind+1)*16])
            aft = datetime.now()
            if ind == 0:
                rv = ic
            else:
                rv = np.concatenate((rv, ic), axis=0)
        return rv

    def get_token_count(self, inp):
        encoded_input = self._tokenizer(inp, max_length=512, padding=True, truncation=True)
        return len(encoded_input['input_ids'])

if __name__ == '__main__':
    vectorizer = CustomModel(sys.argv[1], sys.argv[2])
    queries = ['problem copying from mps tensor to cpu']
    embeddings = vectorizer(queries)
    emb_npa = np.array(embeddings, dtype=np.float32)
    print(emb_npa[0])
    print(len(emb_npa[0]))
    print("===========================================")
    from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation, util
    model = SentenceTransformer('/home/jagane/tmp/tmp8j4s76yc')
    embeddings = model.encode(['problem copying from mps tensor to cpu'])
    print(embeddings)
    print(len(embeddings[0]))
