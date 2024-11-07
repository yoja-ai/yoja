import os
import io
import sys
import json
import codecs
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
        self._device = torch.device('cpu')
        self._model = AutoModel.from_pretrained(temp_dir, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(temp_dir, model_max_length=512)
        print(f"CustomModel: chosen device={self._device}")

    def _inner_call(self, sentences:List):
        # Tokenize the input sentences and get token embeddings
        inputs = self._tokenizer(sentences, max_length=512, return_tensors="pt", padding=True, truncation=True)
        outputs = self._model(**inputs)

        # Extract the last hidden state tensor (token embeddings)
        token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)
        attention_mask = inputs['attention_mask']  # Shape: (batch_size, seq_length)

        # Apply mean pooling
        # Mask padding tokens before averaging
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embeddings = token_embeddings * attention_mask_expanded
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)  # Avoid division by zero
        mean_pooled_embeddings = sum_embeddings / sum_mask
        print(f"inner_call: returning type {type(mean_pooled_embeddings)}")
        return mean_pooled_embeddings.detach().numpy()

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
