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
from sklearn.preprocessing import normalize

class StellaV5():
    def __init__(
        self,
        tokenizer_name_or_path = "dunzhang/stella_en_400M_v5",
        model_name_or_path = "dunzhang/stella_en_400M_v5",
    ):
        vector_dim = 1024
        vector_linear_directory = f"2_Dense_{vector_dim}"
        if torch.cuda.is_available():
            self._device = torch.device('cuda:0')
            self._model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, use_memory_efficient_attention=False,unpad_inputs=False).cuda().eval()
        elif torch.backends.mps.is_available():
            self._device = torch.device('mps')
            self._model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, use_memory_efficient_attention=False,unpad_inputs=False).mps().eval()
        else:
            self._device = torch.device('cpu')
            self._model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, use_memory_efficient_attention=False,unpad_inputs=False).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)

        self._vector_linear = torch.nn.Linear(in_features=self._model.config.hidden_size, out_features=vector_dim)
        vector_linear_dict = {
            k.replace("linear.", ""): v for k, v in
            torch.load(os.path.join(model_name_or_path, f"{vector_linear_directory}/pytorch_model.bin"), map_location=torch.device('cpu')).items()
        }
        self._vector_linear.load_state_dict(vector_linear_dict)
        if torch.cuda.is_available():
            self._vector_linear.cuda()
        elif torch.backends.mps.is_available():
            self._vector_linear.mps()
        print(f"StellaV5: chosen device={self._device}")

    def _inner_call(self, docs: List) -> np.ndarray:
        with torch.no_grad():
            input_data = self._tokenizer(docs, padding="longest", truncation=True, max_length=512, return_tensors="pt")
            if self._device == torch.device('cuda:0'):
                input_data = {k: v.cuda() for k, v in input_data.items()}
            elif self._device == torch.device('mps'):
                input_data = {k: v.mps() for k, v in input_data.items()}
            attention_mask = input_data["attention_mask"]
            last_hidden_state = self._model(**input_data)[0]
            last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            docs_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            docs_vectors = normalize(self._vector_linear(docs_vectors).cpu().numpy())
            return docs_vectors.tolist()

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
