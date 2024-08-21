FROM amazon/aws-lambda-python:3.11

RUN yum install \
    tar \
    gzip \
    libdbusmenu.x86_64 \
    libdbusmenu-gtk2.x86_64 \
    libSM.x86_64 \
    xorg-x11-fonts-* \
    google-noto-sans-cjk-fonts.noarch \
    binutils.x86_64 \
    poppler-utils \
    -y && \
    yum clean all

# Begin tesseract block
RUN yum install -y git libtool clang gcc-c++.x86_64
RUN yum install -y zlib zlib-devel libjpeg libjpeg-devel libwebp libwebp-devel libtiff libtiff-devel libpng libpng-devel
RUN cp /usr/lib64/libjpeg.so.62 /usr/local/lib/
RUN cp /usr/lib64/libwebp.so.4 /usr/local/lib/
RUN cp /usr/lib64/libtiff.so.5 /usr/local/lib/
RUN cp /usr/lib64/libpng15.so.15 /usr/local/lib/
RUN rpm -Uvh https://archives.fedoraproject.org/pub/archive/epel/7/x86_64/Packages/e/epel-release-7-14.noarch.rpm
RUN yum -y update
RUN yum install -y tesseract
RUN yum install -y tesseract-osd
RUN yum install -y vim
# End tesseract block

RUN set -xo pipefail && \
    curl "https://mirror.clarkson.edu/tdf/libreoffice/stable/7.6.7/rpm/x86_64/LibreOffice_7.6.7_Linux_x86-64_rpm.tar.gz" | tar -xz

RUN cd LibreOffice_7.6.7.2_Linux_x86-64_rpm/RPMS && \
    yum install *.rpm -y && \
    rm -rf /var/task/LibreOffice_7.6.7.2* && \
    cd /opt/libreoffice7.6/ && \
    strip ./**/* || true

ENV HOME=/tmp

# Trigger dummy run to generate bootstrap files to improve cold start performance
RUN touch /tmp/test.txt \
    && cd /tmp \
    && libreoffice7.6 --headless --invisible --nodefault --view \
        --nolockcheck --nologo --norestore --convert-to pdf \
        --outdir /tmp /tmp/test.txt \
    && rm /tmp/test.*

RUN pip3 install faiss-cpu 
RUN pip3 install polyleven 
RUN pip3 install numpy==1.26.4
RUN pip3 install scipy 
RUN pip3 install pandas 
RUN pip3 install boto3 
RUN pip3 install cloudpickle 
RUN pip3 install openai==1.35.1
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 
RUN pip3 install transformers 
# https://stackoverflow.com/questions/77205123/how-do-i-slim-down-sberts-sentencer-transformer-library
# https://github.com/UKPLab/sentence-transformers/issues/1409
# 
# Installing collected packages: mpmath, urllib3, typing-extensions, tqdm, threadpoolctl, sympy, safetensors, regex, pyyaml, Pillow, packaging, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, networkx, MarkupSafe, joblib, idna, fsspec, filelock, charset-normalizer, certifi, triton, scipy, requests, nvidia-cusparse-cu12, nvidia-cudnn-cu12, jinja2, scikit-learn, nvidia-cusolver-cu12, huggingface-hub, torch, tokenizers, transformers, sentence-transformers
# 
# ├── huggingface-hub [required: >=0.4.0, installed: 0.14.1]
# ├── nltk [required: Any, installed: 3.8.1]
# ├── numpy [required: Any, installed: 1.26.3]
# ├── scikit-learn [required: Any, installed: 1.3.2]
# ├── scipy [required: Any, installed: 1.11.4]
# ├── sentencepiece [required: Any, installed: 0.1.99]
# ├── torch [required: >=1.6.0, installed: 2.1.2]
# ├── torchvision [required: Any, installed: 0.16.2]
# ├── tqdm [required: Any, installed: 4.66.1]
# └── transformers [required: >=4.6.0,<5.0.0, installed: 4.36.2]
# 
# install all dependencies needed for sentence-transformers, except torch and torchvision, since the CPU version is already installed above (but not recoginized by pip as already installed due to +cpu)
RUN pip3 install huggingface-hub nltk scikit-learn sentencepiece tqdm transformers
RUN pip3 install --no-cache-dir sentence-transformers 
RUN pip3 install pipdeptree 
RUN pip3 install traceback-with-variables 
RUN pip3 install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib 
RUN pip3 install cryptography 
RUN pip3 install python-pptx 
RUN pip3 install einops 
RUN pip3 install python-docx
RUN pip3 install reportlab 

RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='sentence-transformers/msmarco-distilbert-base-dot-prod-v3', filename='config.json', local_dir='${LAMBDA_TASK_ROOT}/sentence-transformers/msmarco-distilbert-base-dot-prod-v3', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='sentence-transformers/msmarco-distilbert-base-dot-prod-v3', filename='model.safetensors', local_dir='${LAMBDA_TASK_ROOT}/sentence-transformers/msmarco-distilbert-base-dot-prod-v3', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='sentence-transformers/msmarco-distilbert-base-dot-prod-v3', filename='tokenizer_config.json', local_dir='${LAMBDA_TASK_ROOT}/sentence-transformers/msmarco-distilbert-base-dot-prod-v3', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='sentence-transformers/msmarco-distilbert-base-dot-prod-v3', filename='vocab.txt', local_dir='${LAMBDA_TASK_ROOT}/sentence-transformers/msmarco-distilbert-base-dot-prod-v3', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='sentence-transformers/msmarco-distilbert-base-dot-prod-v3', filename='tokenizer.json', local_dir='${LAMBDA_TASK_ROOT}/sentence-transformers/msmarco-distilbert-base-dot-prod-v3', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='sentence-transformers/msmarco-distilbert-base-dot-prod-v3', filename='special_tokens_map.json', local_dir='${LAMBDA_TASK_ROOT}/sentence-transformers/msmarco-distilbert-base-dot-prod-v3', local_dir_use_symlinks=False)"

RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='cross-encoder/ms-marco-MiniLM-L-6-v2', filename='config.json', local_dir='${LAMBDA_TASK_ROOT}/cross-encoder/ms-marco-MiniLM-L-6-v2', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='cross-encoder/ms-marco-MiniLM-L-6-v2', filename='pytorch_model.bin', local_dir='${LAMBDA_TASK_ROOT}/cross-encoder/ms-marco-MiniLM-L-6-v2', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='cross-encoder/ms-marco-MiniLM-L-6-v2', filename='special_tokens_map.json', local_dir='${LAMBDA_TASK_ROOT}/cross-encoder/ms-marco-MiniLM-L-6-v2', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='cross-encoder/ms-marco-MiniLM-L-6-v2', filename='tokenizer_config.json', local_dir='${LAMBDA_TASK_ROOT}/cross-encoder/ms-marco-MiniLM-L-6-v2', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='cross-encoder/ms-marco-MiniLM-L-6-v2', filename='vocab.txt', local_dir='${LAMBDA_TASK_ROOT}/cross-encoder/ms-marco-MiniLM-L-6-v2', local_dir_use_symlinks=False)"

RUN pip3 install jsons 
RUN pip3 install pandasql 
RUN pip3 install traceback_with_variables pylaprof
RUN pip3 install tenacity
RUN pip3 install dropbox
RUN pip3 install tiktoken
RUN pip3 install BeautifulSoup4

COPY entrypoint.py ${LAMBDA_TASK_ROOT}
COPY getversion.py ${LAMBDA_TASK_ROOT}
COPY login.py ${LAMBDA_TASK_ROOT}
COPY oauth2cb.py ${LAMBDA_TASK_ROOT}
COPY utils.py ${LAMBDA_TASK_ROOT}
COPY periodic.py ${LAMBDA_TASK_ROOT}
COPY distilbert_dotprod.py ${LAMBDA_TASK_ROOT}
COPY chat.py ${LAMBDA_TASK_ROOT}
COPY index_utils.py ${LAMBDA_TASK_ROOT}
COPY pdf.py ${LAMBDA_TASK_ROOT}
COPY faiss_rm.py ${LAMBDA_TASK_ROOT}
COPY table_qa.py ${LAMBDA_TASK_ROOT}
COPY table_qa_lambda.py ${LAMBDA_TASK_ROOT}
COPY openai_ner.py ${LAMBDA_TASK_ROOT}
COPY send_email.py ${LAMBDA_TASK_ROOT}
COPY generate_pdf.py ${LAMBDA_TASK_ROOT}
RUN yum clean all
RUN /bin/rm -rf /var/cache/yum
