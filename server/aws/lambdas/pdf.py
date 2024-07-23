import sys
from bs4 import BeautifulSoup
import re
import tempfile
import subprocess
import io
import datetime
import os
from utils import lambda_timelimit_exceeded
import base64
import traceback
import traceback_with_variables
import pickle
from typing import TYPE_CHECKING
import shutil

import llama_index
import llama_index.core
import llama_index.readers.file
import llama_index.core.node_parser

def read_text_pdf(bio):
    # https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/
    pdf_reader:llama_index.readers.file.PyMuPDFReader = llama_index.readers.file.PyMuPDFReader()
    # write the bytes out to a file
    with tempfile.NamedTemporaryFile('wb') as tfile:
        bio.seek(0)
        tfile.write(bio.read()); tfile.flush(); tfile.seek(0)
        # error: fitz.EmptyFileError: Cannot open empty file: filename='/tmp/tmpt6o3l__m'.
        # error: FileDataError – if the document has an invalid structure for the given type – or is no file at all (but e.g. a folder). A subclass of RuntimeError. ( https://pymupdf.readthedocs.io/en/latest/document.html )
        docs:List[llama_index.core.Document] = pdf_reader.load(file_path=tfile.name)
    
    sent_split:llama_index.core.node_parser.SentenceSplitter = llama_index.core.node_parser.SentenceSplitter(chunk_size=512)
    # 
    chunks:List[str] = []
    for doc in docs:
        chunks.extend(sent_split.split_text(doc.text))
    return chunks

def read_image_pdf(bio):
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmpfp:
        tmpfp.write(bio.read())
        tmpfp.close()
        outdir=tempfile.mkdtemp()
        print(f"tmpfile={tmpfp.name}, outdir={outdir}")
        args=['pdfimages', '-png', tmpfp.name, 'out']
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, cwd=outdir, close_fds=True)
        for line in process.stdout:
            lined=line.decode('utf-8').rstrip()
        rc = process.returncode
        try:
            os.remove(tmpfp.name)
        except Exception as ex:
            print(f"Exception {ex} removing temp file")
        if rc:
            if rc == 1:
                print(f"read_image_pdf: pdfimages error opening input pdf file")
            elif rc == 2:
                print(f"read_image_pdf: pdfimages error opening output file")
            elif rc == 3:
                print(f"read_image_pdf: pdfimages pdf permissions error")
            elif rc == 98:
                print(f"read_image_pdf: pdfimages Out of memory")
            elif rc == 99:
                print(f"read_image_pdf: pdfimages other error")
            else:
                print(f"read_image_pdf: unknown pdfimages error {rc}")
            return None
    for filename in os.listdir(outdir):
        try:
            args=['tesseract', filename, f"{os.path.basename(filename)[:-4]}", '-l', 'eng', 'hocr']
            process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, cwd=outdir, close_fds=True)
            for line in process.stdout:
                lined=line.decode('utf-8').rstrip()
                print(lined)
            rc = process.returncode
            if rc:
                print(f"read_image_pdf: tesseract error {rc}. Ignoring {filename} and continuing..")
        except Exception as ex:
            print(f"read_image_pdf: Caught {ex} while processing {filename}. Ignoring and continuing ..")

    chunks:List[str] = []
    chunk = ''
    for filename in sorted(os.listdir(outdir)):
        try:
            if (filename.endswith(".hocr")):
                with open(os.path.join(outdir, filename), 'rb') as fp:
                    hh = fp.read()
                soup = BeautifulSoup(hh, features="html.parser")
                for para in soup.find_all("p"): 
                    text = para.get_text()
                    nonl_text = text.replace('\n', ' ').replace('\r', '')
                    nonl_ss = re.sub(' +', ' ', nonl_text)
                    #print(nonl_ss)
                    chunk += f"{nonl_ss}\n"
                    if len(chunk) > 2048:
                        print(f"chunk{len(chunks)}={chunk}")
                        chunks.append(chunk)
                        chunk = ''
        except Exception as ex:
            print(f"read_image_pdf: Caught {ex} while processing {filename}. Ignoring and continuing ..")
    try:
        shutil.rmtree(outdir)
    except Exception as ex:
        print(f"Caught {ex} deleting output dir {outdir}")
    return chunks

def read_pdf(filename, fileid, bio, mtime, vectorizer, prev_paras):
    pdfsize = bio.getbuffer().nbytes
    if pdfsize < (400 * 1024):
        print(f"read_pdf: size is {int(pdfsize/1024)} kilobytes. Treating as text based pdf..")
        chunks = read_text_pdf(bio)
    else:
        print(f"read_pdf: size is {int(pdfsize/1024)} kilobytes. Treating as image based pdf..")
        chunks = read_image_pdf(bio)
    doc_dct={"filename": filename, "fileid": fileid, "mtime": mtime, "paragraphs": prev_paras} 
    for ind in range(len(prev_paras), len(chunks)):
        chunk = chunks[ind]
        para_dct = {'paragraph_text':chunk} # {'paragraph_text': 'Module 2: How To Manage Change', 'embedding': 'gASVCBsAAAAAAA...GVhLg=='}
        try:
            embedding = vectorizer([f"The name of the file is {filename} and the paragraph is {chunk}"])
            eem = base64.b64encode(pickle.dumps(embedding)).decode('ascii')
            para_dct['embedding'] = eem
            doc_dct['paragraphs'].append(para_dct)
            if lambda_timelimit_exceeded():
                doc_dct['partial'] = "true"
                print(f"read_text_pdf: Timelimit exceeded when reading pdf files. Breaking..")
                break
        except Exception as ex:
            print(f"Exception {ex} while creating para embedding")
            traceback.print_exc()
    print(f"read_pdf: fn={filename}. returning. num paras={len(doc_dct['paragraphs'])}")
    return doc_dct


if __name__=="__main__":
    with open(sys.argv[1], 'rb') as f:
        bio = io.BytesIO(f.read())
    rv = read_pdf(sys.argv[1], 'abc', bio, datetime.datetime.now(), 0, 0, [])
    sys.exit(0)
