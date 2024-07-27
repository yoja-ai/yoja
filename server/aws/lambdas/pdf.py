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

def get_pdf_info(pdf_file_name):
    try:
        process = subprocess.Popen(['pdfinfo', pdf_file_name], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                cwd='/tmp', close_fds=True)
        ov = {}
        for line in process.stdout:
            lined=line.decode('utf-8').rstrip()
            spl = lined.split(':', 1)
            if (len(spl) == 2):
                ov[spl[0]] = spl[1]
        rc = process.returncode
        if rc:
            print(f"Error {rc} while getting pdfinfo. Returning without doing any work")
            return 0, 0
        # Pages:           8
        # File size:       676081 bytes
        return int(ov['Pages'].strip()), int(ov['File size'].strip().split()[0])
    except Exception as ex:
        print(f"get_pdf_info: Caught {ex}")
        return 0, 0

def _lsdir(ldir):
    try:
        lstmp=subprocess.check_output(['ls', ldir]).decode('utf-8')
        print(f"_lsdir: contents of {ldir} = {lstmp}")
        return
    except Exception as ex:
        print(f"_lsdir: dir={ldir}, Caught {ex}")
        return

def render_png(filename, start_page, end_page, tmpdir):
    try:
        cmd = ['pdftocairo', '-png', '-r', '300', '-f', str(start_page), '-l', str(end_page), filename, 'out']
        print(f"render_png: cmd={cmd}")
        process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, cwd=tmpdir, close_fds=True)
        for line in process.stdout:
            lined=line.decode('utf-8').rstrip()
            print(lined)
        rc = process.returncode
        if rc:
            print(f"Error {rc} while rendering png. Returning without doing any work")
            return False
        return True
    except Exception as ex:
        print(f"render_png: Caught {ex}")
        return False

def tesseract_one_page(pgnum, pages, tmpdir):
    fmt1=f"out-%0{len(str(pages))}d.png"
    fmt2=f"out-%0{len(str(pages))}d"
    fmt3=f"out-%0{len(str(pages))}d.hocr"
    inputfn=str(fmt1 % pgnum)
    outputfn=str(fmt2 % pgnum)
    outputfn1=str(fmt3 % pgnum)
    print(f"tesseract_one_page: input={inputfn}, outputfn={outputfn}, outputfn1={outputfn1}")
    try:
        tenv=os.environ.copy()
        tenv['TESSDATA_PREFIX']='/usr/share/tesseract/tessdata'
        cmd = ['tesseract', os.path.join(tmpdir, inputfn), os.path.join(tmpdir, outputfn), '-l', 'eng', 'hocr']
        print(f"tesseract_one_page: cmd={cmd}")
        process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    cwd=tmpdir, close_fds=True, env=tenv)
        for line in process.stdout:
            lined=line.decode('utf-8').rstrip()
            print(lined)
        rc = process.returncode
        if rc:
            print(f"Error {rc} while running tesseract. Returning without doing any work")
            return None
        with open(os.path.join(tmpdir, outputfn1), 'rb') as fp:
            hh = fp.read()
        chunk = ''
        soup = BeautifulSoup(hh, features="html.parser")
        for para in soup.find_all("p"): 
            text = para.get_text()
            nonl_text = text.replace('\n', ' ').replace('\r', '')
            nonl_ss = re.sub(' +', ' ', nonl_text)
            print(nonl_ss)
            chunk += f"{nonl_ss}\n"
        return chunk
    except Exception as ex:
        print(f"tesseract_one_page: Caught {ex}")
        return None

def read_pdf(filename, fileid, bio, mtime, vectorizer, prev_paras):
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmpfp:
            tmpfp.write(bio.read())
            tmpfp.close()
        tmpdir = tempfile.mkdtemp()
        print(f"tmpdir where pngs are created={tmpdir}")
        pages, pdfsize = get_pdf_info(tmpfp.name)
        print(f"pages={pages}, pdfsize={pdfsize}")
        if not render_png(tmpfp.name, 1, pages+1, tmpdir):
            return
        doc_dct={"filename": filename, "fileid": fileid, "mtime": mtime, "paragraphs": prev_paras} 
        chunks:List[str] = []
        for pgnum in range(1, pages+1):
            chunk = tesseract_one_page(pgnum, pages, tmpdir)
            if chunk:
                chunks.append(chunk)

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
    except Exception as ex:
        print(f"read_pdf: fn={filename}. Caught {ex}")
        traceback.print_exc()
        return None
