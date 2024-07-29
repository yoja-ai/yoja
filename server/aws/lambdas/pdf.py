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
import fcntl
import select

PARALLEL_PROCESSES=os.cpu_count()-1 if os.cpu_count() > 1 else 1
print(f"pdf: using {PARALLEL_PROCESSES} cpus for processing pdfs")

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
    print(f"render_png: Entered. filename {filename}, start_page {start_page}, end_page {end_page}, tmpdir {tmpdir}")
    try:
        fnx_start = datetime.datetime.now()
        first = start_page
        while first <= end_page:
            print(f"render_png: outer loop begin: first={first}")
            cmds = []
            for cmd_index in range(PARALLEL_PROCESSES):
                print(f"render_png: inner loop begin: page={first}")
                cmd = ['pdftocairo', '-png', '-r', '300', '-f',
                            str(first), '-l', str(first), filename, 'out']
                print(f"render_png: inner loop append cmd: {cmd}")
                cmds.append(cmd)
                first += 1
                if first > end_page:
                    break
            print(f"render_png: inner loop complete: cmds array has {len(cmds)} entries")
            processes=[]
            for cmd in cmds:
                process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, cwd=tmpdir, close_fds=True)
                fd=process.stdout.fileno()
                flags=fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags|os.O_NONBLOCK)
                processes.append(process)
            poller=select.poll()
            fd_to_process={}
            for process in processes:
                fd=process.stdout.fileno()
                fd_to_process[fd]=process
                poller.register(fd, select.POLLIN|select.POLLPRI|select.POLLHUP|select.POLLERR)
            print(f"Entering poll loop ..")
            while fd_to_process.items():
                events = poller.poll(1000)
                for fd, flag in events:
                    print(f"render_png: poll loop. fd {fd}, pid {fd_to_process[fd].pid}")
                    if flag & (select.POLLIN | select.POLLPRI):
                        bts=os.read(fd, 10248)
                        print(f"[{fd_to_process[fd].pid}] {bts.decode('utf-8').rstrip()}")
                    elif flag & select.POLLHUP:
                        print(f"[{fd_to_process[fd].pid}] Received HUP. Child finished")
                        poller.unregister(fd)
                        del fd_to_process[fd]
                    elif flag & select.POLLERR:
                        print(f"[{fd_to_process[fd].pid}] Received ERR. Child error")
                        poller.unregister(fd)
                        del fd_to_process[fd]
            print(f"Exited poll loop ..")
            for process in processes:
                rc = process.returncode
                if rc:
                    print(f"[{process.pid}] Process returncode {rc}")
                else:
                    print(f"[{process.pid}] sucecss")
        fnx_end = datetime.datetime.now()
        print(f"render_png: time taken {fnx_end - fnx_start}")
        return True
    except Exception as ex:
        print(f"render_png: Caught {ex}")
        return False

class ProcessInfo:
    process = None
    outfn = None
    page = 0

    def __init__(self, process, outfn, page):
        self.process=process
        self.outfn=outfn
        self.page=page

    def __str__(self):
        return f"page={self.page}, pid={self.process.pid}, outfn={self.outfn}"
    

# start_page and end_page and inclusive
# max_pages is max pages in pdf, use to determine the name of the png file, e.g. out-0011.png
def tesseract_pages(filename, start_page, end_page, max_pages, tmpdir):
    num_pages = end_page-start_page+1
    print(f"tesseract_pages: Entered. filename {filename}, start_page {start_page}, end_page {end_page}, num_pages={num_pages}, max_pages={max_pages}, tmpdir {tmpdir}")
    rv = ['' for ind in range(num_pages)]
    try:
        fnx_start = datetime.datetime.now()
        tenv=os.environ.copy()
        tenv['TESSDATA_PREFIX']='/usr/share/tesseract/tessdata'
        first = start_page
        while first <= end_page:
            print(f"tesseract_pages: outer loop begin: first={first}")
            fd_to_processinfo = {}
            poller=select.poll()
            items_in_poll=0
            for cmd_index in range(PARALLEL_PROCESSES):
                print(f"tesseract_pages: inner loop begin: page={first}")
                fmt1=f"out-%0{len(str(max_pages))}d.png"
                fmt2=f"out-%0{len(str(max_pages))}d"
                fmt3=f"out-%0{len(str(max_pages))}d.hocr"
                inputfn=str(fmt1 % first)
                outputfn=str(fmt2 % first)
                outputfn1=str(fmt3 % first)
                cmd = ['tesseract', os.path.join(tmpdir, inputfn), os.path.join(tmpdir, outputfn), '-l', 'eng', 'hocr']
                print(f"tesseract_pages: inner loop cmd: {cmd}")
                process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, cwd=tmpdir, close_fds=True, env=tenv)
                fd=process.stdout.fileno()
                flags=fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags|os.O_NONBLOCK)
                poller.register(fd, select.POLLIN|select.POLLPRI|select.POLLHUP|select.POLLERR)
                items_in_poll += 1
                fd_to_processinfo[fd] = ProcessInfo(process, outputfn1, first)
                first += 1
                if first > end_page:
                    break
            print(f"tesseract_pages: inner loop complete. Spawned {len(fd_to_processinfo.items())} processes")
            print(f"Entering poll loop items in poll={items_in_poll} ..")
            while items_in_poll > 0:
                print(f"About to call poll. items in poll={items_in_poll}")
                events = poller.poll(1000)
                for fd, flag in events:
                    print(f"tesseract_pages: poll loop. fd {fd}, pid {fd_to_processinfo[fd].process.pid}")
                    if flag & (select.POLLIN | select.POLLPRI):
                        bts=os.read(fd, 10248)
                        print(f"[{fd_to_processinfo[fd].process.pid}] {bts.decode('utf-8').rstrip()}")
                    elif flag & select.POLLHUP:
                        print(f"[{fd_to_processinfo[fd].process.pid}] Received HUP. Child finished")
                        poller.unregister(fd)
                        items_in_poll -= 1
                    elif flag & select.POLLERR:
                        print(f"[{fd_to_processinfo[fd].process.pid}] Received ERR. Child error")
                        poller.unregister(fd)
                        items_in_poll -= 1
            print(f"Exited poll loop ..")
            for fd, processinfo in fd_to_processinfo.items():
                rc = processinfo.process.returncode
                if rc:
                    print(f"[{processinfo.process.pid}] Process returncode {rc}")
                else:
                    print(f"[{processinfo.process.pid}] success")
                    with open(os.path.join(tmpdir, processinfo.outfn), 'rb') as fp:
                        hh = fp.read()
                    chunk = ''
                    soup = BeautifulSoup(hh, features="html.parser")
                    for para in soup.find_all("p"): 
                        text = para.get_text()
                        nonl_text = text.replace('\n', ' ').replace('\r', '')
                        nonl_ss = re.sub(' +', ' ', nonl_text)
                        #print(nonl_ss)
                        chunk += f"{nonl_ss}\n"
                    rv[processinfo.page-start_page] = chunk
        fnx_end = datetime.datetime.now()
        print(f"tesseract_pages: time taken {fnx_end - fnx_start}")
    except Exception as ex:
        print(f"tesseract_pages: Caught {ex}")
        traceback.print_exc()
    return rv

def read_pdf(filename, fileid, bio, mtime, vectorizer, prev_paras):
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmpfp:
            tmpfp.write(bio.read())
            tmpfp.close()
        tmpdir = tempfile.mkdtemp()
        pages, pdfsize = get_pdf_info(tmpfp.name)
        start_page = len(prev_paras)+1 if prev_paras else 1
        end_page = start_page + 64
        if end_page > pages:
            end_page = pages
        print(f"read_pdf: pages={pages}, pdfsize={pdfsize}, start_page={start_page}, end_page={end_page}")
        if not render_png(tmpfp.name, start_page, end_page, tmpdir):
            return None
        doc_dct={"filename": filename, "fileid": fileid, "mtime": mtime, "paragraphs": prev_paras} 
        chunks:List[str] = tesseract_pages(filename, start_page, end_page, pages, tmpdir)
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
                    print(f"read_pdf: Timelimit exceeded when reading pdf files. Breaking..")
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
