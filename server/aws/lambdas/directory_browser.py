import io
import json
import os
from utils import respond, check_cookie, get_service_conf
import boto3
from index_utils import get_s3_index

def directory_browser(event, context):
    operation = event['requestContext']['http']['method']
    if (operation != 'POST'):
        print(f"Error: unsupported method: operation={operation}")
        return respond({"error_msg": str(ValueError('Unsupported method ' + str(operation)))}, status=400)
    if 'INDEX_DIR' in os.environ:
        return respond(None, res={'directories': [], 'files': []})
    rv = check_cookie(event, False)
    email = rv['google']
    if not email:
        print(f"directory_browser: check_cookie did not return email. Sending 403")
        return respond({"status": "Unauthorized: please login using google"}, 403, None)
    try:
        service_conf = get_service_conf()
    except Exception as ex:
        print(f"Caught {ex} while getting service_conf")
        return respond({"error_msg": f"Caught {ex} while getting service_conf"}, status=403)
    if 'bucket' not in service_conf or 'prefix' not in service_conf:
        print(f"Error. bucket and prefix not specified in service conf")
        return respond({"error_msg": "Error. bucket and prefix not specified in service_conf"}, status=403)
    bucket = service_conf['bucket']['S']
    prefix = service_conf['prefix']['S'].strip().strip('/')
    print(f"Index Location: s3://{bucket}/{prefix}")
    if not 'body' in event:
        return respond({"error_msg": "Error. body not present"}, status=400)
    body = json.loads(event['body'])
    print(f"directory_browser: body={body}")
    if 'parentdir' not in body or not body['parentdir']:
        resp = {'directories': [], 'files': []}
        resp['directories'].append('[gdrive]')
        if 'dropbox' in rv:
            resp['directories'].append('[dropbox]')
        return respond(None, res=resp)
    parentdir = body['parentdir']
    mimetype=None
    if 'mimetype' in body:
        print(f"In addition to directories, files of mimetype {mimetype} requested")
        mimetype = body['mimetype']
    s3client = boto3.client('s3')
    storage = ''
    if parentdir.startswith('[gdrive]'):
        try:
            s3_index = get_s3_index(s3client, bucket, f"{prefix}/{email}")
            parentdir = parentdir[8:].lstrip('/')
            storage = '[gdrive]'
        except Exception as ex:
            print(f"directory_browser: Exception occurred: {ex}")
            return respond({"error_msg": "parentdir not found 2"}, status=404)
    elif parentdir.startswith('[dropbox]'):
        try:
            s3_index = get_s3_index(s3client, bucket, f"{prefix}/{email}/dropbox")
            parentdir = parentdir[9:].lstrip('/')
            storage = '[dropbox]'
        except Exception as ex:
            print(f"directory_browser: Exception occurred: {ex}")
            return respond({"error_msg": "parentdir not found 4"}, status=404)
    else:
        return respond({"error_msg": "parentdir not found 3"}, status=404)
    print(f"directory_browser: parentdir={parentdir}, storage={storage}. Start of scan..")
    dirs = {}
    for ky, vl in s3_index.items():
        if 'path' in vl and vl['path']:
            pth = ''
            vlpa = vl['path'].lstrip('/').rstrip('/').split('/')
            for ind in range(len(vlpa)):
                pth = pth + vlpa[ind] + '/'
                dirs[pth] = 0
    rdirs = {}
    if parentdir:
        spdir = parentdir.lstrip('/').rstrip('/').split('/')
    else:
        spdir = []
    for ky1 in dirs.keys():
        kya = ky1.lstrip('/').rstrip('/').split('/')
        print(f"directory_browser: begin eval. spdir={spdir}, kya={kya}")
        if len(kya) == (len(spdir) + 1):
            print(f"directory_browser: length is right. spdir={spdir}, kya={kya}")
            add = True
            for ind in range(len(spdir)):
                if spdir[ind] != kya[ind]:
                    add = False
                    break
            if add:
                print(f"directory_browser: Adding. spdir={spdir}, kya={kya}, add={kya[-1]}")
                rdirs[kya[-1]] = 0
    ra = []
    for ky in rdirs.keys():
        ra.append(str(ky))
    # Add files of mimetype in the listing
    files = []
    for ky, vl in s3_index.items():
        if mimetype and vl['mimetype'] != mimetype:
            print(f"skipping for wrong mimetype: {vl}")
            continue
        if 'path' in vl and vl['path']:
            vlpa = vl['path'].lstrip('/').rstrip('/').split('/')
            if len(vlpa) == len(spdir):
                add = True
                for ind in range(len(vlpa)):
                    if vlpa[ind] != spdir[ind]:
                        print(f"not adding 1 {vl['filename']}, {vl['path']}")
                        add = False
                        break
                if add:
                    print(f"adding 1 {vl['filename']}, {vl['path']}")
                    files.append(vl['filename'])
            else:
                print(f"not adding for wrong path len {vl['filename']}, {vl['path']}")
        else:
            if not spdir:
                print(f"adding 2 {vl['filename']}, {vl['path']}")
                files.append(vl['filename'])
            else:
                print(f"not adding 2 {vl['filename']}, {vl['path']}")
    print(f"directory_browser: parentdir = {body['parentdir']}, returning {ra}")
    res={'directories': ra}
    res['files'] = files
    return respond(None, res=res)
