import io
import json
import os
from utils import respond, check_cookie, get_service_conf
import boto3

def read_indexing_progress(s3client, bucket, prefix, storage_prefix):
    pfx = f"{prefix}/{storage_prefix}indexing_progress.json"
    try:
        s3client.download_file(bucket, pfx, "/tmp/indexing_progress.json")
        with open('/tmp/indexing_progress.json', 'r') as fl:
            js = json.load(fl)
        return js
    except Exception as ex:
        print(f"Caught {ex} while reading indexing progress file {pfx}")
        return {}

def get_indexing_progress(event, context):
    operation = event['requestContext']['http']['method']
    if (operation != 'GET'):
        print(f"Error: unsupported method: operation={operation}")
        return respond({"error_msg": str(ValueError('Unsupported method ' + str(operation)))}, status=400)
    rv = check_cookie(event, False)
    email = rv['google']
    if not email:
        print(f"get_indexing_progress: check_cookie did not return email. Sending 403")
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
    s3client = boto3.client('s3')
    js_gdrive = read_indexing_progress(s3client, bucket, f"{prefix}/{email}", "")
    js_dropbox = read_indexing_progress(s3client, bucket, f"{prefix}/{email}", "dropbox/")
    rv = {}
    if 'gdrive_num_unmodified' in js_gdrive:
        rv['gdrive_num_unmodified'] = js_gdrive['gdrive_num_unmodified']
    if 'dropbox_num_unmodified' in js_dropbox:
        rv['dropbox_num_unmodified'] = js_dropbox['dropbox_num_unmodified']
    if 'gdrive_unmodified_size' in js_gdrive:
        rv['gdrive_unmodified_size'] = js_gdrive['gdrive_unmodified_size']
    if 'dropbox_unmodified_size' in js_dropbox:
        rv['dropbox_unmodified_size'] = js_dropbox['dropbox_unmodified_size']
    if 'gdrive_num_needs_embedding' in js_gdrive:
        rv['gdrive_num_needs_embedding'] = js_gdrive['gdrive_num_needs_embedding']
    if 'dropbox_num_needs_embedding' in js_dropbox:
        rv['dropbox_num_needs_embedding'] = js_dropbox['dropbox_num_needs_embedding']
    if 'gdrive_needs_embedding_size' in js_gdrive:
        rv['gdrive_needs_embedding_size'] = js_gdrive['gdrive_needs_embedding_size']
    if 'dropbox_needs_embedding_size' in js_dropbox:
        rv['dropbox_needs_embedding_size'] = js_dropbox['dropbox_needs_embedding_size']
    print(f"get_indexing_progress: rv={rv}")
    return respond(None, res=rv)
