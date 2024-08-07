import io
import json
import os
import sys
import traceback
import tempfile
import uuid
import base64
import zlib
from urllib.parse import unquote
import faiss
import boto3
from utils import respond, get_service_conf, get_user_table_entry, get_user_table_entry_dropbox_sub
import os.path
import io
from index_utils import update_index_for_user, lock_user, unlock_user
import jsons
import dataclasses
from typing import List, Dict, Optional, Any
import time
import datetime

@dataclasses.dataclass
class PeriodicBody:
    username:Optional[str] = ''
    dropbox_sub:Optional[str] = ''

def upd(client, item, s3client, bucket, prefix, start_time):
    gdrive_next_page_token, dropbox_next_page_token, status = lock_user(item, client)
    if status:
        print(f"periodic.upd: before updating index. gdrive_next_page_token={gdrive_next_page_token}, dropbox_next_page_token={dropbox_next_page_token}")
        gdrive_next_page_token, dropbox_next_page_token = update_index_for_user(item, s3client,
                                                bucket=bucket, prefix=prefix,
                                                start_time=start_time, only_create_index=False,
                                                gdrive_next_page_token=gdrive_next_page_token, dropbox_next_page_token=dropbox_next_page_token)
        print(f"periodic.upd: after updating index. gdrive_next_page_token={gdrive_next_page_token}, dropbox_next_page_token={dropbox_next_page_token}")

        unlock_user(item, client, gdrive_next_page_token, dropbox_next_page_token)
    return

def do_full_scan(s3client, client, bucket, prefix, start_time):
    try:
        last_evaluated_key = None
        while True:
            if last_evaluated_key:
                resp = client.scan(TableName=os.environ['USERS_TABLE'], Select='ALL_ATTRIBUTES',
                    ExclusiveStartKey=last_evaluated_key)
            else:
                resp = client.scan(TableName=os.environ['USERS_TABLE'], Select='ALL_ATTRIBUTES')
            if 'Items' in resp:
                for item in resp['Items']:
                    email = item['email']['S']
                    print(f"do_full_scan: Updating user {email}")
                    upd(client, item, s3client, bucket, prefix, start_time)
                if 'LastEvaluatedKey' in resp:
                    last_evaluated_key = resp['LastEvaluatedKey']
                else:
                    break
            else:
                break
    except Exception as ex:
        print(f"Caught {ex} while scanning users table")
        traceback.print_exc(ex)
        return respond({"error_msg": f"Caught {ex} while scanning users table"}, status=403)
    return respond(None, res={'version': os.environ['LAMBDA_VERSION']})

def update_gdrive_user(s3client, client, email, bucket, prefix, start_time):
    item = get_user_table_entry(email)
    if not item:
        print(f"update_gdrive_user: Hmm. user table entry not found for {email}")
        return respond({"error_msg": f"update_gdrive_user: Hmm. user table entry not found for {email}"}, status=403)
    upd(client, item, s3client, bucket, prefix, start_time)
    return respond(None, res={'version': os.environ['LAMBDA_VERSION']})

def update_dropbox_user(s3client, client, dropbox_sub, bucket, prefix, start_time):
    item = get_user_table_entry_dropbox_sub(dropbox_sub)
    if not item:
        print(f"update_dropbox_user: Hmm. user table entry not found for dropbox_sub {dropbox_sub}")
        return respond({"error_msg": f"update_dropbox_user: Hmm. user table entry not found for dropbox_sub{dropbox_sub}"}, status=403)
    upd(client, item, s3client, bucket, prefix, start_time)
    return respond(None, res={'version': os.environ['LAMBDA_VERSION']})

########
#  use this payload in EventBridge Schedules to setup hourly runs:
## GDrive
## {"requestContext": { "http": { "method":"POST", "path": "/rest/entrypoint/periodic" } }, "body":"{\"username\":\"raj@yoja.ai\"}" }
## Dropbox
## {"requestContext": { "http": { "method":"POST", "path": "/rest/entrypoint/periodic" } }, "body":"{\"dropbox_sub\":\"dbid:AACGAAAlllfungvklgiugkfvknskshdhhXM\"}" }
#######
def periodic(event:dict, context) -> dict:
    body_str:str = None
    if event.get('requestContext'):
        # event/body
        body_str:str = event['body'] if event.get('body') else None
    else:
        # event/details/body
        details:dict = event['details'] if event.get('details') else None
        if details: body_str = details.get('body')

    post_body:PeriodicBody = jsons.loads(body_str, PeriodicBody) if event.get('body') else None

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
    client = boto3.client('dynamodb')
    start_time:datetime.datetime = datetime.datetime.now()
    if post_body.username:
        print(f"periodic: post_body contains username {post_body.username}. Updating")
        return update_gdrive_user(s3client, client, post_body.username, bucket, prefix, start_time)
    elif post_body.dropbox_sub:
        print(f"periodic: post_body contains dropbox_sub {post_body.dropbox_sub}. Updating")
        return update_dropbox_user(s3client, client, post_body.dropbox_sub, bucket, prefix, start_time)
    else:
        print(f"periodic: post_body does not contain username or dropbox_sub. Doing a full scan")
        return do_full_scan(s3client, client, bucket, prefix, start_time)

# You can invoke and run the periodic lambda in your local machine as follows
#
# OAUTH_CLIENT_ID='123456789012-abcdefghijklmnopqrstuvwxuzabcdef.apps.googleusercontent.com' OAUTH_CLIENT_SECRET='GOCSPX-ABCDEFGHIJKLMNOPQRSTUVWXUB01' OAUTH_REDIRECT_URI='https://chat.example.ai/rest/entrypoint/oauth2cb' AWS_PROFILE=example.ai AWS_DEFAULT_REGION=us-east-1 PERIOIDIC_PROCESS_FILES_TIME_LIMIT=240 SERVICECONF_TABLE=yoja-ServiceConf LAMBDA_VERSION=dummy python periodic.py  example.email@gmail.com
#
# This will use boto3 to read the ddb table yoja-users, get the gdrive access token, use the token to read and process the files and generate the index
if __name__=="__main__":
    if len(sys.argv) < 2:
        print(f"Usage: periodic.py user_email")
        sys.exit(255)
    event = {
            'requestContext': {'http': {'method': 'POST', 'path': '/rest/entrypoint/periodic'}},
            'body': json.dumps({'username': sys.argv[1]})
        }
    periodic(event, None)
    sys.exit(0)
