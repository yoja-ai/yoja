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
from utils import respond, get_service_conf, get_user_table_entry, get_user_table_entry_dropbox_sub, set_start_time, set_time_limit
import os.path
import io
from index_utils import update_index_for_user, lock_user, update_next_page_tokens
import jsons
import dataclasses
from typing import List, Dict, Optional, Any
import time
import datetime
import signal
from ecs import get_ecs_task_count, start_ecs_task

@dataclasses.dataclass
class PeriodicBody:
    username:Optional[str] = ''
    dropbox_sub:Optional[str] = ''

def upd_in_lambda(service_conf, email, client, s3client, bucket, prefix, start_time):
    print(f"upd_in_lambda: Entered {email}")
    gdrive_next_page_token, dropbox_next_page_token, status = lock_user(email, client)
    if status:
        if 'ecs_clustername' in service_conf and 'ecs_maxtasks' in service_conf \
                        and 'ecs_subnets' in service_conf and 'ecs_securitygroups' in service_conf:
            ecs_maxtasks = int(service_conf['ecs_maxtasks']['N'])
            ecs_clustername = service_conf['ecs_clustername']['S']
            ecs_subnets = service_conf['ecs_subnets']['S']
            ecs_securitygroups = service_conf['ecs_securitygroups']['S']
            print(f"upd_in_lambda: No gdrive next page token, ecs_clustername={ecs_clustername}, ecs_maxtasks={ecs_maxtasks}, ecs_subnets={ecs_subnets}, ecs_securitygroups={ecs_securitygroups}. Checking num of current ECS tasks")
            try:
                ecs_client = boto3.client('ecs')
                task_count = get_ecs_task_count(ecs_client, ecs_clustername)
                if task_count < ecs_maxtasks:
                    print(f"periodic.upd_in_lambda: task_count {task_count} less than ecs_maxtasks. Kicking off ECS task")
                    update_next_page_tokens(email, client, gdrive_next_page_token, dropbox_next_page_token)
                    start_ecs_task(ecs_client, ecs_clustername, ecs_subnets, ecs_securitygroups, email)
                    return gdrive_next_page_token, dropbox_next_page_token
                else:
                    print(f"periodic.upd_in_lambda: task_count {task_count} greater than ecs_maxtasks. Not kicking off ECS task. Processing in lambda..")
            except Exception as ex:
                print(f"upd_in_lambda: ecs: caught exception {ex}")
        else:
            print(f"upd_in_lambda: gdrive_next_page_token present, or ecs config absent. Processing in lambda..")
        gdrive_next_page_token, dropbox_next_page_token = update_index_for_user(email, s3client,
                                bucket=bucket, prefix=prefix,
                                start_time=start_time,
                                gdrive_next_page_token=gdrive_next_page_token,
                                dropbox_next_page_token=dropbox_next_page_token)
        print(f"periodic.upd: after updating index. gdrive_next_page_token={gdrive_next_page_token}, dropbox_next_page_token={dropbox_next_page_token}")
    else:
        print(f"periodic.upd_in_lambda: failed to get lock for {email}. Returning without doing any work..")
    return gdrive_next_page_token, dropbox_next_page_token

def upd_in_non_lambda(service_conf, email, client, s3client, bucket, prefix, start_time):
    print(f"periodic.upd_in_non_lambda: Entered {email}")
    if 'YOJA_TAKEOVER_LOCK_END_TIME' in os.environ:
        print(f"periodic.upd_in_non_lambda: {email} YOJA_TAKEOVER_LOCK_END_TIME present. Trying to lock")
        gdrive_next_page_token, dropbox_next_page_token, status = lock_user(email, client,
                                    takeover_lock_end_time=int(os.environ['YOJA_TAKEOVER_LOCK_END_TIME']))
    else:
        print(f"periodic.upd_in_non_lambda: {email} YOJA_TAKEOVER_LOCK_END_TIME absent. Trying to lock")
        gdrive_next_page_token, dropbox_next_page_token, status = lock_user(email, client)
    if status:
        gdrive_next_page_token, dropbox_next_page_token = update_index_for_user(email, s3client,
                            bucket=bucket, prefix=prefix,
                            start_time=start_time,
                            gdrive_next_page_token=gdrive_next_page_token,
                            dropbox_next_page_token=dropbox_next_page_token)
        print(f"periodic.upd_in_non_lambda: after updating index. gdrive_next_page_token={gdrive_next_page_token}, dropbox_next_page_token={dropbox_next_page_token}")
        update_next_page_tokens(email, client, gdrive_next_page_token, dropbox_next_page_token)
    else:
        print(f"periodic.upd_in_non_lambda: failed to get lock for {email}. Returning without doing any work..")
    return gdrive_next_page_token, dropbox_next_page_token

def upd(service_conf, email, client, s3client, bucket, prefix, start_time):
    if 'AWS_LAMBDA_FUNCTION_NAME' in os.environ:
        return upd_in_lambda(service_conf, email, client, s3client, bucket, prefix, start_time)
    else:
        return upd_in_non_lambda(service_conf, email, client, s3client, bucket, prefix, start_time)

def do_full_scan(service_conf, s3client, client, bucket, prefix, start_time):
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
                    upd(service_conf, email, client, s3client, bucket, prefix, start_time)
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

def update_gdrive_user(service_conf, s3client, client, email, bucket, prefix, start_time):
    item = get_user_table_entry(email)
    if not item:
        print(f"update_gdrive_user: Hmm. user table entry not found for {email}")
        return respond({"error_msg": f"update_gdrive_user: Hmm. user table entry not found for {email}"}, status=403)
    gdrive_next_page_token, dropbox_next_page_token = upd(service_conf, email, client, s3client, bucket, prefix, start_time)
    res={'version': os.environ['LAMBDA_VERSION']}
    if gdrive_next_page_token:
        res['gdrive_next_page_token'] = gdrive_next_page_token
    if dropbox_next_page_token:
        res['dropbox_next_page_token'] = dropbox_next_page_token
    return respond(None, res=res)

def update_dropbox_user(service_conf, s3client, client, dropbox_sub, bucket, prefix, start_time):
    item = get_user_table_entry_dropbox_sub(dropbox_sub)
    if not item:
        print(f"update_dropbox_user: Hmm. user table entry not found for dropbox_sub {dropbox_sub}")
        return respond({"error_msg": f"update_dropbox_user: Hmm. user table entry not found for dropbox_sub{dropbox_sub}"}, status=403)
    gdrive_next_page_token, dropbox_next_page_token = upd(service_conf, item['email']['S'], client, s3client, bucket, prefix, start_time)
    res={'version': os.environ['LAMBDA_VERSION']}
    if gdrive_next_page_token:
        res['gdrive_next_page_token'] = gdrive_next_page_token
    if dropbox_next_page_token:
        res['dropbox_next_page_token'] = dropbox_next_page_token
    return respond(None, res=res)

########
#  use this payload in EventBridge Schedules to setup hourly runs:
## GDrive
## {"requestContext": { "http": { "method":"POST", "path": "/rest/entrypoint/periodic" } }, "body":"{\"username\":\"raj@yoja.ai\"}" }
## Dropbox
## {"requestContext": { "http": { "method":"POST", "path": "/rest/entrypoint/periodic" } }, "body":"{\"dropbox_sub\":\"dbid:AACGAAAlllfungvklgiugkfvknskshdhhXM\"}" }
## Full Scan
## {"requestContext": { "http": { "method":"POST", "path": "/rest/entrypoint/periodic" } }}
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
    set_start_time(start_time)
    if post_body:
        if post_body.username:
            print(f"periodic: post_body contains username {post_body.username}. Updating")
            return update_gdrive_user(service_conf, s3client, client, post_body.username, bucket, prefix, start_time)
        elif post_body.dropbox_sub:
            print(f"periodic: post_body contains dropbox_sub {post_body.dropbox_sub}. Updating")
            return update_dropbox_user(service_conf, s3client, client, post_body.dropbox_sub, bucket, prefix, start_time)
    print(f"periodic: no post_body or post_body does not contain username or dropbox_sub. Doing a full scan")
    return do_full_scan(service_conf, s3client, client, bucket, prefix, start_time)

# You can invoke and run the periodic lambda in your local machine as follows
#
# OAUTH_CLIENT_ID='123456789012-abcdefghijklmnopqrstuvwxuzabcdef.apps.googleusercontent.com' OAUTH_CLIENT_SECRET='GOCSPX-ABCDEFGHIJKLMNOPQRSTUVWXUB01' OAUTH_REDIRECT_URI='https://chat.example.ai/rest/entrypoint/oauth2cb' AWS_PROFILE=example.ai AWS_DEFAULT_REGION=us-east-1 PERIOIDIC_PROCESS_FILES_TIME_LIMIT=240 USERS_TABLE=yoja-users SERVICECONF_TABLE=yoja-ServiceConf LAMBDA_VERSION=dummy YOJA_FORCE_FULL_INDEX='true' python periodic.py  example.email@gmail.com
#
# This will use boto3 to read the ddb table yoja-users, get the gdrive access token, use the token to read and process the files and generate the index
# You will need a conda environment with all the right packages. The easiest way is to run create-container.sh.
# Then run 'docker run --interactive --tty --entrypoint /bin/bash yoja-img'
# When the docker container starts up, copy the aws credentials profile in the above command (example.ai) into a file called credentials in /var/task. Then add AWS_SHARED_CREDENTIALS_FILE=./credentials to the above command line.
#
def exit_gracefully(signum, frame):
    print(f"Received {signum} signal")
    set_time_limit(0)

if __name__=="__main__":
    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)

    event = {'requestContext': {'http': {'method': 'POST', 'path': '/rest/entrypoint/periodic'}}}
    if len(sys.argv) < 2:
        if 'YOJA_USER' in os.environ:
            event['body'] = json.dumps({'username': os.environ['YOJA_USER']})
        else:
            print(f"Usage: periodic.py user_email")
            sys.exit(255)
    else:
        event['body'] = json.dumps({'username': sys.argv[1]})
    res = periodic(event, None)
    if 'gdrive_next_page_token' in res:
        sys.exit(0)
    else:
        sys.exit(1)
