import io
import json
import os
import traceback
import tempfile
import uuid
import time
import base64
import zlib
import datetime
from datetime import timezone
from urllib.parse import unquote
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import boto3
import sys
import base64
from botocore.exceptions import ClientError
from cryptography.fernet import Fernet
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from typing import Union, Dict, List, Any, Tuple

SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly",
          "https://www.googleapis.com/auth/drive.readonly",
          "https://www.googleapis.com/auth/userinfo.email",
          "https://www.googleapis.com/auth/userinfo.profile"]

def is_lambda_debug_enabled() -> bool:
     return os.getenv('LAMBDA_LOG_LEVEL', 'INFO').lower() == 'debug'

def respond(err, status=None, res=None):
    if status:
        statusCode = status
    elif err:
        statusCode = 400
    else:
        statusCode = 200
    return {
        'statusCode': statusCode,
        'body': json.dumps(err) if err else json.dumps(res),
        'headers': {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache, no-store, must-revalidate, private',
            'Pragma': 'no-cache',
            'Expires': '0'
        },
    }

def parse_id_token(token):
    parts = token.split(".")
    if len(parts) != 3:
        raise Exception("Incorrect id token format")
    payload = parts[1]
    padded = payload + '=' * (4 - len(payload) % 4)
    decoded = base64.b64decode(padded)
    return json.loads(decoded)

def create_service_conf_entry(client):
    try:
        key = Fernet.generate_key()
        key_s = key.decode()
        resp = client.put_item(TableName=os.environ['SERVICECONF_TABLE'],
                                Item={'configVersion': {'N': str(1)}, 'key': {'S': key_s}})
    except Exception as ex:
        print(f"Caught {ex} while creating new service conf entry")

def add_key(service_conf):
    try:
        key = Fernet.generate_key()
        key_s = key.decode()
        boto3.client('dynamodb').update_item(
                        TableName=os.environ['SERVICECONF_TABLE'],
                        Key={'configVersion': {'N': str(1)}},
                        UpdateExpression='SET #ky = :ky',
                        ExpressionAttributeNames={'#ky': 'key'},
                        ExpressionAttributeValues={':ky': {'S': key_s}}
                    )
    except Exception as ex:
        print(f"add_key: caught {ex} while adding key to service config entry")
        traceback.print_exc()

cached_service_conf = None
def get_service_conf():
    global cached_service_conf
    if cached_service_conf:
        return cached_service_conf
    if 'SERVICECONF_TABLE' not in os.environ:
        cached_service_conf = {}
        return cached_service_conf
    client = boto3.client('dynamodb')
    for ind in range(2):
        try:
            resp = client.get_item(TableName=os.environ['SERVICECONF_TABLE'],
                                            Key={'configVersion': {'N': str(1)}})
            print(f"get_service_conf: resp={resp}")
            if 'Item' in resp:
                cached_service_conf = resp['Item']
                if 'YOJA_INDEX_BUCKET' in os.environ:
                    cached_service_conf['bucket'] = {'S': os.environ['YOJA_INDEX_BUCKET']}
                if 'YOJA_INDEX_PREFIX' in os.environ:
                    cached_service_conf['prefix'] = {'S': os.environ['YOJA_INDEX_PREFIX']}
                return cached_service_conf
            else:
                if ind == 0:
                    print(f"No item. serviceconf table entry for version 1. Creating a new entry")
                    create_service_conf_entry(client)
                else:
                    print(f"No item. serviceconf table entry for version 1 not found. Throwing exception")
                    raise Exception("Cannot find service conf")
        except ClientError as err:
            print(f"get_service_conf: caught ClientError {err}")
            if err.response['Error']['Code'] == 'ResourceNotFoundException':
                if ind == 0:
                    print(f"serviceconf table entry for version 1 not found. Creating a new entry")
                    create_service_conf_entry(client)
                else:
                    print(f"serviceconf table entry for version 1 not found. Throwing exception")
                    raise Exception("Cannot find service conf")
            else:
                print(f"get_service_conf: err.response.code={err.response['Error']['Code']}")
        except Exception as ex:
            print(f"get_service_conf: caught {ex}")
    raise Exception("Unknown error looking up service conf")

def encrypt_email(email, service_conf):
    if not 'key' in service_conf:
        add_key(service_conf)
        global cached_service_conf
        cached_service_conf = None
        service_conf = get_service_conf()
        print(f"encrypt_email: updated service_conf={service_conf}")
    fky = Fernet(service_conf['key']['S'])
    return fky.encrypt(email.encode()).decode()

user_table_cache = {}
def get_user_table_entry(email):
    global user_table_cache
    if email in user_table_cache:
        return user_table_cache[email]
    try:
        if 'USERS_TABLE' not in os.environ:
            user_table_cache[email] = {'email': {'S': email}}
        else:
            client = boto3.client('dynamodb')
            response = client.get_item(TableName=os.environ['USERS_TABLE'], Key={'email': {'S': email}}, ConsistentRead=True)
            user_table_cache[email] = response['Item']
        return user_table_cache[email]
    except Exception as ex:
        print(f"Caught {ex} while getting info for {email} from users table")
        return None

def set_user_table_cache_entry(email, entry):
    global user_table_cache
    user_table_cache[email] = entry

def get_user_table_entry_dropbox_sub(dropbox_sub):
    try:
        client = boto3.client('dynamodb')
        result = client.query(TableName=os.environ['USERS_TABLE'],
                            IndexName="dropbox_sub-index",
                            KeyConditionExpression = 'dropbox_sub = :ds',
                            ExpressionAttributeValues={':ds': {'S': dropbox_sub}})
        if (result and 'Items' in result and len(result['Items']) == 1):
            return result['Items'][0]
        else:
            print(f"Error getting info for dropbox_sub {dropbox_sub}, result={result}")
            return None
    except Exception as ex:
        print(f"Caught {ex} while getting info for dropbox_sub {dropbox_sub} from users table")
        return None

def init_gdrive_webhook(item, email, service_conf):
    if 'gdrive_next_page_token' in item:
        start_page_token = item['gdrive_next_page_token']['S']
    else:
        start_page_token = "1"
    resp = None
    e_email = encrypt_email(email, service_conf)
    try:
        resource_id = str(uuid.uuid4())
        addr = f"{os.environ['OAUTH_REDIRECT_URI'][:-24]}webhook/webhook_gdrive"
        headers = {"Authorization": f"Bearer {item['access_token']['S']}", "Content-Type": "application/json"}
        print(f"init_gdrive_webhook: resource_id {resource_id}, address={addr} for email {email}")
        postdata={'id': resource_id,
                'type': 'web_hook',
                'address': addr,
                'token': e_email,
                'expiration': (time.time_ns()//1_000_000+(7*24*60*60*1000))}
        params={'includeCorpusRemovals': True,
                'includeItemsFromAllDrives': True,
                'includeRemoved': True,
                'pageToken': start_page_token,
                'pageSize': 100,
                'restrictToMyDrive': False,
                'spaces': 'drive',
                'supportsAllDrives': True}
        print(f"init_gdrive_webhook: watch. hdrs={json.dumps(headers)}, postdata={json.dumps(postdata)}, params={json.dumps(params)}")
        resp = requests.post('https://www.googleapis.com/drive/v3/changes/watch', headers=headers, json=postdata, params=params)
        resp.raise_for_status()
        print(f"init_gdrive_webhook: post resp.text={resp.text}")
    except Exception as ex:
        if resp:
            print(f"init_gdrive_webhook: In changes.watch for user {email}, caught {ex}. Response={resp.content}")
        else:
            print(f"init_gdrive_webhook: In changes.watch for user {email}, caught {ex}")
        return False
    try:
        response = boto3.client('dynamodb').update_item(
                        TableName=os.environ['USERS_TABLE'],
                        Key={'email': {'S': email}},
                        UpdateExpression="SET gdrive_next_page_token = :tk",
                        ExpressionAttributeValues={':tk': {'S': start_page_token}},
                        ReturnValues="ALL_NEW"
                    )
        set_user_table_cache_entry(email, response['Attributes'])
    except Exception as ex:
        print(f"init_gdrive_webhook: Error. Caught {ex} while saving nextPageToken in yoja-users")
        return False
    return True

def check_webhook_gdrive(email):
    service_conf = get_service_conf()
    item = get_user_table_entry(email)
    if 'gw_expires' in item:
        gw_exp_time = datetime.datetime.strptime(item['gw_expires']['S'], '%a, %d %b %Y %H:%M:%S %Z')
        now = datetime.datetime.now()
        if now > gw_exp_time:
            print(f"check_webhook_gdrive: now={now} after gdrive webhook expiry {gw_exp_time}. Initializing gdrive webhook")
            init_gdrive_webhook(item, email, service_conf)
        else:
            print(f"check_webhook_gdrive: now={now} before gdrive webhook expiry {gw_exp_time}. Not initializing gdrive webhook")
    else:
        print(f"check_webhook_gdrive: gw_expires not present in user table. Initializing gdrive webhook")
        init_gdrive_webhook(item, email, service_conf)

def check_user(service_conf, cookie_val:str, refresh_access_token, user_type):
    print(f"check_user:= Entered. cookie_val={cookie_val}")
    try:
        fky = Fernet(service_conf['key']['S'])
        email = fky.decrypt(cookie_val.encode() if isinstance(cookie_val,str) else cookie_val ).decode()
        print(f"check_user: decrypted email={email}")
        if refresh_access_token:
            print(f"check_user: refreshing access token")
            item = get_user_table_entry(email)
            if not item:
                print(f"check_user: weird error - Cannot find user {email} in user table")
                return None
            if user_type == 'google':
                refresh_user_google(item)
            elif user_type == 'dropbox':
                refresh_user_dropbox(item)
        return email
    except Exception as ex:
        print(f"check_user: cookie={cookie_val}, refresh_access_token={refresh_access_token}, user_type={user_type}: caught {ex}")
        traceback.print_exc()
        return None

def check_cookie(event, refresh_access_token):
    service_conf = get_service_conf()
    rv = {'google': '', 'dropbox': '', 'fullname': '', 'picture': ''}
    for hdr in event['headers']:
        if hdr.lower() == 'cookie':
            cookies = event['headers'][hdr].split(';')
            print(f"check_cookie: cookies={cookies}")
            for cookie in cookies:
                cookie=cookie.strip()
                print(f"check_cookie: processing cookie={cookie}")
                ind=cookie.find('=')
                if ind > 0:
                    cookie_name=cookie[:ind]
                    cookie_val=cookie[ind+1:]
                    print(f"check_cookie: key={cookie_name} val={cookie_val}")
                    cn = cookie_name.strip()
                    if cn == '__Host-yoja-user':
                        email = check_user(service_conf, cookie_val.strip(), refresh_access_token, 'google')
                        if email:
                            rv['google'] = email
                            item = get_user_table_entry(email)
                            print(f"check_cookie: item={item}")
                            if 'fullname' in item:
                                rv['fullname'] = item['fullname']['S']
                            if 'picture' in item:
                                rv['picture'] = item['picture']['S']
                    elif cn == '__Host-yoja-dropbox-user':
                        dropbox_email = check_user(service_conf, cookie_val.strip(), refresh_access_token, 'dropbox')
                        if dropbox_email:
                            rv['dropbox'] = dropbox_email
                    elif cn == '__Host-yoja-searchsubdir':
                        rv['searchsubdir'] = cookie_val.strip()
    return rv

def update_users_table(email, refresh_token, access_token, expires_in, id_token=None, fullname=None, picture=None):
    print(f"update_users_table: Entered. email={email}, ref={refresh_token},"
            f" acc={access_token}, expires_in={expires_in}, id_token={id_token},"
            f" fullname={fullname}, picture={picture}")
    try:
        ue="SET refresh_token = :rt, access_token = :at, created = :ct, expires_in = :exp"
        eav={
            ':rt': {'S': refresh_token},
            ':at': {'S': access_token},
            ':ct': {'N': str(int(time.time()))},
            ':exp':{'N': str(expires_in)}
            }
        if id_token:
            ue = f"{ue}, id_token = :idt"
            eav[':idt'] = {'S': id_token}
        if fullname:
            ue = f"{ue}, fullname = :fn"
            eav[':fn'] = {'S': fullname}
        if picture:
            ue = f"{ue}, picture = :pc"
            eav[':pc'] = {'S': picture}
        response = boto3.client('dynamodb').update_item(
                    TableName=os.environ['USERS_TABLE'],
                    Key={'email': {'S': email}},
                    UpdateExpression=ue,
                    ExpressionAttributeValues=eav,
                    ReturnValues="ALL_NEW"
                )
        set_user_table_cache_entry(email, response['Attributes'])
        return True
    except Exception as ex:
        print(f"Caught {ex} while updating users table")
        traceback.print_exc()
        return False

def refresh_user_google(item) -> Credentials:
    """ refresh the credentials of the user stored in 'item' if necessary.  Also updates the credentials in the yoja-users table, if refreshed """
    email = item['email']['S']
    refresh_token = item['refresh_token']['S']
    access_token = item['access_token']['S']
    id_token = item['id_token']['S']
    created = int(item['created']['N'])
    expires_in = int(item['expires_in']['N'])
    print(f"refresh_user_google: user_type=google, email={email}, access token created={datetime.datetime.fromtimestamp(created)}")
    token={"access_token": None,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "scope": " ".join(SCOPES),
            "created": created,
            "expires_in": None,
            "client_id": os.environ['OAUTH_CLIENT_ID'],
            "client_secret": os.environ['OAUTH_CLIENT_SECRET']}
    with open("/tmp/token.json", "w") as fp:
        fp.write(json.dumps(token))
    creds = Credentials.from_authorized_user_file("/tmp/token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
    if not creds or not creds.valid:
        print("Creds not valid!")
        raise Exception("Creds not valid!")
    print(f"creds after refresh={creds.to_json()}")
    cj = json.loads(creds.to_json())
    now = datetime.datetime.utcnow()
    expiry = datetime.datetime.strptime(cj['expiry'], "%Y-%m-%dT%H:%M:%S.%fZ")
    expires_in = (expiry - now).total_seconds()
    update_users_table(email, cj['refresh_token'], cj['token'], int(expires_in))
    check_webhook_gdrive(email)
    return creds

def refresh_user_dropbox(item):
    try:
        email = item['email']['S']
        refresh_token = item['dropbox_refresh_token']['S']
        postdata={'client_id': os.environ['DROPBOX_OAUTH_CLIENT_ID'],
                'client_secret': os.environ['DROPBOX_OAUTH_CLIENT_SECRET'], 
                'refresh_token': refresh_token,
                'grant_type': 'refresh_token'}
        resp = requests.post('https://www.dropbox.com/oauth2/token', data=postdata)
        resp.raise_for_status()
        print(f"refresh access token post resp.text={resp.text}")
        rj = json.loads(resp.text)
        created=int(time.time())
        expires_in = rj['expires_in']
        access_token=rj['access_token']
    except Exception as ex:
        print(f"while refreshing dropbox access token, post caught {ex}")
        return None
    try:
        response = boto3.client('dynamodb').update_item(
                        TableName=os.environ['USERS_TABLE'],
                        Key={'email': {'S': email}},
                        UpdateExpression="SET dropbox_access_token = :at, dropbox_created = :ct, dropbox_expires_in = :exp",
                        ExpressionAttributeValues={':at': {'S': access_token}, ':ct': {'N': str(int(time.time()))}, ':exp':{'N': str(expires_in)} },
                        ReturnValues="ALL_NEW"
                    )
        set_user_table_cache_entry(email, response['Attributes'])
    except Exception as ex:
        print(f"Caught {ex} while saving dropbox_access_token, dropbox_refresh_token for {email}")
        return None
    return access_token

g_start_time:datetime.datetime = None # initialized further below
g_time_limit = int(os.getenv("PERIODIC_PROCESS_FILES_TIME_LIMIT", 12))*60

def set_start_time(start_time):
    global g_start_time
    g_start_time = start_time

def set_time_limit(time_limit):
    global g_time_limit
    g_time_limit = time_limit

def extend_lock_time(email, index_dir, time_left):
    if 'AWS_LAMBDA_FUNCTION_NAME' not in os.environ:
        item = get_user_table_entry(email)
        if not item:
            print(f"extend_lock_time: Error. Cannot get user entry for {email}")
            return
        now = time.time()
        now_s = datetime.datetime.fromtimestamp(now).strftime('%Y-%m-%d %I:%M:%S')
        if 'lock_end_time' in item:
            l_e_t = int(item['lock_end_time']['N'])
            l_e_t_s = datetime.datetime.fromtimestamp(l_e_t).strftime('%Y-%m-%d %I:%M:%S')
            # if lock_end_time is less than 3 minutes away, push it out by 12 minutes
            if (l_e_t - int(now)) < (3 * 60):
                time_to_add = 12*60
                if index_dir:
                    ute = {'email': {'S': email}, 'lock_end_time': {'N': str(int(now)+time_to_add)}}
                    set_user_table_cache_entry(email, ute)
                else:
                    try:
                        response = boto3.client('dynamodb').update_item(
                            TableName=os.environ['USERS_TABLE'],
                            Key={'email': {'S': email}},
                            UpdateExpression="set #lm = :st",
                            ConditionExpression=f"#lm = :ev",
                            ExpressionAttributeNames={'#lm': 'lock_end_time'},
                            ExpressionAttributeValues={':ev': {'N': item['lock_end_time']['N']}, ':st': {'N': str(int(now)+time_to_add)} },
                            ReturnValues="ALL_NEW"
                        )
                        set_user_table_cache_entry(email, response['Attributes'])
                    except ClientError as e:
                        if e.response['Error']['Code'] == "ConditionalCheckFailedException":
                            # This should not happen
                            print(f"extend_lock_time: conditional check failed. {e.response['Error']['Message']}. Another instance of lambda is active for {email}")
                            g_time_limit = 0
                        else:
                            print(f"extend_lock_time: Exception. Non CCFE. Re-raising {e}")
                            raise
        else:
            print(f"extend_lock_time: Error. No lock_end_time entry for user {email}")

def lambda_timelimit_exceeded() -> bool:
    global g_start_time, g_time_limit
    now = datetime.datetime.now()
    if not g_start_time:
        g_start_time = now
    return True if (now - g_start_time) > datetime.timedelta(seconds=g_time_limit) else False

def lambda_time_left_seconds() -> int:
    global g_start_time, g_time_limit
    return int(g_time_limit - (datetime.datetime.now() - g_start_time).total_seconds())

def prtime():
    nw=datetime.datetime.now()
    return f"{nw.hour}:{nw.minute}:{nw.second}"

class llm_run_usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

