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
          "https://www.googleapis.com/auth/userinfo.email"]

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
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Methods': '*',
            'Access-Control-Allow-Credentials': '*'
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
    client = boto3.client('dynamodb')
    for ind in range(2):
        try:
            resp = client.get_item(TableName=os.environ['SERVICECONF_TABLE'],
                                            Key={'configVersion': {'N': str(1)}})
            print(f"get_service_conf: resp={resp}")
            if 'Item' in resp:
                cached_service_conf = resp['Item']
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

def get_user_table_entry(email):
    try:
        client = boto3.client('dynamodb')
        response = client.get_item(TableName=os.environ['USERS_TABLE'], Key={'email': {'S': email}})
        return response['Item']
    except Exception as ex:
        print(f"Caught {ex} while getting info for {email} from users table")
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
        print(f"check_webhook_gdrive: resource_id {resource_id}, address={addr} for email {email}")
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
        print(f"check_webhook_gdrive: watch. hdrs={json.dumps(headers)}, postdata={json.dumps(postdata)}, params={json.dumps(params)}")
        resp = requests.post('https://www.googleapis.com/drive/v3/changes/watch', headers=headers, json=postdata, params=params)
        resp.raise_for_status()
        print(f"check_webhook_gdrive: post resp.text={resp.text}")
    except Exception as ex:
        if resp:
            print(f"check_webhook_gdrive: In changes.watch for user {email}, caught {ex}. Response={resp.content}")
        else:
            print(f"check_webhook_gdrive: In changes.watch for user {email}, caught {ex}")
        return False
    try:
        boto3.client('dynamodb').update_item(
                        TableName=os.environ['USERS_TABLE'],
                        Key={'email': {'S': email}},
                        UpdateExpression="SET gdrive_next_page_token = :tk",
                        ExpressionAttributeValues={':tk': {'S': start_page_token}}
                    )
    except Exception as ex:
        print(f"check_webhook_gdrive: Error. Caught {ex} while saving nextPageToken in yoja-users")
        return False
    return True

def check_webhook_gdrive(email, service_conf):
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
        return None

def check_cookie(event, refresh_access_token):
    service_conf = get_service_conf()
    rv = {'google': '', 'dropbox': ''}
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
                    if cn == 'yoja-user':
                        email = check_user(service_conf, cookie_val.strip(), refresh_access_token, 'google')
                        if email:
                            rv['google'] = email
                            check_webhook_gdrive(email, service_conf)
                    elif cn == 'yoja-dropbox-user':
                        dropbox_email = check_user(service_conf, cookie_val.strip(), refresh_access_token, 'dropbox')
                        if dropbox_email:
                            rv['dropbox'] = dropbox_email
    return rv

def update_users_table(email, crds):
    try:
        # https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/API_UpdateItem.html
        boto3.client('dynamodb').update_item(
                        TableName=os.environ['USERS_TABLE'],
                        Key={'email': {'S': email}},
                        UpdateExpression="SET refresh_token = :rt, access_token = :at, created = :ct",
                        ExpressionAttributeValues={':rt': {'S': crds['refresh_token']}, ':at': {'S': crds['token']}, ':ct': {'N': str(int(time.time()))}}
                    )
    except Exception as ex:
        print(f"Caught {ex} while updating users table")
        traceback.print_exc()

def refresh_user_google(item) -> Credentials:
    """ refresh the credentials of the user stored in 'item' if necessary.  Also updates the credentials in the yoja-users table, if refreshed """
    email = item['email']['S']
    refresh_token = item['refresh_token']['S']
    access_token = item['access_token']['S']
    id_token = item['id_token']['S']
    created = int(item['created']['N'])
    expires_in = int(item['expires_in']['N'])
    print(f"refresh_user_google: user_type=google, email={email}, access token created={datetime.datetime.fromtimestamp(created)}")
    token={"access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "scope": " ".join(SCOPES),
            "created": created,
            "expires_in": expires_in,
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
    update_users_table(email, json.loads(creds.to_json()))
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
        boto3.client('dynamodb').update_item(
                        TableName=os.environ['USERS_TABLE'],
                        Key={'email': {'S': email}},
                        UpdateExpression="SET dropbox_access_token = :at, dropbox_created = :ct, dropbox_expires_in = :exp",
                        ExpressionAttributeValues={':at': {'S': access_token}, ':ct': {'N': str(int(time.time()))}, ':exp':{'N': str(expires_in)} }
                    )
    except Exception as ex:
        print(f"Caught {ex} while saving dropbox_access_token, dropbox_refresh_token for {email}")
        return None
    return access_token
