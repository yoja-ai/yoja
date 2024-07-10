import io
import json
import os
import traceback
import tempfile
import time
import datetime
from urllib.parse import unquote
import boto3
from botocore.exceptions import ClientError
from cryptography.fernet import Fernet
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from typing import Union, Dict, List, Any, Tuple

SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly",
          "https://www.googleapis.com/auth/drive.readonly",
          "https://www.googleapis.com/auth/userinfo.email"]

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

def invoke_periodic_lambda(function_arn, email):
    bodydict={"username": email}
    lambda_client = boto3.client('lambda')
    run_params = {
        "requestContext": {
                    "http": {"method": "POST", "path": "/rest/entrypoint/periodic"}
        },
        "body": json.dumps(bodydict)
    }
    try:
        print(f"invoke_periodic_lambda: invoking function {function_arn} with run_params {json.dumps(run_params)}")
        lambda_client.invoke(FunctionName=function_arn,
                        InvocationType='Event',
                        Payload=json.dumps(run_params))
        return True
    except Exception as ex:
        print(f"Caught {ex} while invoking periodic run lambda")
        return False

def process_sync(event, context, email):
    print(f"webhook_gdrive.process_sync: Entered. email={email}")
    resource_id = event['headers']['x-goog-resource-id']
    channel_id = event['headers']['x-goog-channel-id']
    expires = event['headers']['x-goog-channel-expiration']
    print(f"webhook_gdrive.process_sync: email={email}, resource_id={resource_id}, channel_id={channel_id}")
    try:
        boto3.client('dynamodb').update_item(
                        TableName=os.environ['USERS_TABLE'],
                        Key={'email': {'S': email}},
                        UpdateExpression="SET gw_resource_id = :ri, gw_channel_id = :ci, gw_expires = :ge",
                        ExpressionAttributeValues={':ri': {'S': resource_id}, ':ci': {'S': channel_id}, ':ge': {'S': expires}}
                    )
        invoke_periodic_lambda(os.environ['YOJA_LAMBDA_ARN'], email)
        return {
            'statusCode': 204,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Methods': '*',
                'Access-Control-Allow-Credentials': '*'
            },
        }
    except Exception as ex:
        print(f"webhook_gdrive.process_sync: Caught {ex} while updating users table for {email}")
        traceback.print_exc()
        print("webhook_gdrive.process_sync: Error updating users table")
        return respond(None, res={})

def process_msg(event, context, email, lambda_end_time, state):
    print(f"process_msg: Entered. email={email}, state={state}, lambda_end_time={lambda_end_time}")
    if lambda_end_time:
        l_e_t = int(lambda_end_time)
        l_e_t_s = datetime.datetime.fromtimestamp(l_e_t).strftime('%Y-%m-%d %I:%M:%S')
        now = time.time()
        now_s = datetime.datetime.fromtimestamp(now).strftime('%Y-%m-%d %I:%M:%S')
        if l_e_t < now:
            print(f"process_msg: email={email}, state={state}, lambda_end_time={l_e_t_s} before now={now_s}. Invoking...")
            invoke_periodic_lambda(os.environ['YOJA_LAMBDA_ARN'], email)
        else:
            print(f"process_msg: email={email}, state={state}, lambda_end_time={l_e_t_s} after now={now_s}. Not invoking...")
    else:
        print(f"process_msg: lambda_end_time not present. Invoking yoja lambda...")
        invoke_periodic_lambda(os.environ['YOJA_LAMBDA_ARN'], email)
    return {
        'statusCode': 204,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Methods': '*',
            'Access-Control-Allow-Credentials': '*'
        },
    }

cached_service_conf = None
def get_service_conf():
    global cached_service_conf
    if cached_service_conf:
        return cached_service_conf
    client = boto3.client('dynamodb')
    try:
        resp = client.get_item(TableName=os.environ['SERVICECONF_TABLE'],
                                        Key={'configVersion': {'N': str(1)}})
        print(f"get_service_conf: resp={resp}")
        if 'Item' in resp:
            cached_service_conf = resp['Item']
            return cached_service_conf
        else:
            print(f"No item. serviceconf table entry for version 1 not found. Throwing exception")
            raise Exception("Cannot find service conf")
    except ClientError as err:
        print(f"get_service_conf: caught ClientError {err}")
        if err.response['Error']['Code'] == 'ResourceNotFoundException':
            print(f"serviceconf table entry for version 1 not found. Throwing exception")
            raise Exception("Cannot find service conf")
        else:
            print(f"get_service_conf: err.response.code={err.response['Error']['Code']}")
    except Exception as ex:
        print(f"get_service_conf: caught {ex}")
    raise Exception("Unknown error looking up service conf")

def get_user_table_entry(email):
    try:
        client = boto3.client('dynamodb')
        response = client.get_item(TableName=os.environ['USERS_TABLE'], Key={'email': {'S': email}})
        return response['Item']
    except Exception as ex:
        print(f"Caught {ex} while getting info for {email} from users table")
        return None

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
        if 'lambda_end_time' in  item:
            return email, item['lambda_end_time']['N']
        else:
            return email, None
    except Exception as ex:
        print(f"check_user: cookie={cookie_val}, refresh_access_token={refresh_access_token}, user_type={user_type}: caught {ex}")
        return None

def webhook_gdrive(event, context):
    print('## ENVIRONMENT VARIABLES')
    print(os.environ)
    print('## EVENT')
    print(event)
    service_conf = get_service_conf()
    operation = event['requestContext']['http']['method']
    if not 'headers' in event:
        print("webhook_gdrive: Error. No headers?")
        return respond(None, res={})

    if 'x-goog-channel-token' in event['headers']:
        token = event['headers']['x-goog-channel-token']
    else:
        print("webhook_gdrive: Error. x-goog-channel-token not present")
        return respond(None, res={})
    email, lambda_end_time = check_user(service_conf, token.strip(), True, 'google')
    if not email:
        print("webhook_gdrive: Error. Unable to process token")
        return respond(None, res={})

    if 'x-goog-resource-state' in event['headers']:
        resource_state = event['headers']['x-goog-resource-state']
    else:
        print("webhook_gdrive: Error. x-goog-resource-state not present")
        return respond(None, res={})

    if resource_state == 'sync':
        return process_sync(event, context, email)
    elif resource_state == 'add':
        return process_msg(event, context, email, lambda_end_time, 'add')
    elif resource_state == 'remove':
        return process_msg(event, context, email, lambda_end_time, 'remove')
    elif resource_state == 'update':
        return process_msg(event, context, email, lambda_end_time, 'update')
    elif resource_state == 'trash':
        return process_msg(event, context, email, lambda_end_time, 'trash')
    elif resource_state == 'untrash':
        return process_msg(event, context, email, lambda_end_time, 'untrash')
    elif resource_state == 'change':
        return process_msg(event, context, email, lambda_end_time, 'change')
    else:
        print(f"webhook_gdrive: Unknown resource state {resource_state}")
        return respond(None, res={})
