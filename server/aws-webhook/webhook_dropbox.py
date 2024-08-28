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
from typing import Union, Dict, List, Any, Tuple
from hashlib import sha256
import hmac

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

def invoke_periodic_lambda(function_arn, dropbox_sub):
    bodydict={"dropbox_sub": dropbox_sub}
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
        print(f"Caught {ex} invoking periodic run lambda")
        return False

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

def test_invoke(dropbox_sub):
    item = get_user_table_entry_dropbox_sub(dropbox_sub)
    if not item:
        print(f"test_invoke: Hmm. user table entry not found for dropbox_sub {dropbox_sub}")
        return False
    email=item['email']['S']
    if 'lock_end_time' in  item:
        l_e_t = int(item['lock_end_time']['N'])
        l_e_t_s = datetime.datetime.fromtimestamp(l_e_t).strftime('%Y-%m-%d %I:%M:%S')
        now = time.time()
        now_s = datetime.datetime.fromtimestamp(now).strftime('%Y-%m-%d %I:%M:%S')
        if l_e_t < now:
            print(f"test_invoke: email={email}, lock_end_time={l_e_t_s} before now={now_s}. Invoking...")
            invoke_periodic_lambda(os.environ['YOJA_LAMBDA_ARN'], dropbox_sub)
        else:
            print(f"test_invoke: email={email}, lock_end_time={l_e_t_s} after now={now_s}. Not invoking...")
    else:
        print(f"test_invoke: lock_end_time not present. Invoking yoja lambda...")
        invoke_periodic_lambda(os.environ['YOJA_LAMBDA_ARN'], dropbox_sub)

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

def webhook_dropbox(event, context):
    print('## ENVIRONMENT VARIABLES')
    print(os.environ)
    print('## EVENT')
    print(event)
    service_conf = get_service_conf()
    operation = event['requestContext']['http']['method']
    if operation == "GET": # verification
        print("webhook_dropbox: Received verification")
        if not 'queryStringParameters' in event:
            print("webhook_dropbox: verification error. No params?")
            return respond(None, res={})
        qsp=event['queryStringParameters']
        print(f"webhook_dropbox: queryStringParameters={qsp}")
        if not 'challenge' in qsp:
            print("webhook_dropbox: verification error. No challenge?")
            return respond(None, res={})
        return {
            'statusCode': 200,
            'body': qsp['challenge'],
            'headers': {
                'Content-Type': 'text/plain',
                'X-Content-Type-Options': 'nosniff'
            }
        }
    elif operation == "POST": # notification request
        if not 'headers' in event:
            print("webhook_dropbox: Error. No headers?")
            return respond(None, res={})
        if not 'x-dropbox-signature' in event['headers']:
            print("webhook_dropbox: Error. No x-dropbox-signature?")
            return respond(None, res={})
        body = event['body']
        signature = event['headers']['x-dropbox-signature']
        if not hmac.compare_digest(signature, hmac.new(os.environ['DROPBOX_OAUTH_CLIENT_SECRET'].encode('utf-8'), body.encode('utf-8'), sha256).hexdigest()):
            print(f"webhook_dropbox: Signature Error. from_header={signature}, calc={hmac.new(os.environ['DROPBOX_OAUTH_CLIENT_SECRET'].encode('utf-8'), body.encode('utf-8'), sha256).hexdigest()}")
            return respond({"error_msg": f"Signature error"}, status=403)
        bodyj = json.loads(body)
        print(f"webhook_dropbox: bodyj={bodyj}")
        for account in bodyj['list_folder']['accounts']:
            print(f"need to invoke for account {account}")
            test_invoke(account)
        return respond(None, res={})
    else:
        print(f"webhook_dropbox: Error. op={operation} not GET or POST?")
        return respond(None, res={})
