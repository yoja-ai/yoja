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
from utils import respond, check_user, get_service_conf
import boto3
from botocore.exceptions import ClientError
import hashlib
from index_utils import invoke_periodic_lambda

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
        invoke_periodic_lambda(context.invoked_function_arn, email)
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
        return respond(None, res={'version': os.environ['LAMBDA_VERSION']})

def process_msg(event, context, email, state):
    print(f"webhook_gdrive.process_msg: Entered. email={email}, state={state}")
    invoke_periodic_lambda(context.invoked_function_arn, email)
    return {
        'statusCode': 204,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Methods': '*',
            'Access-Control-Allow-Credentials': '*'
        },
    }

def webhook_gdrive(event, context):
    service_conf = get_service_conf()
    operation = event['requestContext']['http']['method']
    if not 'headers' in event:
        print("webhook_gdrive: Error. No headers?")
        return respond(None, res={'version': os.environ['LAMBDA_VERSION']})

    if 'x-goog-channel-token' in event['headers']:
        token = event['headers']['x-goog-channel-token']
    else:
        print("webhook_gdrive: Error. x-goog-channel-token not present")
        return respond(None, res={'version': os.environ['LAMBDA_VERSION']})
    email = check_user(service_conf, token.strip(), True, 'google')
    if not email:
        print("webhook_gdrive: Error. Unable to process token")
        return respond(None, res={'version': os.environ['LAMBDA_VERSION']})

    if 'x-goog-resource-state' in event['headers']:
        resource_state = event['headers']['x-goog-resource-state']
    else:
        print("webhook_gdrive: Error. x-goog-resource-state not present")
        return respond(None, res={'version': os.environ['LAMBDA_VERSION']})

    if resource_state == 'sync':
        return process_sync(event, context, email)
    elif resource_state == 'add':
        return process_msg(event, context, email, 'add')
    elif resource_state == 'remove':
        return process_msg(event, context, email, 'remove')
    elif resource_state == 'update':
        return process_msg(event, context, email, 'update')
    elif resource_state == 'trash':
        return process_msg(event, context, email, 'trash')
    elif resource_state == 'untrash':
        return process_msg(event, context, email, 'untrash')
    elif resource_state == 'change':
        return process_msg(event, context, email, 'change')
    else:
        print(f"webhook_gdrive: Unknown resource state {resource_state}")
        return respond(None, res={'version': os.environ['LAMBDA_VERSION']})
