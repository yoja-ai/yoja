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
from utils import parse_id_token, get_service_conf, encrypt_email, respond

def oauth2cb(event, context):
    print('## ENVIRONMENT VARIABLES')
    print(os.environ)
    print('## EVENT')
    print(event)

    operation = event['requestContext']['http']['method']
    if (operation != 'GET'):
        print(f"Error: unsupported method: operation={operation}")
        return respond({"error_msg": str(ValueError('Unsupported method ' + str(operation)))}, status=400)
    qs=event['queryStringParameters']
    if 'state' in qs and qs['state'] == 'dropbox':
        return oauth2cb_dropbox(qs)
    else:
        return oauth2cb_google(qs)

def oauth2cb_google(qs):
    print(f"code={qs['code']}, scope={qs['scope']}, state={qs['state']}")
    try:
        postdata={'client_id': os.environ['OAUTH_CLIENT_ID'],
                'client_secret': os.environ['OAUTH_CLIENT_SECRET'], 
                'code': qs['code'],
                'grant_type': 'authorization_code',
                'redirect_uri': os.environ['OAUTH_REDIRECT_URI']}
        resp = requests.post('https://oauth2.googleapis.com/token', data=postdata)
        resp.raise_for_status()
        print(f"get access token post resp.text={resp.text}")
        rj = json.loads(resp.text)
        refresh_token=rj['refresh_token']
    except Exception as ex:
        print(f"while getting access token, post caught {ex}")
        return respond({"error_msg": f"Exception {ex} exchanging code for access_token"}, status=403)

    try:
        postdata={'client_id': os.environ['OAUTH_CLIENT_ID'],
                'client_secret': os.environ['OAUTH_CLIENT_SECRET'], 
                'refresh_token': rj['refresh_token'],
                'grant_type': 'refresh_token'}
        resp = requests.post('https://oauth2.googleapis.com/token', data=postdata)
        resp.raise_for_status()
        print(f"refresh access token post resp.text={resp.text}")
        rj = json.loads(resp.text)
        created=int(time.time())
        expires_in = rj['expires_in']
        id_token=rj['id_token']
        id_token_dct=parse_id_token(id_token)
        email=id_token_dct['email']
        print(f"email={email}")
        access_token=rj['access_token']
    except Exception as ex:
        print(f"while refreshing access token, post caught {ex}")
        return respond({"error_msg": f"Exception {ex} refreshing access_token"}, status=403)
    try:
        boto3.client('dynamodb').update_item(
                        TableName=os.environ['USERS_TABLE'],
                        Key={'email': {'S': email}},
                        UpdateExpression="SET refresh_token = :rt, access_token = :at, id_token = :idt, created = :ct, expires_in = :exp",
                        ExpressionAttributeValues={':rt': {'S': refresh_token}, ':at': {'S': access_token}, ':idt':{'S': id_token}, ':ct': {'N': str(int(time.time()))}, ':exp':{'N': str(expires_in)} }
                    )
    except Exception as ex:
        print(f"Caught {ex} while saving access_token, refresh_token for {email}")
        return respond({"error_msg": f"Exception {ex} while saving access_token, refresh_token for {email}"}, status=403)

    try:
        service_conf = get_service_conf()
        print(f"service_conf={service_conf}")
        e_email = encrypt_email(email, service_conf)
        cookie = f"yoja-user={e_email}; domain={os.environ['COOKIE_DOMAIN']}; Path=/; Secure; Max-Age=604800"
        print(f"email={email}, cookie={cookie}")
        return {
            'statusCode': 302,
            'body': f"<html>{email} logged in to software version {os.environ['LAMBDA_VERSION']}</html>",
            'headers': {
                'Content-Type': 'text/html',
                'Set-Cookie': cookie,
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Methods': '*',
                'Access-Control-Allow-Credentials': '*',
                'Location': "/chatindex.html"
            }
        }
    except Exception as ex:
        print(f"Caught {ex} while creating and setting cookie for {email}")
        return respond({"error_msg": f"Caught {ex} while creating and setting cookie for {email}"}, status=403)
    return respond(None, res={'version': os.environ['LAMBDA_VERSION']})

def oauth2cb_dropbox(qs):
    print(f"qs={qs}")
    try:
        postdata={'client_id': os.environ['DROPBOX_OAUTH_CLIENT_ID'],
                'client_secret': os.environ['DROPBOX_OAUTH_CLIENT_SECRET'], 
                'code': qs['code'],
                'grant_type': 'authorization_code',
                'redirect_uri': os.environ['OAUTH_REDIRECT_URI']}
        resp = requests.post('https://www.dropbox.com/oauth2/token', data=postdata)
        resp.raise_for_status()
        print(f"get access token post resp.text={resp.text}")
        rj = json.loads(resp.text)
        refresh_token=rj['refresh_token']
        if 'id_token' in rj:
            id_token=rj['id_token']
            id_token_dct=parse_id_token(id_token)
            email=id_token_dct['email']
            print(f"email={email}")
    except Exception as ex:
        print(f"while getting access token, post caught {ex}")
        return respond({"error_msg": f"Exception {ex} exchanging code for access_token"}, status=403)

    try:
        postdata={'client_id': os.environ['DROPBOX_OAUTH_CLIENT_ID'],
                'client_secret': os.environ['DROPBOX_OAUTH_CLIENT_SECRET'], 
                'refresh_token': rj['refresh_token'],
                'grant_type': 'refresh_token'}
        resp = requests.post('https://www.dropbox.com/oauth2/token', data=postdata)
        resp.raise_for_status()
        print(f"refresh access token post resp.text={resp.text}")
        rj = json.loads(resp.text)
        created=int(time.time())
        expires_in = rj['expires_in']
        access_token=rj['access_token']
    except Exception as ex:
        print(f"while refreshing access token, post caught {ex}")
        return respond({"error_msg": f"Exception {ex} refreshing access_token"}, status=403)

    try:
        boto3.client('dynamodb').update_item(
                        TableName=os.environ['USERS_TABLE'],
                        Key={'email': {'S': email}},
                        UpdateExpression="SET dropbox_refresh_token = :rt, dropbox_access_token = :at, dropbox_id_token = :idt, dropbox_created = :ct, dropbox_expires_in = :exp",
                        ExpressionAttributeValues={':rt': {'S': refresh_token}, ':at': {'S': access_token}, ':idt':{'S': id_token}, ':ct': {'N': str(int(time.time()))}, ':exp':{'N': str(expires_in)} }
                    )
    except Exception as ex:
        print(f"Caught {ex} while saving dropbox_access_token, dropbox_refresh_token for {email}")
        return respond({"error_msg": f"Exception {ex} while saving dropbox_access_token, dropbox_refresh_token for {email}"}, status=403)

    try:
        service_conf = get_service_conf()
        print(f"service_conf={service_conf}")
        e_email = encrypt_email(email, service_conf)
        cookie = f"yoja-dropbox-user={e_email}; domain={os.environ['COOKIE_DOMAIN']}; Path=/; Secure; Max-Age=604800"
        print(f"email={email}, cookie={cookie}")
        return {
            'statusCode': 302,
            'body': f"<html>{email} logged in to software version {os.environ['LAMBDA_VERSION']}</html>",
            'headers': {
                'Content-Type': 'text/html',
                'Set-Cookie': cookie,
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Methods': '*',
                'Access-Control-Allow-Credentials': '*',
                'Location': "/chatindex.html"
            }
        }
    except Exception as ex:
        print(f"Caught {ex} while creating and setting dropbox cookie for {email}")
        return respond({"error_msg": f"Caught {ex} while creating and setting cookie for {email}"}, status=403)
    return respond(None, res={'version': os.environ['LAMBDA_VERSION']})
