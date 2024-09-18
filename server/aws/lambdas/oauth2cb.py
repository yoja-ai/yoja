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
from urllib.parse import unquote, urlencode
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import boto3
from utils import parse_id_token, get_service_conf, encrypt_email, respond, get_user_table_entry, update_users_table

def oauth2cb(event, context):
    operation = event['requestContext']['http']['method']
    if (operation != 'GET'):
        print(f"Error: unsupported method: operation={operation}")
        return respond({"error_msg": str(ValueError('Unsupported method ' + str(operation)))}, status=400)
    if 'queryStringParameters' in event:
        qs=event['queryStringParameters']
        if 'state' in qs and qs['state'] == 'dropbox':
            return oauth2cb_dropbox(qs)
        else:
            return oauth2cb_google(qs)
    else:
        print(f"oauth2cb_google: qs missing")
        return respond({"error_msg": "required parameters missing"}, status=403)

def oauth2cb_google(qs):
    if 'code' not in qs or 'scope' not in qs or 'state' not in qs:
        print(f"oauth2cb_google: qs incomplete = {qs}")
        return respond({"error_msg": "one or more required parameters missing"}, status=403)
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

    # if the yoja-users row does not contain fullname, then try to get the name and profile picture URL
    fullname = None
    picture = None
    item = get_user_table_entry(email)
    if not item or 'fullname' not in item:
        try:
            params={'access_token': access_token}
            ui_url = f"https://www.googleapis.com/oauth2/v3/userinfo"
            resp = requests.get(ui_url, params=params)
            resp.raise_for_status()
            rj = json.loads(resp.text)
            print(f"oauth2cb_google: userinfo resp={rj}")
            if 'name' in rj:
                fullname = rj['name']
            else:
                given_name = ''
                family_name = ''
                if 'given_name' in rj:
                    given_name = rj['given_name']
                if 'family_name' in rj:
                    family_name = rj['family_name']
                fullname = f"{given_name} {family_name}".strip()
            if 'picture' in rj:
                picture = rj['picture']
        except Exception as ex:
            print(f"while getting fullname, post caught {ex}")
            return respond({"error_msg": f"Exception {ex} getting fullname"}, status=403)
    if not update_users_table(email, refresh_token, access_token, expires_in, id_token=id_token, fullname=fullname, picture=picture):
        return respond({"error_msg": f"Error while updating users table for {email}"}, status=403)

    try:
        service_conf = get_service_conf()
        print(f"service_conf={service_conf}")
        e_email = encrypt_email(email, service_conf)
        cookie = f"__Host-yoja-user={e_email}; Path=/; Secure; SameSite=Strict; Max-Age=604800"
        print(f"email={email}, cookie={cookie}")
        get_params={'google': email, 'dropbox': '', 'fullname': '', 'picture': ''}
        if fullname:
            get_params['fullname'] = fullname
        if picture:
            get_params['picture'] = picture
        encoded_get_params=urlencode(get_params)
        redir_location=f"/index.html?{encoded_get_params}"
        print(f"oauth2cb_google: Redirect URL is {redir_location}")
        return {
            'statusCode': 302,
            'body': f"<html>{email} logged in to software version {os.environ['LAMBDA_VERSION']}</html>",
            'headers': {
                'Content-Type': 'text/html',
                'Cache-Control': 'no-cache, no-store, must-revalidate, private',
                'Pragma': 'no-cache',
                'Expires': '0',
                'Set-Cookie': cookie,
                'Location': redir_location
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
                #'scope':"openid email profile"}
        resp = requests.post('https://www.dropbox.com/oauth2/token', data=postdata)
        resp.raise_for_status()
        print(f"get access token post resp.text={resp.text}")
        rj = json.loads(resp.text)
        refresh_token=rj['refresh_token']
        if 'id_token' in rj:
            id_token=rj['id_token']
            id_token_dct=parse_id_token(id_token)
            print(f"oauth2cb_dropbox: id_token={id_token_dct}")
            email=id_token_dct['email']
            sub=id_token_dct['sub']
            print(f"oauth2cb_dropbox: email={email}")
    except Exception as ex:
        print(f"while getting access token, post caught {ex}")
        return respond({"error_msg": f"Exception {ex} exchanging code for access_token"}, status=403)

    try:
        postdata={'client_id': os.environ['DROPBOX_OAUTH_CLIENT_ID'],
                'client_secret': os.environ['DROPBOX_OAUTH_CLIENT_SECRET'], 
                'refresh_token': rj['refresh_token'],
                'grant_type': 'refresh_token'}
                #'scope':"openid email profile"}
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
                        UpdateExpression="SET dropbox_refresh_token = :rt, dropbox_access_token = :at, dropbox_id_token = :idt, dropbox_created = :ct, dropbox_expires_in = :exp, dropbox_sub = :ds",
                        ExpressionAttributeValues={
                            ':rt': {'S': refresh_token},
                            ':at': {'S': access_token},
                            ':idt':{'S': id_token},
                            ':ct': {'N': str(int(time.time()))},
                            ':exp':{'N': str(expires_in)},
                            ':ds': {'S': sub}
                        }
                    )
    except Exception as ex:
        print(f"Caught {ex} while saving dropbox_access_token, dropbox_refresh_token for {email}")
        return respond({"error_msg": f"Exception {ex} while saving dropbox_access_token, dropbox_refresh_token for {email}"}, status=403)

    # XXX Need to get fullname and picture URL from dropbox and add as GET params to /index.html below
    try:
        service_conf = get_service_conf()
        print(f"service_conf={service_conf}")
        e_email = encrypt_email(email, service_conf)
        cookie = f"__Host-yoja-dropbox-user={e_email}; Path=/; Secure; SameSite=Strict; Max-Age=604800"
        print(f"email={email}, cookie={cookie}")
        get_params={'google': email, 'dropbox': email, 'fullname': '', 'picture': ''}
        if fullname:
            get_params['fullname'] = fullname
        if picture:
            get_params['picture'] = picture
        return {
            'statusCode': 302,
            'body': f"<html>{email} logged in to software version {os.environ['LAMBDA_VERSION']}</html>",
            'headers': {
                'Content-Type': 'text/html',
                'Cache-Control': 'no-cache, no-store, must-revalidate, private',
                'Pragma': 'no-cache',
                'Expires': '0',
                'Set-Cookie': cookie,
                'Location': "/index.html"
            }
        }
    except Exception as ex:
        print(f"Caught {ex} while creating and setting dropbox cookie for {email}")
        return respond({"error_msg": f"Caught {ex} while creating and setting cookie for {email}"}, status=403)
    return respond(None, res={'version': os.environ['LAMBDA_VERSION']})
