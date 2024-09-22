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
from utils import respond, check_cookie

def set_searchsubdir(event, context):
    rv = check_cookie(event, False)
    email = rv['google']
    if not email:
        print(f"set_searchsubdir: check_cookie did not return email. Sending 403")
        return respond({"status": "Unauthorized: please login to google auth"}, 403, None)
    operation = event['requestContext']['http']['method']
    if (operation != 'POST'):
        return respond({"error_msg": "Error. http operation {operation} not supported"}, status=400)
    if not 'body' in event:
        return respond({"error_msg": "Error. body not present"}, status=400)
    body = json.loads(event['body'])
    print(f"set_searchsubdir: body={body}")
    searchsubdir = body['searchsubdir']
    print(f"set_searchsubdir: searchsubdir={searchsubdir}")
    cookie = f"__Host-yoja-searchsubdir={searchsubdir}; Path=/; Secure; SameSite=Strict; Max-Age=604800"
    return {
        'statusCode': 200,
        'body': f"<html>{email} set search subdir</html>",
        'headers': {
            'Content-Type': 'text/html',
            'Cache-Control': 'no-cache, no-store, must-revalidate, private',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Set-Cookie': cookie
        }
    }
