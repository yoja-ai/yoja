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
from utils import check_user, respond, check_cookie

def login(event, context):
    print('## ENVIRONMENT VARIABLES')
    print(os.environ)
    print('## EVENT')
    print(event)

    operation = event['requestContext']['http']['method']
    if (operation != 'POST'):
        print(f"Error: unsupported method: operation={operation}")
        return respond({"error_msg": str(ValueError('Unsupported method ' + str(operation)))}, status=400)
    rv, sstr = check_cookie(event, True)
    if rv == 0:
        return respond(None, res={'version': os.environ['LAMBDA_VERSION']})
    else:
        return respond({"status": sstr}, rv, None)
