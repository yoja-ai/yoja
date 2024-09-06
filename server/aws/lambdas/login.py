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
from utils import respond, check_cookie

def login(event, context):
    operation = event['requestContext']['http']['method']
    if (operation != 'POST'):
        print(f"Error: unsupported method: operation={operation}")
        return respond({"error_msg": str(ValueError('Unsupported method ' + str(operation)))}, status=400)
    rv = check_cookie(event, True)
    email = rv['google']
    if not email:
        print(f"login: check_cookie did not return email. Sending 403")
        return respond({"status": "Unauthorized: please login using google"}, 403, None)
    rv['version'] = os.environ['LAMBDA_VERSION']
    rv['main_lambdas_sar_semantic_version'] =  os.environ['MAIN_LAMBDAS_SAR_SEMANTIC_VERSION']
    rv['webhook_lambdas_sar_semantic_version'] =  os.environ['WEBHOOK_LAMBDAS_SAR_SEMANTIC_VERSION']
    rv['ui_semantic_version'] =  os.environ['UI_SEMANTIC_VERSION']
    print(f"login: rv={rv}")
    return respond(None, res=rv)
