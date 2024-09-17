import json
import os
import sys

from getversion import getversion
from login import login
from oauth2cb import oauth2cb
from periodic import periodic
from utils import respond
from chat import chat_completions
from table_qa_lambda import table_qa_handler
from send_email import send_email
from generate_pdf import generate_pdf

fnxmap = {
        '/rest/entrypoint/getversion': getversion,
        '/rest/entrypoint/oauth2cb': oauth2cb,
        '/rest/entrypoint/login': login,
        '/rest/entrypoint/periodic': periodic,
        '/rest/v1/chat/completions': chat_completions,
        '/rest/entrypoint/tableqa': table_qa_handler,
        '/rest/v1/send-email': send_email,
        '/rest/entrypoint/generate-pdf': generate_pdf
}

def entrypoint(event, context):
    print('## ENVIRONMENT VARIABLES')
    print(os.environ)

    print('## EVENT')
    print(event)
    if 'headers' in event and 'origin' in event['headers'] and event['headers']['origin'] != f"https://chat.{os.environ['COOKIE_DOMAIN']}":
        print(f"entrypoint: origin {event['headers']['origin']} unacceptable!. Returning 401")
        return respond({"error_msg": "Origin not permitted"}, status=401)
    operation = event['requestContext']['http']['method']
    if (operation == 'OPTIONS'):
        return respond(None, res={})
    path = event['requestContext']['http']['path']
    if (path in fnxmap):
        print('entrypoint: forwarding ' + path + ' to ' + fnxmap[path].__name__)
        return fnxmap[path](event, context)
    print('entrypoint: ERROR! Cannot find function to map ' + path)
    return respond({"error_msg": "REST endpoint not found"}, status=404)
