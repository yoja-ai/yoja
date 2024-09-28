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
from indexing_progress import get_indexing_progress
from searchsubdir import set_searchsubdir
from directory_browser import directory_browser

fnxmap = {
        '/rest/entrypoint/getversion': getversion,
        '/rest/entrypoint/oauth2cb': oauth2cb,
        '/rest/entrypoint/login': login,
        '/rest/entrypoint/periodic': periodic,
        '/rest/v1/chat/completions': chat_completions,
        '/rest/entrypoint/tableqa': table_qa_handler,
        '/rest/v1/send-email': send_email,
        '/rest/entrypoint/generate-pdf': generate_pdf,
        '/rest/entrypoint/get-indexing-progress': get_indexing_progress,
        '/rest/entrypoint/set-searchsubdir': set_searchsubdir,
        '/rest/entrypoint/directory-browser': directory_browser
}

def entrypoint(event, context):
    print('## ENVIRONMENT VARIABLES')
    print(os.environ)
    print('## EVENT')
    print(event)
    operation = event['requestContext']['http']['method']
    if (operation == 'OPTIONS'):
        return respond(None, res={})
    path = event['requestContext']['http']['path']
    if (path in fnxmap):
        print('entrypoint: forwarding ' + path + ' to ' + fnxmap[path].__name__)
        return fnxmap[path](event, context)
    print('entrypoint: ERROR! Cannot find function to map ' + path)
    return respond({"error_msg": "REST endpoint not found"}, status=404)
