import io
import json
import jsons
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
from sentence_transformers import SentenceTransformer
from utils import respond, get_service_conf, check_cookie
import boto3
import table_qa
import traceback_with_variables
from typing import Dict, List, Any, Optional, Tuple

def table_qa_handler(event:dict, context) -> Dict[str, Any]:
    try: 
        operation = event['requestContext']['http']['method']
        if (operation != 'POST'):
            return respond({"error_msg": "Error. http operation {operation} not supported"}, status=400)
        
        if not event.get('debug'): 
            try:
                service_conf = get_service_conf()
            except Exception as ex:
                print(f"Caught {ex} while getting service_conf")
                return respond({"error_msg": f"Caught {ex} while getting service_conf"}, status=403)

            if 'bucket' not in service_conf or 'prefix' not in service_conf:
                print(f"Error. bucket and prefix not specified in service conf")
                return respond({"error_msg": "Error. bucket and prefix not specified in service_conf"}, status=403)
            bucket = service_conf['bucket']['S']
            prefix = service_conf['prefix']['S'].strip().strip('/')
            print(f"Index Location: s3://{bucket}/{prefix}")

        if not 'body' in event:
            return respond({"error_msg": "Error. body not present"}, status=400)

        table_qa_req:table_qa.TableQARequest = jsons.loads(event['body'], table_qa.TableQARequest)
        opt_args:dict = {}
        if table_qa_req.temperature: opt_args['temperature'] = table_qa_req.temperature
        (answer, messages) = table_qa.iterate_prompt_exec_until_ans(table_qa_req.model, table_qa_req.table_delimited_str, table_qa_req.delimiter, None, table_qa_req.question, **opt_args)
        
        res_str:str = json.dumps({"answer":answer, "messages":messages})
        return {
            'statusCode': 200,
            'body': res_str,
            'headers': {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache, no-store, must-revalidate, private',
                'Pragma': 'no-cache',
                'Expires': '0'
            },
        }
    except Exception as e:
        # print_exc() return value signature is 'no return': to avoid vscode marking code as unreachable, use 'if'
        if len(__name__): traceback_with_variables.print_exc(e)
        return {
            'statusCode': 403,
            'body': json.dumps({ "exception":str(e), "traceback":traceback.format_exc()}),
            'headers': {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache, no-store, must-revalidate, private',
                'Pragma': 'no-cache',
                'Expires': '0'
            },
        }

if __name__ == '__main__':
    table_str = \
"""
Year|Division|League|Regular Season|Playoffs|Open Cup|Avg. Attendance
2001|2|USL A-League|4th, Western|Quarterfinals|Did not qualify|7,169
2002|2|USL A-League|2nd, Pacific|1st Round|Did not qualify|6,260
2003|2|USL A-League|3rd, Pacific|Did not qualify|Did not qualify|5,871
2004|2|USL A-League|1st, Western|Quarterfinals|4th Round|5,628
2005|2|USL First Division|5th|Quarterfinals|4th Round|6,028
2006|2|USL First Division|11th|Did not qualify|3rd Round|5,575
2007|2|USL First Division|2nd|Semifinals|2nd Round|6,851
2008|2|USL First Division|11th|Did not qualify|1st Round|8,567
2009|2|USL First Division|1st|Semifinals|3rd Round|9,734
2010|2|USSF D-2 Pro League|3rd, USL (3rd)|Quarterfinals|3rd Round|10,727
"""
    
    event = { 
             "requestContext": {
                 "http": {
                     "method": "POST"
                 }
             },
             "body":json.dumps( { "table_delimited_str":table_str, "delimiter":'|', "question": "Count the number of times the team played in each League?"} ),
             "debug": True
            }
    
    retval:dict = table_qa_handler(event, {})
    print(retval)
