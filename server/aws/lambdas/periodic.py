import io
import json
import os
import sys
import traceback
import tempfile
import uuid
import base64
import zlib
from urllib.parse import unquote
import faiss
import boto3
from utils import respond, get_service_conf
import os.path
import io
from index_utils import update_index_for_user, lock_user, unlock_user, invoke_periodic_lambda
import jsons
import dataclasses
from typing import List, Dict, Optional, Any
import time
import datetime

@dataclasses.dataclass
class PeriodicBody:
    username:Optional[str] = ''

########
#  use this payload in EventBridge Schedules to setup hourly runs: { "requestContext": { "http": { "method":"POST", "path": "/rest/entrypoint/periodic" } }, "body":"{\"username\":\"raj@yoja.ai\"}" }
#######
def periodic(event:dict, context) -> dict:
    # {'version': '0', 'id': '48661d16-9b69-4270-8100-d2cd279d09d9', 'detail-type': 'Scheduled Event', 'source': 'aws.scheduler', 'account': '876038950809', 'time': '2024-04-15T11:59:23Z', 'region': 'us-east-1', 'resources': ['arn:aws:scheduler:us-east-1:876038950809:schedule/default/yoja_hourly'], 'detail': '{}'
    # {'version': '2.0', 'routeKey': '$default', 'rawPath': '/rest/v1/chat/completions', 'rawQueryString': '', 'cookies': ['yoja-user=gAAAAABmGUXxnuib369i2GQfBt4xlgmU0iSmUwxKT3GOpk51FImTA5Lp0oSt4_3wPeMrYAXvgwv6ajdXtw6wiotQ3JQVNh-l5waT0VFVzeIdSPzf16GqiQo='], 'headers': {'cloudfront-is-android-viewer': 'false', 'content-length': '128', 'referer': 'https://yoja.isstage7.com/chatindex.html', 'x-amzn-tls-version': 'TLSv1.2', 'cloudfront-viewer-country': 'IN', 'sec-fetch-site': 'same-origin', 'origin': 'https://yoja.isstage7.com', 'cloudfront-viewer-postal-code': '600001', 'cloudfront-viewer-tls': 'TLSv1.3:TLS_AES_128_GCM_SHA256:connectionReused', 'x-forwarded-port': '443', 'via': '2.0 20eddc312f5fafe3d85effa2fe22f9e6.cloudfront.net (CloudFront)', 'authorization': 'Bearer unused', 'x-amzn-tls-cipher-suite': 'ECDHE-RSA-AES128-GCM-SHA256', 'sec-ch-ua-mobile': '?0', 'cloudfront-viewer-country-name': 'India', 'cloudfront-viewer-asn': '9829', 'cloudfront-is-desktop-viewer': 'true', 'host': 'e25gfeddvhs4shmoutz6bcqkxe0kjobi.lambda-url.us-east-1.on.aws', 'content-type': 'application/json', 'cloudfront-viewer-city': 'Chennai', 'sec-fetch-mode': 'cors', 'cloudfront-viewer-http-version': '2.0', 'cookie': 'yoja-user=gAAAAABmGUXxnuib369i2GQfBt4xlgmU0iSmUwxKT3GOpk51FImTA5Lp0oSt4_3wPeMrYAXvgwv6ajdXtw6wiotQ3JQVNh-l5waT0VFVzeIdSPzf16GqiQo=', 'cloudfront-viewer-address': '117.193.190.227:63738', 'x-forwarded-proto': 'https', 'accept-language': 'en-US,en;q=0.9', 'cloudfront-is-ios-viewer': 'false', 'x-forwarded-for': '117.193.190.227', 'cloudfront-viewer-country-region': 'TN', 'accept': '*/*', 'cloudfront-viewer-time-zone': 'Asia/Kolkata', 'cloudfront-is-smarttv-viewer': 'false', 'sec-ch-ua': '"Microsoft Edge";v="123", "Not:A-Brand";v="8", "Chromium";v="123"', 'x-amzn-trace-id': 'Root=1-66194a4c-4000dba64e9384b46cb8fe22', 'cloudfront-viewer-longitude': '80.22090', 'cloudfront-is-tablet-viewer': 'false', 'sec-ch-ua-platform': '"Windows"', 'cloudfront-forwarded-proto': 'https', 'cloudfront-viewer-latitude': '12.89960', 'cloudfront-viewer-country-region-name': 'Tamil Nadu', 'accept-encoding': 'gzip, deflate, br, zstd', 'x-amz-cf-id': '8GoI_mkm17bvMBsE0QL94A2mb2V9RifntrxWHt-AS2xjiDVdR6hNwQ==', 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0', 'cloudfront-is-mobile-viewer': 'false', 'sec-fetch-dest': 'empty'}, 'requestContext': {'accountId': 'anonymous', 'apiId': 'e25gfeddvhs4shmoutz6bcqkxe0kjobi', 'domainName': 'e25gfeddvhs4shmoutz6bcqkxe0kjobi.lambda-url.us-east-1.on.aws', 'domainPrefix': 'e25gfeddvhs4shmoutz6bcqkxe0kjobi', 'http': {'method': 'POST', 'path': '/rest/v1/chat/completions', 'protocol': 'HTTP/1.1', 'sourceIp': '130.176.16.70', 'userAgent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0'}, 'requestId': 'd216d3d0-3983-473a-9268-c85e2b2af15b', 'routeKey': '$default', 'stage': '$default', 'time': '12/Apr/2024:14:50:52 +0000', 'timeEpoch': 1712933452680}, 'body': '{"messages":[{"role":"user","content":"what is a document?"}],"model":"gpt-3.5-turbo","stream":true,"temperature":1,"top_p":0.7}', 'isBase64Encoded': False}

    body_str:str = None
    if event.get('requestContext'):
        # event/body
        body_str:str = event['body'] if event.get('body') else None
    else:
        # event/details/body
        details:dict = event['details'] if event.get('details') else None
        if details: body_str = details.get('body')

    post_body:PeriodicBody = jsons.loads(body_str, PeriodicBody) if event.get('body') else None

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

    try:
        s3client = boto3.client('s3')
        client = boto3.client('dynamodb')
        last_evaluated_key = None
        # ideally should use datetime.timezone.utc but timezone not used consistently in the code
        start_time:datetime.datetime = datetime.datetime.now()
        while True:
            if last_evaluated_key:
                resp = client.scan(TableName=os.environ['USERS_TABLE'], Select='ALL_ATTRIBUTES',
                    ExclusiveStartKey=last_evaluated_key)
            else:
                resp = client.scan(TableName=os.environ['USERS_TABLE'], Select='ALL_ATTRIBUTES')
            if 'Items' in resp:
                # {"expires_in": 3599.0, "id_token": "abc", "refresh_token": "xyz", "created": 1712156791.0, "email": "raj@yoja.ai", "access_token": "mno"}
                for item in resp['Items']:
                    # if username not specified in post or if specified and matches the item, then process this time.
                    if not post_body.username or post_body.username == item['email']['S']:
                        gdrive_next_page_token, status = lock_user(item, client)
                        if status:
                            print(f"periodic: before updating index. gdrive_next_page_token={gdrive_next_page_token}")
                            gdrive_next_page_token = update_index_for_user(item, s3client,
                                                                    bucket=bucket, prefix=prefix,
                                                                    start_time=start_time,
                                                                    gdrive_next_page_token=gdrive_next_page_token)
                            print(f"periodic: after updating index. gdrive_next_page_token={gdrive_next_page_token}")
                            unlock_user(item, client, gdrive_next_page_token)
                            if not gdrive_next_page_token:
                                invoke_periodic_lambda(context.invoked_function_arn, item['email']['S'])
                if 'LastEvaluatedKey' in resp:
                    last_evaluated_key = resp['LastEvaluatedKey']
                else:
                    break
            else:
                break
    except Exception as ex:
        print(f"Caught {ex} while scanning users table")
        traceback.print_exc(ex)
        return respond({"error_msg": f"Caught {ex} while scanning users table"}, status=403)
    return respond(None, res={'version': os.environ['LAMBDA_VERSION']})
