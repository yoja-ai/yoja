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

fnxmap = {
        '/rest/entrypoint/getversion': getversion,
        '/rest/entrypoint/oauth2cb': oauth2cb,
        '/rest/entrypoint/login': login,
        '/rest/entrypoint/periodic': periodic,
        '/rest/v1/chat/completions': chat_completions,
        '/rest/entrypoint/tableqa': table_qa_handler
}

def entrypoint(event, context):
    print('## ENVIRONMENT VARIABLES')
    print(os.environ)

    # {'version': '0', 'id': '48661d16-9b69-4270-8100-d2cd279d09d9', 'detail-type': 'Scheduled Event', 'source': 'aws.scheduler', 'account': '876038950809', 'time': '2024-04-15T11:59:23Z', 'region': 'us-east-1', 'resources': ['arn:aws:scheduler:us-east-1:876038950809:schedule/default/yoja_hourly'], 'detail': '{}'
    # {'version': '2.0', 'routeKey': '$default', 'rawPath': '/rest/v1/chat/completions', 'rawQueryString': '', 'cookies': ['yoja-user=gAAAAABmGUXxnuib369i2GQfBt4xlgmU0iSmUwxKT3GOpk51FImTA5Lp0oSt4_3wPeMrYAXvgwv6ajdXtw6wiotQ3JQVNh-l5waT0VFVzeIdSPzf16GqiQo='], 'headers': {'cloudfront-is-android-viewer': 'false', 'content-length': '128', 'referer': 'https://yoja.isstage7.com/chatindex.html', 'x-amzn-tls-version': 'TLSv1.2', 'cloudfront-viewer-country': 'IN', 'sec-fetch-site': 'same-origin', 'origin': 'https://yoja.isstage7.com', 'cloudfront-viewer-postal-code': '600001', 'cloudfront-viewer-tls': 'TLSv1.3:TLS_AES_128_GCM_SHA256:connectionReused', 'x-forwarded-port': '443', 'via': '2.0 20eddc312f5fafe3d85effa2fe22f9e6.cloudfront.net (CloudFront)', 'authorization': 'Bearer unused', 'x-amzn-tls-cipher-suite': 'ECDHE-RSA-AES128-GCM-SHA256', 'sec-ch-ua-mobile': '?0', 'cloudfront-viewer-country-name': 'India', 'cloudfront-viewer-asn': '9829', 'cloudfront-is-desktop-viewer': 'true', 'host': 'e25gfeddvhs4shmoutz6bcqkxe0kjobi.lambda-url.us-east-1.on.aws', 'content-type': 'application/json', 'cloudfront-viewer-city': 'Chennai', 'sec-fetch-mode': 'cors', 'cloudfront-viewer-http-version': '2.0', 'cookie': 'yoja-user=gAAAAABmGUXxnuib369i2GQfBt4xlgmU0iSmUwxKT3GOpk51FImTA5Lp0oSt4_3wPeMrYAXvgwv6ajdXtw6wiotQ3JQVNh-l5waT0VFVzeIdSPzf16GqiQo=', 'cloudfront-viewer-address': '117.193.190.227:63738', 'x-forwarded-proto': 'https', 'accept-language': 'en-US,en;q=0.9', 'cloudfront-is-ios-viewer': 'false', 'x-forwarded-for': '117.193.190.227', 'cloudfront-viewer-country-region': 'TN', 'accept': '*/*', 'cloudfront-viewer-time-zone': 'Asia/Kolkata', 'cloudfront-is-smarttv-viewer': 'false', 'sec-ch-ua': '"Microsoft Edge";v="123", "Not:A-Brand";v="8", "Chromium";v="123"', 'x-amzn-trace-id': 'Root=1-66194a4c-4000dba64e9384b46cb8fe22', 'cloudfront-viewer-longitude': '80.22090', 'cloudfront-is-tablet-viewer': 'false', 'sec-ch-ua-platform': '"Windows"', 'cloudfront-forwarded-proto': 'https', 'cloudfront-viewer-latitude': '12.89960', 'cloudfront-viewer-country-region-name': 'Tamil Nadu', 'accept-encoding': 'gzip, deflate, br, zstd', 'x-amz-cf-id': '8GoI_mkm17bvMBsE0QL94A2mb2V9RifntrxWHt-AS2xjiDVdR6hNwQ==', 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0', 'cloudfront-is-mobile-viewer': 'false', 'sec-fetch-dest': 'empty'}, 'requestContext': {'accountId': 'anonymous', 'apiId': 'e25gfeddvhs4shmoutz6bcqkxe0kjobi', 'domainName': 'e25gfeddvhs4shmoutz6bcqkxe0kjobi.lambda-url.us-east-1.on.aws', 'domainPrefix': 'e25gfeddvhs4shmoutz6bcqkxe0kjobi', 'http': {'method': 'POST', 'path': '/rest/v1/chat/completions', 'protocol': 'HTTP/1.1', 'sourceIp': '130.176.16.70', 'userAgent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0'}, 'requestId': 'd216d3d0-3983-473a-9268-c85e2b2af15b', 'routeKey': '$default', 'stage': '$default', 'time': '12/Apr/2024:14:50:52 +0000', 'timeEpoch': 1712933452680}, 'body': '{"messages":[{"role":"user","content":"what is a document?"}],"model":"gpt-3.5-turbo","stream":true,"temperature":1,"top_p":0.7}', 'isBase64Encoded': False}
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
    return respond(None, res={})
