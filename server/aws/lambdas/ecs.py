import os
import sys
import io
import boto3
from utils import get_user_table_entry

def get_ecs_task_count(ecs_client, ecs_clustername):
    print(f"get_ecs_task_count: Entered. clustername={ecs_clustername}")
    resp = ecs_client.list_tasks(cluster=ecs_clustername, family='yoja-periodic', desiredStatus='RUNNING')
    if 'taskArns' in resp:
        for ta in resp['taskArns']:
            print(f"get_ecs_task_count: task ARN {ta}")
        rv = len(resp['taskArns'])
        print(f"get_ecs_task_count: returning {rv}")
        return rv
    else:
        return 0

def start_ecs_task(ecs_client, ecs_clustername, ecs_subnets, ecs_securitygroups, email):
    item = get_user_table_entry(email)
    takeover_lock_end_time = int(item['lock_end_time']['N'])
    print(f"start_ecs_task: Entered. clustername={ecs_clustername}, email={email}, takeover_lock_end_time={takeover_lock_end_time}")
    resp = ecs_client.run_task(
        capacityProviderStrategy = [
            {
                'capacityProvider': 'FARGATE_SPOT',
                'weight': 1,
                'base': 0
            }
        ],
        cluster=ecs_clustername,
        taskDefinition='yoja-periodic',
        count=1,
        enableExecuteCommand=True,
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': ecs_subnets.strip().split(','),
                'securityGroups': ecs_securitygroups.strip().split(','),
                'assignPublicIp': 'ENABLED'
            }
        },
        overrides={
            'containerOverrides': [
                {
                    'name': 'periodic',
                    'environment': [
                        {
                            'name': 'OAUTH_REDIRECT_URI',
                            'value': os.environ['OAUTH_REDIRECT_URI']
                        },
                        {
                            'name': 'LAMBDA_VERSION',
                            'value': os.environ['LAMBDA_VERSION']
                        },
                        {
                            'name': 'OAUTH_CLIENT_ID',
                            'value': os.environ['OAUTH_CLIENT_ID']
                        },
                        {
                            'name': 'OAUTH_CLIENT_SECRET',
                            'value': os.environ['OAUTH_CLIENT_SECRET']
                        },
                        {
                            'name': 'DROPBOX_OAUTH_CLIENT_ID',
                            'value': os.environ['DROPBOX_OAUTH_CLIENT_ID']
                        },
                        {
                            'name': 'DROPBOX_OAUTH_CLIENT_SECRET',
                            'value': os.environ['DROPBOX_OAUTH_CLIENT_SECRET']
                        },
                        {
                            'name': 'SERVICECONF_TABLE',
                            'value': 'yoja-ServiceConf'
                        },
                        {
                            'name': 'USERS_TABLE',
                            'value': 'yoja-users'
                        },
                        {
                            'name': 'YOJA_USER',
                            'value': email
                        },
                        {
                            'name': 'YOJA_TAKEOVER_LOCK_END_TIME',
                            'value': str(takeover_lock_end_time)
                        }
                    ]
                }
            ]
        }
    )
    print(f"start_ecs_task: called run_task resp={resp}")
    return
