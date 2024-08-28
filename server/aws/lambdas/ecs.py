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

def start_ecs_task(ecs_client, ecs_clustername, email):
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
                'subnets': ['subnet-0868138880e84997a', 'subnet-06016fd0cd6c75ee0', 'subnet-07eb8a31dcbf7d02d', 'subnet-0a0525d59ba82b072', 'subnet-0d9571415523c4745', 'subnet-0bd258d61a5d69d3d'],
                'securityGroups': ['sg-0d84a6e50fee1a231'],
                'assignPublicIp': 'ENABLED'
            }
        },
        overrides={
            'containerOverrides': [
                {
                    'name': 'periodic',
                    'environment': [
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
