import os
import sys
import io
import boto3

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
    print(f"start_ecs_task: Entered. clustername={ecs_clustername}, email={email}")
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
                        }
                    ]
                }
            ]
        }
    )
    print(f"start_ecs_task: called run_task resp={resp}")
    return
