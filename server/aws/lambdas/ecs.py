import os
import sys
import io
import boto3

def get_ecs_task_count(ecs_client, ecs_clustername):
    print(f"get_ecs_task_count: Entered. clustername={ecs_clustername}")
    resp = ecs_client.list_tasks(cluster=ecs_clustername, family='yoja-periodic', desiredStatus='RUNNING')
    if 'taskArns' in resp:
        return len(resp['taskArns'])
    else:
        return 0

def start_ecs_task(ecs_client, ecs_clustername, email):
    print(f"start_ecs_task: Entered. clustername={ecs_clustername}, email={email}")
    return
