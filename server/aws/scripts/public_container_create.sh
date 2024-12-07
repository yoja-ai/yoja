#!/bin/bash
(cd lambdas; docker build -t yoja-img .)
aws --profile yojadist ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/r9b3c9q7
docker tag yoja-img:latest public.ecr.aws/r9b3c9q7/yoja-img:latest
docker push public.ecr.aws/r9b3c9q7/yoja-img:latest
