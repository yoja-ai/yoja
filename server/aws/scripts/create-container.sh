#!/bin/bash
# This script should not print anything other than the 'Image = ' line
set -e

AWS_REGN="us-east-1"
if [ x"$1" == "x" ] ; then
    AWS_CREDS=default
else
    AWS_CREDS="$1"
    if [ x"${2}" != "x" ] ; then
        AWS_REGN="${2}"
    fi
fi

HERE=`pwd`
(cd ../../html_new; ./scripts/copy-to-local.sh ${HERE}/lambdas/www) >& /tmp/t.log

(cd lambdas; docker build -t yoja-img .) >& /dev/null

aws_account_id=`aws --profile ${AWS_CREDS} --region ${AWS_REGN} sts get-caller-identity --query "Account" --output text`

epoch=$(date +'%Y%m%d_%H%M%S')_$(git describe --tags --dirty --long)

docker_reg="${aws_account_id}.dkr.ecr.${AWS_REGN}.amazonaws.com"
image="$docker_reg/yoja-img:$epoch"

aws --profile ${AWS_CREDS} --region ${AWS_REGN} ecr get-login-password | docker login --username AWS --password-stdin "$docker_reg" >& /dev/null

docker tag yoja-img:latest ${image} >& /dev/null
docker push ${image} >& /dev/null

echo "Image = yoja-img:$epoch"
exit 0
