#!/bin/bash
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
echo "Using AWS Credentials Profile $AWS_CREDS"
echo "Using AWS Region $AWS_REGN"

(cd lambdas; docker build -t yoja-img .)

aws_account_id=`aws --profile ${AWS_CREDS} --region ${AWS_REGN} sts get-caller-identity --query "Account" --output text`
echo "aws_account_id=$aws_account_id"
epoch=$(date +'%Y%m%d_%H%M%S')_$(git describe --tags --dirty --long)
image="${aws_account_id}.dkr.ecr.${AWS_REGN}.amazonaws.com/yoja-img:$epoch"
aws --profile ${AWS_CREDS} --region ${AWS_REGN} ecr get-login-password | docker login --username AWS --password-stdin ${aws_account_id}.dkr.ecr.${AWS_REGN}.amazonaws.com
docker tag yoja-img:latest ${image}
docker push ${image}
echo "Image = yoja-img:$epoch"

exit 0
