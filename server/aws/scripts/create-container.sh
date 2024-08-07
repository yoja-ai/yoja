#!/bin/bash
set -e

AWS_REGN="us-east-1"
if [ x"$1" == "x" ] ; then
    AWS_CREDS=default
else
    AWS_CREDS="$1"
    if [ x"${2}" != "x" ] ; then
    #if [ x"${2}" != "x" ] && [ "$2" != "yojadist" ] ; then
        AWS_REGN="${2}"
    fi
fi

# [ x"${2}" != "x" ] && [ "$2" == "yojadist" ] && YOJADIST="True"
# [ x"${3}" != "x" ] && [ "$3" == "yojadist" ] && YOJADIST="True"

echo "Using AWS Credentials Profile $AWS_CREDS"
echo "Using AWS Region $AWS_REGN"

(cd lambdas; docker build -t yoja-img .)

aws_account_id=`aws --profile ${AWS_CREDS} --region ${AWS_REGN} sts get-caller-identity --query "Account" --output text`
echo "aws_account_id=$aws_account_id"

epoch=$(date +'%Y%m%d_%H%M%S')_$(git describe --tags --dirty --long)

# if [ -n "$YOJADIST" ]; then
#     docker_reg="public.ecr.aws/r9b3c9q7"
#     image="$docker_reg/yoja-img:$epoch"
# else
    docker_reg="${aws_account_id}.dkr.ecr.${AWS_REGN}.amazonaws.com"
    image="$docker_reg/yoja-img:$epoch"
# fi

# if [ -n "$YOJADIST" ]; then
#     aws --profile $AWS_CREDS   --region $AWS_REGN   ecr-public get-login-password  | docker login --username AWS --password-stdin "$docker_reg"
# else
    aws --profile ${AWS_CREDS} --region ${AWS_REGN} ecr get-login-password | docker login --username AWS --password-stdin "$docker_reg"
# fi

docker tag yoja-img:latest ${image}
docker push ${image}

echo "Image = yoja-img:$epoch"

exit 0
