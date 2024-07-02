#!/bin/bash

if [ x"${9}" == "x" ] ; then
  echo "Usage: $0"
  echo "  oauth_client_id"
  echo "  oauth_client_secret"
  echo "  dropbox_oauth_client_id"
  echo "  dropbox_oauth_client_secret"
  echo "  users_table_name"
  echo "  serviceconf_table_name"
  echo "  aws_credentials_profile"
  echo "  aws_region"
  echo "  scratch_bucket"
  exit 255
fi

oauth_client_id=$1
oauth_client_secret=$2
dropbox_oauth_client_id=$3
dropbox_oauth_client_secret=$4
users_table_name=$5
serviceconf_table_name=$6
AWS_CREDS="${7}"
AWS_REGN="${8}"
scratch_bucket=${9}

echo "Using AWS Credentials Profile $AWS_CREDS"
echo "Using AWS Region $AWS_REGN"
aws_account_id=`aws --profile ${AWS_CREDS} --region ${AWS_REGN} sts get-caller-identity --query "Account" --output text`
echo "AWS Account ID= $aws_account_id"

aws --profile ${AWS_CREDS} --region ${AWS_REGN} s3 ls "s3://${scratch_bucket}" || { echo "Error: unable to access the specified s3 bucket ${scratch_bucket}.  Fix and try again"; exit 1; }

sam build --profile ${AWS_CREDS} --region ${AWS_REGN} || exit 1 
sam deploy --profile ${AWS_CREDS} --region ${AWS_REGN} --template template.yaml --stack-name yoja-webhook \
  --s3-bucket ${scratch_bucket} --s3-prefix yoja-webhook \
  --region ${AWS_REGN} --capabilities CAPABILITY_IAM \
  --parameter-overrides ParameterKey=OauthClientId,ParameterValue=${oauth_client_id} \
    ParameterKey=OauthClientSecret,ParameterValue=${oauth_client_secret} \
    ParameterKey=DropboxOauthClientId,ParameterValue=${dropbox_oauth_client_id} \
    ParameterKey=DropboxOauthClientSecret,ParameterValue=${dropbox_oauth_client_secret} \
    ParameterKey=UsersTable,ParameterValue=${users_table_name} \
    ParameterKey=ServiceconfTable,ParameterValue=${serviceconf_table_name}
retval=$?

exit $retval
