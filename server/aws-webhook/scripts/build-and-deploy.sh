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
  echo "  [semantic_version]"
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
[ -n "${10}" ] && sam_semantic_version=${10}

echo "Using AWS Credentials Profile $AWS_CREDS"
echo "Using AWS Region $AWS_REGN"
aws_account_id=`aws --profile ${AWS_CREDS} --region ${AWS_REGN} sts get-caller-identity --query "Account" --output text`
echo "AWS Account ID= $aws_account_id"

export AWS_PROFILE=${AWS_CREDS}
export AWS_DEFAULT_REGION=${AWS_REGN}

  d=`cat <<EOF
import boto3
import sys

client = boto3.client('lambda')
result = client.list_functions()
if 'Functions' in result:
    for fnx in result['Functions']:
        if 'YojaApiService' in fnx['FunctionName']:
            print(fnx['FunctionArn'])
            sys.exit(0)
sys.exit(255)
EOF`
FUNCTION_ARN=`python3 -c "$d"`
if [ $? != 0 ] ; then
    echo "Could not determine Function ARN"
    exit 255
fi
echo "Main Lambda Function ARN is ${FUNCTION_ARN}"

# get the existing stack name
c=`cat <<EOF
import boto3
import sys

client = boto3.Session(profile_name='$AWS_CREDS', region_name='$AWS_REGN').client('cloudformation')
stacks = client.list_stacks()
ssum = stacks['StackSummaries']

for os in ssum: 
    if os['StackStatus'] == 'CREATE_COMPLETE' or os['StackStatus'] == 'UPDATE_COMPLETE' or os['StackStatus'] == 'UPDATE_ROLLBACK_COMPLETE':
      # CFT stack names have YojaApiService in it.  manual stack name was 'yoja'
      if 'YojaWebhookService' in os['StackName'] :
        print(os['StackName'])
        sys.exit(0)
sys.exit(255)
EOF`
STACK_NAME=`python3 -c "$c"`

aws --profile ${AWS_CREDS} --region ${AWS_REGN} s3 ls "s3://${scratch_bucket}" || { echo "Error: unable to access the specified s3 bucket ${scratch_bucket}.  Fix and try again"; exit 1; }

sam build --profile ${AWS_CREDS} --region ${AWS_REGN} || exit 1 
sam deploy --profile ${AWS_CREDS} --region ${AWS_REGN} --template template.yaml --stack-name ${STACK_NAME} \
    --s3-bucket ${scratch_bucket} --s3-prefix yoja-webhook \
    --region ${AWS_REGN} --capabilities CAPABILITY_IAM \
    --parameter-overrides ParameterKey=OauthClientId,ParameterValue=${oauth_client_id} \
      ParameterKey=OauthClientSecret,ParameterValue=${oauth_client_secret} \
      ParameterKey=DropboxOauthClientId,ParameterValue=${dropbox_oauth_client_id} \
      ParameterKey=DropboxOauthClientSecret,ParameterValue=${dropbox_oauth_client_secret} \
      ParameterKey=UsersTable,ParameterValue=${users_table_name} \
      ParameterKey=ServiceconfTable,ParameterValue=${serviceconf_table_name} \
      ParameterKey=YojaApiEntrypointFunctionArn,ParameterValue=${FUNCTION_ARN}
retval=$?

exit $retval
