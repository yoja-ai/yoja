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

[ "$aws_account_id" == "058264066930" ] && YOJADIST="True"

export AWS_PROFILE=${AWS_CREDS}
export AWS_DEFAULT_REGION=${AWS_REGN}

if [ -z "$YOJADIST" ]; then
  c=`cat <<EOF
import boto3
import sys

client = boto3.client('lambda')
result = client.list_layers(
    CompatibleRuntime='python3.11',
    MaxItems=50,
    CompatibleArchitecture='x86_64'
    )
if 'Layers' in result:
    for lyr in result['Layers']:
        if lyr['LayerName'] == 'yoja-webhook-layer':
            print(lyr['LatestMatchingVersion']['LayerVersionArn'])
            sys.exit(0)
sys.exit(255)
EOF`
  LAYER=`python3 -c "$c"`
  if [ $? != 0 ] ; then
    echo "Could not determine layer ARN"
    exit 255
  fi
  echo "Layer ARN is ${LAYER}"

  d=`cat <<EOF
import boto3
import sys

client = boto3.client('lambda')
result = client.list_functions()
if 'Functions' in result:
    for fnx in result['Functions']:
        if fnx['FunctionName'].startswith('yoja-entrypoint-'):
            print(fnx['FunctionArn'])
            sys.exit(0)
sys.exit(255)
EOF`
  FUNCTION_ARN=`python3 -c "$d"`
  if [ $? != 0 ] ; then
    echo "Could not determine Function ARN"
    exit 255
  fi
  echo "Function ARN is ${FUNCTION_ARN}"

  /bin/rm -f template.yaml
  #sed -e "s/BBBBBBBBBB/${LAYER}/" template.yaml.tmpl | sed -e "s/AAAAAAAAAA/${FUNCTION_ARN}/" > template.yaml

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
      ParameterKey=ServiceconfTable,ParameterValue=${serviceconf_table_name} \
      ParameterKey=YojaApiEntrypointFunctionArn,ParameterValue=${FUNCTION_ARN}
  retval=$?

  exit $retval
else
  # https://github.com/aws/aws-sam-cli/issues/6691: the image repo has to be private
  # https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/sam-cli-command-reference-sam-package.html
  # sam package --profile yojadist_root --image-repository  058264066930.dkr.ecr.us-east-1.amazonaws.com/yoja-img --template-file template.yaml --output-template-file packaged.yaml --s3-bucket scratch-bucket-yoja-dist
  sam package --profile ${AWS_CREDS} --template-file template.yaml --output-template-file packaged.yaml --s3-bucket $scratch_bucket --s3-prefix yoja-webhook  || { echo "Failed: error: $!"; exit 1; }

  # Publish your application to the AWS Serverless Application Repository using the AWS SAM CLI or the AWS Management Console. When publishing, you'll need to provide information like the application name, description, semantic version, and a link to the source code.
  # https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-template-publishing-applications.html : s3 policy to allow SAR to read S3 bucket
  # https://docs.aws.amazon.com/serverlessrepo/latest/devguide/serverlessrepo-how-to-publish.html: to setup s3 policies
  # sam publish --profile yojadist_root --template packaged.yaml --region us-east-1 --semantic-version 0.0.2
  sam publish --profile ${AWS_CREDS} --template packaged.yaml --region $AWS_REGN --semantic-version $sam_semantic_version || { echo "Failed: error: $!"; exit 1; }
fi
