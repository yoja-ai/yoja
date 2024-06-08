#!/bin/bash

if [ x"$13" == "x" ] ; then
  echo "Usage: $0"
  echo "  oauth_client_id"
  echo "  oauth_client_secret"
  echo "  dropbox_oauth_client_id"
  echo "  dropbox_oauth_client_secret"
  echo "  oauth_redirect_uri"
  echo "  users_table_name"
  echo "  serviceconf_table_name"
  echo "  cookie_domain"
  echo "  scratch_bucket"
  echo "  openai_api_key"
  echo "  aws_creds_profile"
  echo "  aws_region"
  echo "  image_name"
  exit 255
fi

oauth_client_id=$1
oauth_client_secret=$2
dropbox_oauth_client_id=$3
dropbox_oauth_client_secret=$4
oauth_redirect_uri=$5
users_table_name=$6
serviceconf_table_name=$7
cookie_domain=$8
scratch_bucket=$9
openai_api_key=${10}
AWS_CREDS="${11}"
AWS_REGN="${12}"
image_name=${13}

echo "Using AWS Credentials Profile $AWS_CREDS"
echo "Using AWS Region $AWS_REGN"

aws --profile ${AWS_CREDS} --region ${AWS_REGN} s3 ls "s3://${scratch_bucket}" || { echo "Error: unable to access the specified s3 bucket ${scratch_bucket}.  Fix and try again"; exit 1; }

# setup the image URI
aws_account_id=`aws --profile ${AWS_CREDS} --region ${AWS_REGN} sts get-caller-identity --query "Account" --output text`
CWSEARCH_LAMBDA_IMAGE_URI="${aws_account_id}.dkr.ecr."'${AWS::Region}'".amazonaws.com/yoja-img:$image_name"
export CWSEARCH_LAMBDA_IMAGE_URI
echo CWSEARCH_LAMDBA_IMAGE_URI=$CWSEARCH_LAMBDA_IMAGE_URI

# replace ImageUri attribute in template.yaml
cat << EOF > /tmp/replace_image_$$.py
import os
with open("template.yaml", "w") as outfile:
  with open("template.yaml.tmpl", "r") as infile:
    for line in infile:
        if line.find("IIIIIIIIII") >= 0:
          line = line.replace("IIIIIIIIII", f"!Sub {os.environ['CWSEARCH_LAMBDA_IMAGE_URI']}")
        outfile.write(line)
EOF
python3 /tmp/replace_image_$$.py
rm /tmp/replace_image_$$.py

echo "AWS Account ID= $AWS_ACCOUNT_ID"

sam build --profile ${AWS_CREDS} --region ${AWS_REGN}
sam deploy --profile ${AWS_CREDS} --region ${AWS_REGN} --template template.yaml --stack-name yoja \
  --s3-bucket ${scratch_bucket} --s3-prefix yoja \
  --region ${AWS_REGN} --capabilities CAPABILITY_IAM \
  --image-repository $aws_account_id.dkr.ecr.${AWS_REGN}.amazonaws.com/yoja \
  --parameter-overrides ParameterKey=LambdaVersion,ParameterValue=${image_name} \
    ParameterKey=OauthClientId,ParameterValue=${oauth_client_id} \
    ParameterKey=OauthClientSecret,ParameterValue=${oauth_client_secret} \
    ParameterKey=DropboxOauthClientId,ParameterValue=${dropbox_oauth_client_id} \
    ParameterKey=DropboxOauthClientSecret,ParameterValue=${dropbox_oauth_client_secret} \
    ParameterKey=OauthRedirectUri,ParameterValue=${oauth_redirect_uri} \
    ParameterKey=UsersTable,ParameterValue=${users_table_name} \
    ParameterKey=ServiceconfTable,ParameterValue=${serviceconf_table_name} \
    ParameterKey=CookieDomain,ParameterValue=${cookie_domain} \
    ParameterKey=OpenaiApiKey,ParameterValue=${openai_api_key}
retval=$?

[ -f template.yaml ] && rm template.yaml

exit $retval
