#!/bin/bash

if [ x"${13}" == "x" ] ; then
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
  echo "  [sam_semantic_version]"  
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
[ -n "${14}" ] && sam_semantic_version=${14}

echo "Using AWS Credentials Profile $AWS_CREDS"
echo "Using AWS Region $AWS_REGN"

set -x

aws --profile ${AWS_CREDS} --region ${AWS_REGN} s3 ls "s3://${scratch_bucket}" || { echo "Error: unable to access the specified s3 bucket ${scratch_bucket}.  Fix and try again"; exit 1; }

# setup the image URI
aws_account_id=`aws --profile ${AWS_CREDS} --region ${AWS_REGN} sts get-caller-identity --query "Account" --output text`
echo "AWS Account ID= $aws_account_id"

if [ "$aws_account_id" == "058264066930" ]; then
    [ -z "$sam_semantic_version" ] && { echo "sam_semantic_version argument must be specified"; exit 1; }
    YOJADIST="True"
fi

# if [ -z "$YOJADIST" ]; then
  image_repo="${aws_account_id}.dkr.ecr.$AWS_REGN.amazonaws.com/yoja-img"
  YOJA_LAMBDA_IMAGE_URI="${aws_account_id}.dkr.ecr.$AWS_REGN.amazonaws.com/yoja-img:$image_name"
# else
#   docker_reg="public.ecr.aws/r9b3c9q7"
#   YOJA_LAMBDA_IMAGE_URI="$docker_reg/yoja-img:$image_name"
# fi 

export YOJA_LAMBDA_IMAGE_URI
echo YOJA_LAMDBA_IMAGE_URI=$YOJA_LAMBDA_IMAGE_URI

export YOJA_LAMBDA_VERSION=${image_name}

# replace ImageUri attribute in template.yaml
cat << EOF > /tmp/replace_image_$$.py
import os
with open("template.yaml", "w") as outfile:
  with open("template.yaml.tmpl", "r") as infile:
    for line in infile:
        if line.find("IIIIIIIIII") >= 0:
          line = line.replace("IIIIIIIIII", f"!Sub {os.environ['YOJA_LAMBDA_IMAGE_URI']}")
        if line.find("VVVVVVVVVV") >= 0:
          line = line.replace("VVVVVVVVVV", f"{os.environ['YOJA_LAMBDA_VERSION']}")
        outfile.write(line)
EOF
python3 /tmp/replace_image_$$.py
rm /tmp/replace_image_$$.py

if [ -z "$YOJADIST" ]; then
  sam build --profile ${AWS_CREDS} --region ${AWS_REGN} || exit 1 
  sam deploy --profile ${AWS_CREDS} --region ${AWS_REGN} --template template.yaml --stack-name yoja \
    --s3-bucket ${scratch_bucket} --s3-prefix yoja \
    --region ${AWS_REGN} --capabilities CAPABILITY_IAM \
    --image-repository $aws_account_id.dkr.ecr.${AWS_REGN}.amazonaws.com/yoja \
    --parameter-overrides ParameterKey=OauthClientId,ParameterValue=${oauth_client_id} \
      ParameterKey=OauthClientSecret,ParameterValue=${oauth_client_secret} \
      ParameterKey=DropboxOauthClientId,ParameterValue=${dropbox_oauth_client_id} \
      ParameterKey=DropboxOauthClientSecret,ParameterValue=${dropbox_oauth_client_secret} \
      ParameterKey=OauthRedirectUri,ParameterValue=${oauth_redirect_uri} \
      ParameterKey=UsersTable,ParameterValue=${users_table_name} \
      ParameterKey=ServiceconfTable,ParameterValue=${serviceconf_table_name} \
      ParameterKey=CookieDomain,ParameterValue=${cookie_domain} \
      ParameterKey=OpenaiApiKey,ParameterValue=${openai_api_key}
  retval=$?
else
  # https://github.com/aws/aws-sam-cli/issues/6691: the image repo has to be private
  # https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/sam-cli-command-reference-sam-package.html
  # sam package --profile yojadist_root --image-repository  058264066930.dkr.ecr.us-east-1.amazonaws.com/yoja-img --template-file template.yaml --output-template-file packaged.yaml --s3-bucket scratch-bucket-yoja-dist
  sam package --profile ${AWS_CREDS} --image-repository  $image_repo --template-file template.yaml --output-template-file packaged.yaml --s3-bucket $scratch_bucket --s3-prefix yoja-api || { echo "Failed: error: $!"; exit 1; }

  # Publish your application to the AWS Serverless Application Repository using the AWS SAM CLI or the AWS Management Console. When publishing, you'll need to provide information like the application name, description, semantic version, and a link to the source code.
  # https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-template-publishing-applications.html : s3 policy to allow SAR to read S3 bucket
  # https://docs.aws.amazon.com/serverlessrepo/latest/devguide/serverlessrepo-how-to-publish.html: to setup s3 policies
  # sam publish --profile yojadist_root --template packaged.yaml --region us-east-1 --semantic-version 0.0.2
  sam publish --profile ${AWS_CREDS} --template packaged.yaml --region $AWS_REGN --semantic-version $sam_semantic_version || { echo "Failed: error: $!"; exit 1; }
fi

# [ -f template.yaml ] && rm template.yaml
# [ -f packaged.yaml ] && rm packaged.yaml

exit $retval
