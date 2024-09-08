#!/bin/bash
set -e

if [ x"$6" == "x" ] ; then
    echo "Usage: $0 bucket oauth_redirect_uri envApiEndpoint google_client_id dropbox_client_id aws_creds_profile [web_ui_semantic_version]"
    exit 255
fi

set -x

OAUTH_REDIRECT_URI="$2"
envAPIEndpoint="$3"
GOOGLE_CLIENT_ID="$4"
DROPBOX_CLIENT_ID="$5"
AWS_CREDS="$6"
[ -n "${7}" ] && web_ui_semantic_version=${7}

aws --profile ${AWS_CREDS} s3 ls s3://"$1"/ || {
    echo "Error access s3 bucket $1. Configure aws credentials and try again"
    exit 255
}

aws_account_id=`aws --profile ${AWS_CREDS} sts get-caller-identity --query "Account" --output text`
echo "AWS Account ID= $aws_account_id"

aws --profile ${AWS_CREDS} s3 rm --recursive s3://$1/html/

/bin/rm -f /tmp/serviceconfig.js
echo "var ServiceConfig = {" > /tmp/serviceconfig.js
echo "  OAUTH_REDIRECT_URI: \"${OAUTH_REDIRECT_URI}\"," >> /tmp/serviceconfig.js
echo "  envAPIEndpoint: \"${envAPIEndpoint}\"," >> /tmp/serviceconfig.js
echo "  envAPIKey: \"unused\"," >> /tmp/serviceconfig.js
echo "  GOOGLE_CLIENT_ID: \"${GOOGLE_CLIENT_ID}\"," >> /tmp/serviceconfig.js
echo "  DROPBOX_CLIENT_ID: \"${DROPBOX_CLIENT_ID}\"" >> /tmp/serviceconfig.js
echo "}" >> /tmp/serviceconfig.js

(cd /tmp; aws --profile ${AWS_CREDS} s3 cp serviceconfig.js s3://$1/html/serviceconfig.js)
/bin/rm -f /tmp/serviceconfig.js

FILES="avatar.jpg icon.png login.html login.js login.css Dropbox-sdk.min.js"

for i in $FILES
do
    echo Copying $i
    aws --profile ${AWS_CREDS} s3 cp $i s3://$1/html/$i
done

(cd chat-ui; rm -rf build)
(cd chat-ui; npm install --force)
(cd chat-ui; npm run build)
(cd chat-ui/build; aws --profile ${AWS_CREDS} s3 sync . s3://$1/html/ )
exit 0
