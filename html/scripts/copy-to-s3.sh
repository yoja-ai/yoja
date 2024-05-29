#!/bin/bash
set -e

if [ x"$7" == "x" ] ; then
    echo "Usage: $0 bucket api_key app_id oauth_redirect_uri envApiEndpoint client_id aws_creds_profile"
    exit 255
fi

AWS_CREDS="$7"

aws --profile ${AWS_CREDS} s3 ls s3://"$1"/ || {
    echo "Error access s3 bucket $1. Configure aws credentials and try again"
    exit 255
}

aws --profile ${AWS_CREDS} s3 rm --recursive s3://$1/html/

API_KEY="$2"
APP_ID="$3"
OAUTH_REDIRECT_URI="$4"
envAPIEndpoint="$5"
CLIENT_ID="$6"

echo "var ServiceConfig = {" > /tmp/serviceconfig.js.$$
echo "  API_KEY: \"${API_KEY}\"," >> /tmp/serviceconfig.js.$$
echo "  APP_ID: \"${APP_ID}\"," >> /tmp/serviceconfig.js.$$
echo "  OAUTH_REDIRECT_URI: \"${OAUTH_REDIRECT_URI}\"," >> /tmp/serviceconfig.js.$$
echo "  envAPIEndpoint: \"${envAPIEndpoint}\"," >> /tmp/serviceconfig.js.$$
echo "  envAPIKey: \"unused\"," >> /tmp/serviceconfig.js.$$
echo "  CLIENT_ID: \"${CLIENT_ID}\"" >> /tmp/serviceconfig.js.$$
echo "}" >> /tmp/serviceconfig.js.$$

aws --profile ${AWS_CREDS} s3 cp /tmp/serviceconfig.js.$$ s3://$1/html/serviceconfig.js
/bin/rm -f /tmp/serviceconfig.js.$$

FILES="avatar.jpg chatindex.html icon.png login.html sw.js"

for i in $FILES
do
    echo Copying $i
    aws --profile ${AWS_CREDS} s3 cp $i s3://$1/html/$i
done

exit 0
