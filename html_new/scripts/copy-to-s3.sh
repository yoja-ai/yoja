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

if [ "$aws_account_id" == "058264066930" ]; then
    [ -z "$web_ui_semantic_version" ] && { echo "web_ui_semantic_version argument must be specified"; exit 1; }
    YOJADIST="True"
fi

if [ -z "$YOJADIST" ]; then
    aws --profile ${AWS_CREDS} s3 rm --recursive s3://$1/html/
else
    local_web_ui_dir="/tmp/yoja_web_ui_$(date +'%Y%m%d_%H%M%S' )"
    mkdir $local_web_ui_dir
fi

echo "var ServiceConfig = {" > /tmp/serviceconfig.js.$$
echo "  OAUTH_REDIRECT_URI: \"${OAUTH_REDIRECT_URI}\"," >> /tmp/serviceconfig.js.$$
echo "  envAPIEndpoint: \"${envAPIEndpoint}\"," >> /tmp/serviceconfig.js.$$
echo "  envAPIKey: \"unused\"," >> /tmp/serviceconfig.js.$$
echo "  GOOGLE_CLIENT_ID: \"${GOOGLE_CLIENT_ID}\"," >> /tmp/serviceconfig.js.$$
echo "  DROPBOX_CLIENT_ID: \"${DROPBOX_CLIENT_ID}\"" >> /tmp/serviceconfig.js.$$
echo "}" >> /tmp/serviceconfig.js.$$

if [ -z "$YOJADIST" ]; then
    aws --profile ${AWS_CREDS} s3 cp /tmp/serviceconfig.js.$$ s3://$1/html/serviceconfig.js
else
    cp -v /tmp/serviceconfig.js.$$ "$local_web_ui_dir/"
fi
/bin/rm -f /tmp/serviceconfig.js.$$

FILES="avatar.jpg chatindex.html icon.png login.html sw.js Dropbox-sdk.min.js"

for i in $FILES
do
    echo Copying $i
    if [ -z "$YOJADIST" ]; then
        aws --profile ${AWS_CREDS} s3 cp $i s3://$1/html/$i
    else
        cp -v $i "$local_web_ui_dir/"
    fi
done

(cd chat-ui; rm -rf build)
(cd chat-ui; npm install --force)
(cd chat-ui; npm run build)
if [ -z "$YOJADIST" ]; then
    (cd chat-ui/build; aws --profile ${AWS_CREDS} s3 sync . s3://$1/html/ )
else
    (cd chat-ui/build; cp -vR  ./* "$local_web_ui_dir/" )
    
    local_web_ui_zip="/tmp/yoja_web_ui_$(date +'%Y%m%d_%H%M%S').zip"
    (cd $local_web_ui_dir; zip -r $local_web_ui_zip *)
    # https://s3.amazonaws.com/yojadist/builds/0.0.1/web_ui.zip
    aws --profile ${AWS_CREDS} s3 cp $local_web_ui_zip s3://yojadist/builds/$web_ui_semantic_version/web_ui.zip
fi

exit 0
