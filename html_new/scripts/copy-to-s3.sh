#!/bin/bash
set -e

if [ x"$6" == "x" ] ; then
    echo "Usage: $0 bucket oauth_redirect_uri envApiEndpoint google_client_id dropbox_client_id aws_creds_profile [nonce]"
    exit 255
fi

set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

OAUTH_REDIRECT_URI="$2"
envAPIEndpoint="$3"
GOOGLE_CLIENT_ID="$4"
DROPBOX_CLIENT_ID="$5"
AWS_CREDS="$6"
if [ x"$7" == "x" ] ; then
  echo "Nonce not supplied"
else
  nonce="$7"
  echo "Using nonce ${nonce}"
fi

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

FILES="avatar.jpg icon.png login.html login.js login.css error.html Dropbox-sdk.min.js"

for i in $FILES
do
    echo Copying $i
    aws --profile ${AWS_CREDS} s3 cp $i s3://$1/html/$i
done

(cd chat-ui; rm -rf build)
(cd chat-ui; rm -rf node_modules)
(cd chat-ui; npm install --force)

if [ x"${nonce}" == "x" ] ; then
  echo "Nonce not supplied. Not ejecting and patching source code"
else
  echo "Nonce supplied. Patching source code so that nonce is added to dynamic css"
  (cd chat-ui; npm install --force)
  (cd chat-ui; npm run eject)
  (sed -e "s/WEBPACK_NONCE/${nonce}/" ${SCRIPT_DIR}/webpack.config.js.patch > /tmp/webpack.config.js.patch)
  (cd chat-ui/config; patch < /tmp/webpack.config.js.patch )
  (sed -e "s/WEBPACK_NONCE/${nonce}/" ${SCRIPT_DIR}/dynamiccss.patch > /tmp/dynamiccss.patch)
  (cd chat-ui/node_modules/rc-util/es/Dom/; patch < /tmp/dynamiccss.patch )
fi

(cd chat-ui/; npm run build)
(cd chat-ui/build; aws --profile ${AWS_CREDS} s3 sync . s3://$1/html/ )

if [ x"${nonce}" == "x" ] ; then
  echo "Nonce not supplied. Not resetting git"
else
  git reset --hard
  /bin/rm -rf chat-ui/config
fi

exit 0
