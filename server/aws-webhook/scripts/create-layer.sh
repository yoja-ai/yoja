#!/bin/bash
# if [ x"$2" == "x" ] ; then
#     echo "Usage $0 aws_credentials_profile aws_region"
#     exit 255
# fi
# 
# echo Using AWS Credentials Profile "$1"
# echo Using AWS Region "$2"

sudo /bin/rm -f /tmp/yoja-webhook-layer.zip
sudo /bin/rm -rf ./layer-build
mkdir -p layer-build/python/lib/python3.11/site-packages
(cd layer-build; echo "cryptography" > requirements.txt; echo "google-api-python-client" >> requirements.txt; echo "google-auth-httplib2" >> requirements.txt; echo "google-auth-oauthlib" >> requirements.txt)
(cd layer-build; sudo docker run -v "$PWD":/var/task "public.ecr.aws/sam/build-python3.11:latest" /bin/sh -c "pip install -r requirements.txt -t python/lib/python3.11/site-packages/; exit")
sudo /bin/rm -rf layer-build/python/lib/python3.11/site-packages/googleapiclient/discovery_cache/documents
(cd layer-build; sudo zip -r /tmp/yoja-webhook-layer.zip .)
# aws --profile "$1" --region "$2" lambda publish-layer-version --layer-name yoja-webhook-layer \
#     --description "Yoja Webhook Layer" \
#     --license-info "MIT" \
#     --zip-file fileb:///tmp/yoja-webhook-layer.zip \
#     --compatible-runtimes python3.11 \
#     --compatible-architectures "x86_64"
