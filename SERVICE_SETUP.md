# Setting up the service for the first time

## First, request a certification for chat.<domain> and get it validaded using DNS

## Dropbox Client

- Go to https://www.dropbox.com/developers/apps/info and choose the 'create app' button
- Choose scoped application
- In the settings tab:
    - Click on the 'Enable Additional Users'
    - Redirect URIs: https://chat.<yourdomain>/rest/entrypoint/oauth2cb
    - Webhook URI: https://chat.<yourdomain>/webhook/webhook_dropbox
- In the Permissions tab:
    - select file.content.read
    - select openid scopes -> email

## OpenAI Key

Login to the OpenAI REST API account and create a new secret key

## Create S3 buckets
    - a new s3 bucket for scratch
    - a bucket for storing html/javascript called yoja-html-<something_domain_based> e.g. yoja-html-staging9 for a service at chat.staging9.example.com
    - a bucket for storing indexes called yoja-index-<something_domain_based> e.g. yoja-index-staging9 for a service at chat.staging9.example.com

## DDB Tables

- yoja-ServiceConf
    - Capacity is on-demand
    - Primary Key is configVersion of type N
- yoja-users
    - Capacity is on-demand
    - Primary Key is email
    - Global Secondary Index:
        - dropbox_sub of type String is the Partition Key

## Deploy Main Lambda from <yoja_root>/server/aws
    - create-container.sh
    - build-and-deploy.sh

## Deply Webhook Lambda from <yoja_root>/server/aws-webhook
    - create layer using scripts/create-layer.sh
    - deploy using scripts/build-and-deploy.sh

## Create CloudFront
    - First create a CF distro using the html S3 bucket as the origin
        - Use OAC for the S3 bucket
        - setup Alternate Domain Name (chat.<domain>) and cert using the cert that was created in the first step
        - Copy OAC policy and configure S3 bucket
    - Second, add an origin for the main Lambda using the Lambda Function displayed in the Lambda Console for the 'entrypoint' Lambda
        - Set Response Timeout to 60 seconds
        - Add Behavior pointing /rest/* to this newly created Origin
    - Add origin for webhook_gdrive Lambda
        - Copy the Lambda's Function URL from the Lambda console and use it to create a new Origin
        - Add Behavior pointing /webhook/webhook_gdrive to this newly created origin
    - Add origin for webhook_dropbox Lambda
        - Copy the Lambda's Function URL from the Lambda console and use it to create a new Origin
        - Add Behavior pointing /webhook/webhook_dropbox to this newly created origin

## Add DNS entry pointing chat.<domain> to the above CloudFront distro's URL

## Publish html from <yoja_root>/html_new

- Use the script html_new/scripts/copy-to-s3.sh
