# Setting up the Yoja service for the first time
<br>

## 1. Request a certificate for `chat.<domain>` and get it validated using DNS
<br>

## 2. Configure Google Drive Client
- Go to https://console.cloud.google.com and choose 'APIs and Services'
- Choose 'Credentials'
- Click on 'Create Credentials'
- Credentials type is 'OAuth client ID'
- Application type is 'Web Application'
- Give it a name such as `oauth_for_staging6`
- Add authorized javascript origins - `https://chat.<domain>`
- Add authorized redirect URI - `https://chat.<domain>/rest/entrypoint/oauth2cb`
- Create and save the client ID and client secret
<br>

## 3. Configure Dropbox Client
- Go to https://www.dropbox.com/developers/apps/info and choose the 'Create App' button
- Choose scoped application
- In the settings tab:
    - Click on the 'Enable Additional Users'
    - Redirect URIs: `https://chat.<yourdomain>/rest/entrypoint/oauth2cb`
    - Webhook URI: `https://chat.<yourdomain>/webhook/webhook_dropbox`
- In the Permissions tab:
    - select file.content.read
    - select openid scopes -> email
<br>

## 4. Create an OpenAI Key
- Login to the OpenAI REST API account and create a new secret key
<br>

## 5. Create S3 buckets
- a new s3 bucket for scratch
- a bucket for storing html/javascript called yoja-html-<something_domain_based> e.g. yoja-html-staging9 for a service at `chat.staging9.example.com`
- a bucket for storing indexes called yoja-index-<something_domain_based> e.g. yoja-index-staging9 for a service at `chat.staging9.example.com`
<br>

## 6. Create DynamoDB tables
- yoja-ServiceConf
    - Capacity is on-demand
    - Primary Key is configVersion of type N
- yoja-users
    - Capacity is on-demand
    - Primary Key is email
    - Global Secondary Index:
        - dropbox_sub of type String is the Partition Key
<br>

## 7. Deploy Main Lambda from `<yoja_root>/server/aws`
- create container using `scripts/create-container.sh`
- deploy using `scripts/build-and-deploy.sh`
<br>

## 8. Deploy Webhook Lambda from `<yoja_root>/server/aws-webhook`
- create layer using `scripts/create-layer.sh`
- deploy using `scripts/build-and-deploy.sh`
<br>

## 9. Create CloudFront distribution
    - First create a CloudFront distro using the html S3 bucket as the origin
        - Use OAC for the S3 bucket
        - setup Alternate Domain Name (`chat.<domain>`) and cert using the cert that was created in the first step
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
<br>

## 10. Add DNS entry pointing `chat.<domain>` to the above CloudFront distro's URL
<br>

## 11. Publish HTML from <yoja_root>/html_new
- Use the script `html_new/scripts/copy-to-s3.sh`
