## What is Yoja?
![Yoja Gradient Banner](https://github.com/user-attachments/assets/08e8711c-7c2e-41e5-8845-595eb16585ee)

Yoja is an AI-driven assistant for personal files. Users can chat with Yoja about their Google Drive files and gain insights about their digital life and answers to their questions. Yoja is an MIT licensed open source project.

* [Website](https://yoja.ai)
* [LinkedIn](https://www.linkedin.com/company/yoja-ai)
<br>

## How can I set up Yoja?
The instructions below can help new users set up Yoja for the first time.
<br><br>

- Request a certificate for `chat.<domain>` and get it validated using DNS
<br>

- Configure Google Drive Client
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

- Configure Dropbox Client
  - Go to https://www.dropbox.com/developers/apps/info and choose the 'Create App' button
  - Choose scoped application
  - In the settings tab:
      - Click on the 'Enable Additional Users'
      - Redirect URIs: `https://chat.<yourdomain>/rest/entrypoint/oauth2cb`
      - Webhook URI: `https://chat.<yourdomain>/webhook/webhook_dropbox`
  - In the permissions tab:
      - Select file.content.read
      - Select openid scopes -> email
<br>

- Create an OpenAI Key
  - Login to the OpenAI REST API account and create a new secret key
<br>

- Create S3 buckets
  - a new s3 bucket for scratch
  - a bucket for storing html/javascript called yoja-html-<something_domain_based> e.g. yoja-html-staging9 for a service at `chat.staging9.example.com`
  - a bucket for storing indexes called yoja-index-<something_domain_based> e.g. yoja-index-staging9 for a service at `chat.staging9.example.com`
<br>

- Create DynamoDB tables
  - yoja-ServiceConf
      - Capacity is on-demand
      - Primary Key is configVersion of type N
  - yoja-users
      - Capacity is on-demand
      - Primary Key is email
      - Global Secondary Index:
          - dropbox_sub of type String is the Partition Key
<br>

- Deploy Main Lambda from `<yoja_root>/server/aws`
  - Create container using `scripts/create-container.sh`
  - Deploy using `scripts/build-and-deploy.sh`
<br>

- Deploy Webhook Lambda from `<yoja_root>/server/aws-webhook`
  - Create layer using `scripts/create-layer.sh`
  - Deploy using `scripts/build-and-deploy.sh`
<br>

- Create CloudFront distribution
    - First create a CloudFront distro using the HTML S3 bucket as the origin
        - Use OAC for the S3 bucket
        - Set up Alternate Domain Name (`chat.<domain>`) and cert using the cert that was created in the first step
        - Copy OAC policy and configure S3 bucket
    - Second, add an origin for the main Lambda using the Lambda Function displayed in the Lambda Console for the 'entrypoint' Lambda
        - Set response timeout to 60 seconds
        - Add behavior pointing /rest/* to this newly created Origin
    - Add origin for webhook_gdrive Lambda
        - Copy the Lambda's Function URL from the Lambda console and use it to create a new Origin
        - Add behavior pointing /webhook/webhook_gdrive to this newly created origin
    - Add origin for webhook_dropbox Lambda
        - Copy the Lambda's Function URL from the Lambda console and use it to create a new Origin
        - Add behavior pointing /webhook/webhook_dropbox to this newly created origin
<br>

- Add DNS entry pointing `chat.<domain>` to the above CloudFront distribution's URL
<br>

- Publish HTML from <yoja_root>/html_new
  - Use the script `html_new/scripts/copy-to-s3.sh`
