# Installing Yoja Webhooks

## One Time Setup

### Configure the CloudFront Distribution

After running the development cycle described below at least one time, perform the following steps to configure the CloudFront Distribution

- In the AWS Lambda console, there will be two Functions such as yoja-webhook-webhookgdrive-7309ldgdfkgj and yoja-webhook-webhookdropbox-Vknjdvjsettz. Create an origin for each of these functions as follows:
    - Click on the Function in the Lambda console and copy the Function URL
    - Click on 'Create Origin' in the CloudFront Distro's origins tab and paste the Lambda Function URL in the Origin Domain text box
    - Choose 'HTTPS Only', No origin Path, No OAC and disable Origin Shield
- Create two Behaviors in the CloudFront distribution
    - Click on the 'Create Behavior' button and choose the following patterns:
        - Gdrive: /webhook/webhook_gdrive
        - Dropbox: /webhook/webhook_dropbox
    - Choose the respective Lambda Function URL for the Origin and Origin Group dropdown
    - The rest of the settings are: HTTPS only, GET/HEAD/OPTIONS/PUT/POST/PATCH/DELETE, No to Restrict Viewer Access, Recommended Cache Policy

## Development Cycle

### Create the layer for lambda

`
$ cd <yoja_src_root>/server/aws-webhook
$ ./scripts/create-layer.sh awsstaging4 us-east-1
`

### Publish the Webhook Lambdas

You need the following details in order to build and publish the Yoja Webhook Lambdas:

- Google Oauth Client ID. In the example run below it is **434639645934-oijrogj5905itjtljglkjlrei5ig95j5.apps.googleusercontent.com**
- Google Oauth Client Secret. In the example run below, it is **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**
- Dropbox Oauth Client ID. In the example below, **0940lrgjndgr94l**
- Dropbox Oauth Secret. In the example, **klsl494hoigkjn7**
- Users Table name, usually **yoja-users**
- Serviceconf table name, usually **yoja-ServiceConf**
- AWS Credentials Profile Name. The aws profile in ~/.aws/credentials is **test-aws-profile**
- AWS Region, e.g. **us-east-1**
- Scratch bucket name. In the example, it is **scratch-bucket-aaha-1**

Example run:

`
$ ./scripts/build-and-deploy.sh '434639645934-oijrogj5905itjtljglkjlrei5ig95j5.apps.googleusercontent.com' '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' '0940lrgjndgr94l' 'klsl494hoigkjn7' yoja-users yoja-ServiceConf test-aws-profile us-east-1 scratch-bucket-aaha-1
`
