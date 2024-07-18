# Building and Installing the New UI

## Pre-requisites:

- Google Oauth Client ID
- Dropbox Oauth Client ID
- AWS Profile with the Access Key ID and Secret Access Key

## Run copy-to-s3.sh

`
$ ./scripts/copy-to-s3.sh aaha-html-awsstaging4 https://chat.awsstaging4.aaha.ai/rest/entrypoint/oauth2cb https://chat.awsstaging4.aaha.ai/rest 739054309853-dkdg84gskklfjgldjrgto953ojgeljig.apps.googleusercontent.com 3fd309dglksglla aahataging4
`

In the above example:
- **aaha-html-awsstaging4** is the S3 bucket where the html content is served from
- **https://chat.awsstaging4.aaha.ai/rest/entrypoint/oauth2cb** is the Oauth2 Callback URL
- **https://chat.awsstaging4.aaha.ai/rest** is the rest endpoint for the UI
- **739054309853-dkdg84gskklfjgldjrgto953ojgeljig.apps.googleusercontent.com** is the Google Oauth Client ID
- **3fd309dglksglla** is the Dropbox Oauth Client ID
- **aahataging4** is the credentials profile name as stored in ~/.aws/credentials
