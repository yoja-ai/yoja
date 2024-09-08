const SCOPES = 'https://www.googleapis.com/auth/drive.metadata.readonly https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile';

let gisInited = false;
let loginRestFinished = false;
let gdriveLoginSuccess = false;
let gdriveEmail = "";
let dropboxLoginSuccess = false;
let dropboxEmail = "";
let fullname = "";
let picture = "";

function parseJwt (token) {
  var base64Url = token.split('.')[1];
  var base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
  var jsonPayload = decodeURIComponent(window.atob(base64).split('').map(function(c) {
    return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
  }).join(''));
  return JSON.parse(jsonPayload);
}

function showWaitingFlyout() {
  const waitingFlyout = document.getElementById('waitingFlyout');
  waitingFlyout.style.display = 'block';
}

async function getLoginJson() {
  return fetch('/rest/entrypoint/login', {
    method: 'POST',
    body: "initial login",
    headers: {
      'Content-Type': 'application/json'
    }
  })
    .then((response)=>response.json())
    .then((responseJson)=>{return responseJson});
}
async function tryLogin() {
  const myJson = await this.getLoginJson();
  gdriveLoginSuccess = false;
  dropboxLoginSuccess = false;
  if (typeof myJson.google === 'undefined' || myJson.google.length == 0) {
  } else {
    gdriveLoginSuccess = true;
    gdriveEmail = myJson.google;
  }
  if (typeof myJson.dropbox === 'undefined' || myJson.dropbox.length == 0) {
  } else {
    dropboxLoginSuccess = true;
    dropboxEmail = myJson.dropbox;
  }
  if (typeof myJson.fullname === 'undefined' || myJson.fullname.length == 0) {
  } else {
    fullname = myJson.fullname;
  }
  if (typeof myJson.picture === 'undefined' || myJson.picture.length == 0) {
  } else {
      picture = myJson.picture;
  }
  console.log("tryLogin: gdriveLoginSuccess=" + gdriveLoginSuccess
      + ", dropboxLoginSuccess=" + dropboxLoginSuccess);
  loginRestFinished = true;
}

function handleCredentialResponse(response) {
  console.log("Encoded JWT ID token: " + response.credential);
  cred = parseJwt(response.credential);
  console.log("Parsed Encoded JWT ID token: " + JSON.stringify(cred));
  console.log("Email: " + cred['email']);
  console.log("family_name: " + cred['family_name']);
  console.log("given_name: " + cred['given_name']);
  console.log("picture: " + cred['picture']);
  console.log("name: " + cred['name']);
  codeClient = google.accounts.oauth2.initCodeClient({
    client_id: ServiceConfig.GOOGLE_CLIENT_ID,
    scope: SCOPES,
    ux_mode: 'redirect',
    redirect_uri: ServiceConfig.OAUTH_REDIRECT_URI,
    state: "xyzzy",
    include_granted_scopes: false,
    hint: cred['email'],
    prompt: "none"
  });
  codeClient.requestCode();
  gisInited = true;
}

async function loginToYoja() {
  showWaitingFlyout();
  tryLogin();
  while (!loginRestFinished) {
    // Simulate a short delay before checking again
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  // Hide the waiting flyout when the variable changes
  document.getElementById('waitingFlyout').style.display = 'none';

  if (gdriveLoginSuccess) {
    console.log("login of gdrive success!!!!!!!!!!!!. gdriveEmail=" + gdriveEmail);
    const params = "google=" + encodeURIComponent(gdriveEmail)
                      + "&fullname=" + encodeURIComponent(fullname) + "&picture=" + encodeURIComponent(picture);
    window.location.href = "index.html?" + params;
  } else {
    google.accounts.id.initialize({
      client_id: ServiceConfig.GOOGLE_CLIENT_ID,
      callback: handleCredentialResponse
    });
    google.accounts.id.renderButton(
      document.getElementById("buttonDiv"),
      { theme: "outline", size: "large" }  // customization attributes
    );
    google.accounts.id.prompt(); // also display the One Tap dialog
  }
}

loginToYoja();
