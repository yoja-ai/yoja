import React, { Fragment } from "react"
import { loginApi } from "../../services/ChatService";
import { BallTriangle } from "react-loader-spinner";

const servicesConfig = (window as any).ServiceConfig;
const { useState, useEffect, useRef } = React

const SettingsModel = () => {
  const [isShown, setIsShown] = useState(false)
  const [isAuthorized, setIsAuthorized] = React.useState(false);
  const [isLoading, setIsLoading] = React.useState(true);
  const [gdriveEmail, setGdriveEmail] = React.useState(false);
  const [dropboxEmail, setDropboxEmail] = React.useState(false);
  const [userEmail, setUserEmail] = React.useState('');
  const DROPBOX_CLIENT_ID = servicesConfig?.DROPBOX_CLIENT_ID;
  const [mainLambdasVersion, setMainLambdasVersion] = React.useState('1.1.1');
  const [webhookLambdasVersion, setWebhookLambdasVersion] = React.useState('1.1.1');
  const [uiVersion, setUiVersion] = React.useState('1.1.1');

  useEffect(() => {
    console.log('login api');
    loginApi().then((res: any) => {
      if(res) {
      setIsAuthorized(true);
      if (typeof res.google === 'undefined' || res.google.length == 0) {
      } else {
        setGdriveEmail(true);
        setUserEmail(res.google);
      }
      if (typeof res.dropbox === 'undefined' || res.dropbox.length == 0) {
      } else {
        setDropboxEmail(true);
        setUserEmail(res.dropbox);
      }
      if (typeof res.main_lambdas_sar_semantic_version === 'undefined' || res.main_lambdas_sar_semantic_version.length == 0) {
      } else {
        setMainLambdasVersion(res.main_lambdas_sar_semantic_version);
      }
      if (typeof res.webhook_lambdas_sar_semantic_version === 'undefined' || res.webhook_lambdas_sar_semantic_version.length == 0) {
      } else {
        setWebhookLambdasVersion(res.webhook_lambdas_sar_semantic_version);
      }
      if (typeof res.ui_semantic_version === 'undefined' || res.ui_semantic_version.length == 0) {
      } else {
        setUiVersion(res.ui_semantic_version);
      }
      } else {
        setIsAuthorized(false);
      }
    }).catch((e: any) => {
      setIsAuthorized(false);
    }).finally(()=> {
      setIsLoading(false);
    })
  }, []);

  const connectGDrive = () => {
    {
      const SCOPES = 'https://www.googleapis.com/auth/drive.metadata.readonly https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/userinfo.email';
      (window as any).google.accounts.oauth2.initCodeClient({
        client_id: servicesConfig.GOOGLE_CLIENT_ID,
        scope: SCOPES,
        ux_mode: 'redirect',
        redirect_uri: servicesConfig.OAUTH_REDIRECT_URI,
        state: "xyzzy",
        include_granted_scopes: false,
        prompt: "none"
      }).requestCode();
    }
  }
  
  const connectDropBox = () => {
    const GOOGLE_CLIENT_ID = servicesConfig.GOOGLE_CLIENT_ID;
    const DROPBOX_CLIENT_ID = servicesConfig.DROPBOX_CLIENT_ID;
    const DROPBOX_SCOPES = ['account_info.read', 'files.metadata.read', 'files.content.read', 'openid', 'email'];
    const dbx = new (window as any).Dropbox.Dropbox({ clientId: DROPBOX_CLIENT_ID  });
    let authUrl = dbx.auth.getAuthenticationUrl(servicesConfig.OAUTH_REDIRECT_URI,
                        "dropbox",
                        "code",
                        "offline",
                        DROPBOX_SCOPES,
                        "none",
                        false)
      .then((authUrl: any) => {
        window.location.href = authUrl;
      })
  }

  if(isLoading) {
    return(
      <div style={{height:'500px'}}>
        <div className="loadingSpiner">
          <BallTriangle
            height={100}
            width={100}
            radius={5}
            color="#f47e60"
            ariaLabel="ball-triangle-loading"
            wrapperStyle={{}}
            wrapperClass=""
            visible={true}
          />          
        </div>
      </div>
    );
  } else {
    return (
        <div className="settings-base-layout">
          <div className="settings-inner-layout">
            <span> Account </span>
            <div style={{display: 'flex', justifyContent:'space-between'}}>
              <span className="settings-subhedder"> Email  </span>
              <span style={{ color: '1C1B1F', fontSize: '14px', fontWeight: '700'}}> { userEmail } </span>
            </div>
            <div style={{display: 'flex', justifyContent:'space-between'}}>
              <span className="settings-subhedder"> Versions  </span>
              <span style={{ color: '1C1B1F', fontSize: '14px', fontWeight: '700'}}> Main={mainLambdasVersion},Webhook={webhookLambdasVersion},UI={uiVersion} </span>
            </div>
          </div>

          <div className="settings-inner-layout">
            <span> Connections </span>
            <div>
                <div style={{display: 'flex', justifyContent:'space-between', alignItems:'center'}}>
                  <div style={{width: '50px', display: 'flex', justifyContent:'center', alignItems:'center'}}>
                    <img style={{ height: '32px'}} src="gdrive.png"/>
                  </div>
                  <div className="settings-account-info">
                    <span className="settings-subhedder"> Google Drive  </span>
                    <span className="settings-body-text"> Allow Yoja to access your Google Docs, Sheets, and Slides files.  </span>
                  </div>
                  <button disabled={gdriveEmail} className="settings-model-button" onClick={connectGDrive}> <span style={{ color: '1C1B1F', fontSize: '14px', fontWeight: '700'}}> {gdriveEmail ? "connected": "Connect"} </span> </button>
                </div>
            </div>
            <div>
                <div style={{display: 'flex', justifyContent:'space-between', alignItems:'center'}}>
                  <div style={{width: '50px', display: 'flex', justifyContent:'center', alignItems:'center'}}>
                    <img style={{ height: '32px'}} src="dropbox.png"/>
                  </div>
                  <div className="settings-account-info">
                    <span className="settings-subhedder"> DropBox  </span>
                    <span className="settings-body-text"> Allow Yoja to access your DropBox files, Sheets, and Slides files.  </span>
                  </div>
                  <button disabled={dropboxEmail} className="settings-model-button" onClick={connectDropBox}> <span style={{ color: '1C1B1F', fontSize: '14px', fontWeight: '700'}}> {dropboxEmail ? "connected": "Connect"} </span> </button>
                </div>
            </div>
            <div>
                <div style={{display: 'flex', justifyContent:'space-between', alignItems:'center'}}>
                  <div style={{width: '60px', display: 'flex', justifyContent:'center', alignItems:'center'}}>
                    <img style={{ height: '22px'}} src="onedrive.png"/>
                  </div>
                  <div className="settings-account-info">
                    <span className="settings-subhedder"> Microsoft OneDrive  </span>
                    <span className="settings-body-text"> Allow Yoja to access your Microsoft Word, Excel, Powerpoint files.  </span>
                  </div>
                  <button disabled className="settings-model-button"> <span style={{ color: '1C1B1F', fontSize: '14px', fontWeight: '700'}}> Coming soon </span> </button>
                </div>
            </div>
           </div>
           
           <div className="settings-inner-layout">
            <span> Chat </span>
            {/* <div>
                <div style={{display: 'flex', justifyContent:'space-between'}}>
                <div className="settings-account-info">
                    <span className="settings-subhedder"> Archived chats  </span>
                    <span className="settings-body-text"> Allow Yoja to access your Microsoft Word, Excel, Powerpoint files.  </span>
                </div>
                <button className="settings-model-button"> <span style={{ color: '1C1B1F', fontSize: '14px', fontWeight: '700'}}> Manage </span> </button>
                </div>
            </div>
            <div>
                <div style={{display: 'flex', justifyContent:'space-between'}}>
                <div className="settings-account-info">
                    <span className="settings-subhedder"> Archive all chats </span>
                </div>
                <button className="settings-model-button"> <span style={{ color: '1C1B1F', fontSize: '14px', fontWeight: '700'}}> Archive all </span> </button>
                </div>
            </div> */}
            <div>
                <div style={{display: 'flex', justifyContent:'space-between'}}>
                <div className="settings-account-info">
                    <span className="settings-subhedder"> Delete all chats </span>
                </div>
                <button className="settings-model-button" onClick={()=> {
                  console.log('sdsd');
                  localStorage.setItem("current_chat", JSON.stringify([]));
                  localStorage.setItem("chat_history", JSON.stringify([]));
                  window.location.reload();
                }}> <span style={{ color: '#CB4D4D', fontSize: '14px', fontWeight: '700'}}> Delete all </span> </button>
                </div>
            </div>
           </div>
        </div>
      )
  }
}

export default SettingsModel;

