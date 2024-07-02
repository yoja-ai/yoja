import { useEffect } from 'react';
const servicesConfig = (window as any).ServiceConfig;

const Login = () => {
  useEffect(() => {
    const SCOPES = 'https://www.googleapis.com/auth/drive.metadata.readonly https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/userinfo.email';
    (window as any).google.accounts.oauth2.initCodeClient({
      client_id: servicesConfig.CLIENT_ID,
      scope: SCOPES,
      ux_mode: 'redirect',
      redirect_uri: servicesConfig.OAUTH_REDIRECT_URI,
      state: "xyzzy",
      include_granted_scopes: false,
      prompt: "none"
    }).requestCode();
  }, []);
  return <div></div>;
};

export default Login;
