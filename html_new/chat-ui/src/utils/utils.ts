import store from "../store/store";
import dayjs from 'dayjs';

export function removeCSSClass(ele: { className: string; }, cls: string) {
  const reg = new RegExp("(\\s|^)" + cls + "(\\s|$)");
  ele.className = ele.className.replace(reg, " ");
}

export function addCSSClass(ele: { classList: { add: (arg0: any) => void; }; }, cls: any) {
  ele.classList.add(cls);
}

export const toAbsoluteUrl = (pathname: string) => process.env.PUBLIC_URL + pathname;

export function setupAxios(axios: { interceptors: { request: { use: (arg0: (config: any) => Promise<any>, arg1: (err: any) => Promise<never>) => void; }; }; }, store: { getState: () => { auth: { idToken: any; }; }; }, dispatch: any ) {
  axios.interceptors.request.use(
    async (config: { headers: { Authorization: string; }; }) => {
      const {
        auth: { idToken }
      } = store.getState();
      if (idToken) {
        config.headers.Authorization = `Bearer ${idToken}`;
      }
      return config;
    },
    (    err: any) => Promise.reject(err)
  );
}

export function removeStorage(key: string) {
  try {
    localStorage.setItem(key, "");
    localStorage.setItem(key + "_expiresIn", "");
  } catch (e) {
    console.log(
      "removeStorage: Error removing key [" +
        key +
        "] from localStorage: " +
        JSON.stringify(e)
    );
    return false;
  }
  return true;
}


export function getStorage(key: any) {
  const now = Date.now(); //epoch time, lets deal only with integer
  // set expiration for storage
  let expiresIn: any = localStorage.getItem(key + "_expiresIn");
  if (expiresIn === undefined || expiresIn === null) {
    expiresIn = 0;
  }

  expiresIn = Math.abs(expiresIn);
  if (expiresIn < now) {
    // Expired
    removeStorage(key);
    return null;
  } else {
    try {
      const value = localStorage.getItem(key);
      return value;
    } catch (e) {
      console.log(
        "getStorage: Error reading key [" +
          key +
          "] from localStorage: " +
          JSON.stringify(e)
      );
      return null;
    }
  }
}

export function setStorage(key: string, value: string, expires: number | null | undefined) {
  if (expires === undefined || expires === null) {
    expires = 24 * 60 * 60; // default: seconds for 1 day
  }

  const now = Date.now(); //millisecs since epoch time, lets deal only with integer
  const schedule: any = now + expires * 1000;
  try {
    localStorage.setItem(key, value);
    localStorage.setItem(key + "_expiresIn", schedule);
  } catch (e) {
    console.log(
      "setStorage: Error setting key [" +
        key +
        "] in localStorage: " +
        JSON.stringify(e)
    );
    return false;
  }
  return true;
}

export const getUUID = () => {
  const randomPart = Math.random().toString(36).substring(2, 10);
  return new Date().getTime() + randomPart;
};

export const ListSeletOptions = (list: any[]) => {
  const options: { value: any; label: any; }[] = []
  list.forEach((element: any) => {
    options.push({value: element, label: element})
  });
  return options;
};

export const objectSeletOptions = (object: {}) => {
  const options: { value: string; label: string; }[] = []
  Object.keys(object).forEach(tag => {
    options.push({value: tag, label: tag});
  });
  return options;
};

export const convertTimeToDayJS = (configValues: { startTime: string | number | Date | dayjs.Dayjs | null | undefined; endTime: string | number | Date | dayjs.Dayjs | null | undefined; }) => {
  configValues.startTime = dayjs(configValues.startTime);
  configValues.endTime = dayjs(configValues.endTime);
  return configValues;
};

export const getDateAndHour = (date: string | number | Date) => {
  const formatDate = new Date(date);
  return formatDate;
};

export const updateToken = async(refreshToken: string | null) => {
  if(refreshToken !== null) {
    var postData = "{\n"
    postData += "    \"AuthParameters\" : {\n"
    postData += "        \"REFRESH_TOKEN\" : \"" + refreshToken + "\"\n"
    postData += "    },\n"
    postData += "    \"AuthFlow\" : \"REFRESH_TOKEN_AUTH\",\n"
    postData += "    \"ClientId\" : \"" + (window as any).CwsearchUiClientId + "\"\n"
    postData += "}\n"
    let region = (window as any).CwsearchUserPoolId.split('_')[0];
    var url = 'https://cognito-idp.'+ region +'.amazonaws.com:443/'
    var request = new XMLHttpRequest();
    
    request.open("POST", url, false);
    request.setRequestHeader("Content-Type", "application/x-amz-json-1.1");
    request.setRequestHeader("X-Amz-Target", "AWSCognitoIdentityProviderService.InitiateAuth");
    try {
      request.send(postData);
    } catch (error: any) {
      console.log("REFRESH_TOKEN: Redirecting to login since we caught: " + error.message);
      return false;
    }
    if (request.status === 200) {
      var resp = JSON.parse(request.responseText);
      let authResult = resp.AuthenticationResult;
      let date = new Date();
      date.setTime(date.getTime()+(57*60*1000));
      localStorage.setItem('tokenTime', date.getTime().toString());
      date.setTime(date.getTime()+(1000*60*1000));
      document.cookie = "mlflow-request-header-Authorization=" + authResult.IdToken + "; expires=" + date.toString();
      document.cookie = "Idtoken=" + authResult.IdToken + "; expires=" + date.toString();
      document.cookie = "aws-accessToken=" + authResult.AccessToken + "; expires=" + date.toString();
      const auth = {
        idToken: authResult.IdToken,
        refreshToken: refreshToken,
        accessToken: authResult.AccessToken,
        tokenTime: date.getTime()
      };                  
      store.dispatch({ type: "[Login] Action", payload: auth });
      return true;
    } else {
      return false;
    }
  }
};

export const checkAuthorization = (user: { userName: any; }) => ( user ? true : false );

export function getUserActivities() {
  const state = store.getState();
  const user = state.user;
  const allActivities = state.entities.searchActivitiesByUser;
  const userName = user.userName ? user.userName : null;
  return (allActivities.data[userName]?  allActivities.data[userName] : []);
}

export function getUserChatHistory() {
  const state = store.getState();
  const user = state.user;
  const allActivities = state.entities.chatActivitiesByUser;
  const userName = user.userName ? user.userName : null;
  return (allActivities.data[userName]?  allActivities.data[userName] : {});
}

export function isTokenExpired() {
  const state = store.getState();
  const { auth } = state;
  if(auth.idToken) {
    const expiry = (JSON.parse(atob(auth.idToken.split('.')[1]))).exp;
    if((Math.floor((new Date).getTime() / 1000)) >= expiry) {
      setTimeout(() => {
        store.dispatch({ type: "[Logout] Action"});
      }, 500);
    }
    return (Math.floor((new Date).getTime() / 1000)) >= expiry;
  } else {
    setTimeout(() => {
      store.dispatch({ type: "[Logout] Action"});
    }, 500);
    return true;
  }               
}
