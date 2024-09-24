import axios from "axios";
import { Message } from "../type";

const servicesConfig = (window as any).ServiceConfig;

const getUserToken = async () => {
  //const user = auth.currentUser;
  //const token = user && (await user.getIdToken());
  return '';
};

export const loginApi = async () => await fetch(servicesConfig.envAPIEndpoint + '/entrypoint/login', {
  method: 'POST',
  body: "initial login",
  headers: {
    'Content-Type': 'application/json'
  }
}).then((response)=>response.json())
.then((responseJson)=>{return responseJson});

export const chatApi = async (messages: Message[]) => {
  let headers: any = { "Content-Type": "application/json" };
  const API_URL = "/v1/chat/completions";
  const requestUrl = servicesConfig.envAPIEndpoint + API_URL;
  const requestBody = JSON.stringify({
    messages: messages,
    model: "gpt-3.5-turbo",
    stream: true,
    temperature: 1,
    top_p: 0.7
  });

  var num504s = 0;
  var num503s = 0;
  while (true) {
    let res = await fetch(requestUrl, {
      method: "POST",
      headers,
      body: requestBody,
    });
    if (res.status === 504) {
      console.log("Received 504 Gateway Timeout. Retrying...");
      num504s++;
      if (num504s >= 2) {
        console.log("Received too many Gateway Timeouts. Aborting..")
        alert("Received too many Gateway Timeouts. Aborting..")
        return new Response();
      } else {
        // XXX Need to provide user feedback and the ability to cancel
        await new Promise(resolve => setTimeout(resolve, 1000)); 
        continue;
      }
    } else if (res.status === 503) {
      console.log("Received 503 Service Unavailable. Waiting and retrying...");
      num503s++;
      if (num503s >= 12) {
        console.log("Received too many Service Unavailable responses. Aborting..")
        alert("Received too many Service Unavailable responses. Aborting..")
        return new Response();
      } else {
        // XXX Need to provide user feedback and the ability to cancel
        await new Promise(resolve => setTimeout(resolve, 10000)); 
        continue;
      }
    } else if (res.status == 403) {
      console.log("Received 403 Unauthorized. Redirecting to /login.html ...");
      alert("Login expired");
      window.location.href = "/login.html";
    } else {
      return res;
    }
  }
};

export const searchSubdirApi = async (newSearchSubdir: string) => {
  let res = await fetch(servicesConfig.envAPIEndpoint + '/entrypoint/set-searchsubdir',
    {
      method: 'POST',
      body: JSON.stringify({searchsubdir: newSearchSubdir }),
      headers: {'Content-Type': 'application/json'}
    }
  );
  return res;
};
