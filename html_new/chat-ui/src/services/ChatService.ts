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
  headers["Authorization"] = "Bearer " + servicesConfig.API_KEY;

  const requestUrl = servicesConfig.envAPIEndpoint + API_URL;
  const requestBody = JSON.stringify({
    messages: messages,
    model: "gpt-3.5-turbo",
    stream: true,
    temperature: 1,
    top_p: 0.7
  });

  //initial fetch attempt
  let res = await fetch(requestUrl, {
    method: "POST",
    headers,
    body: requestBody,
  });

  // Retry once after 1s if the first attempt returns a 504 Gateway Timeout
  if (res.status === 504) {
    console.log("Received 504 Gateway Timeout. Retrying...");
    await new Promise(resolve => setTimeout(resolve, 1000)); 
    res = await fetch(requestUrl, {
      method: "POST",
      headers,
      body: requestBody,
    });
  } else if (res.status == 403) {
    console.log("Received 403 Unauthorized. Redirecting to /login.html ...");
    alert("Login expired");
    window.location.href = "/login.html";
  }

  return res;
};
