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
  const res = await fetch(servicesConfig.envAPIEndpoint + API_URL, {
      method: "POST",
      headers,
      body: JSON.stringify({
        messages: messages,
        model: "gpt-3.5-turbo",
        stream: true,
        temperature: 1,
        top_p: 0.7
      }),
  });
  return res;
}
