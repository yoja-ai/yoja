import { getUUID } from "./reducers/utils";
import { Service } from "./services";
import store from "./store";

export const actions = {
  loginApi: "LOGIN_API",
  getLogGroupNames: "GET_LOG_GROUP_NAMES_API",
  getLogGroupTags: "GET_LOG_GROUP_TAGS_API",
  setLogTime: "SET_LOG_TIME_API",
  autocomplete: "AUTO_COMPLETE_API",
  search: "SEARCH_API",
  chat: "CHAT_API",
  storeChat: "STORE_CHAT_INFO",
};


export const customerInfoApi = (id = getUUID()) => {
    return (dispatch: any) => {
      const serveResponse = dispatch({
        type: actions.loginApi,
        payload: Service.getCustomerInfo({ Username:'', customerId:'', productCode:''}),
        meta: { id },
      });
      return serveResponse;
    };
  };

export const loginApi = (id = getUUID()) => {
  return (dispatch: any) => {
    const serveResponse = dispatch({
      type: actions.loginApi,
      payload: Service.login('initial login'),
      meta: { id },
    });
    return serveResponse;
  };
};


export const getLogGroupNkamesApi = (id = getUUID()) => {
  const payload = getIndexTime();
  payload['op'] = "loggroups";
  return {
    type: actions.getLogGroupNames,
    payload: Service.complete(payload),
    meta: {id},
  };
};













const getIndexTime = () => {
  const { configuration } = store.getState();
  let currentDate = new Date();
  let year = currentDate.getFullYear();
  let month = currentDate.getMonth();
  let day = currentDate.getDate();
  let hours = currentDate.getHours();

  let roundedDownEnd = new Date(year, month, day, hours - 1, 0, 0, 0 );
  let roundedDownStart = new Date(year, month, day - 1, hours - 1, 0, 0, 0 );

  const payload:any = {};
  if(!configuration.startTime && !configuration.endTime ) {
    payload['start_time'] = roundedDownStart.valueOf();
    payload['end_time'] = roundedDownEnd.valueOf();
  } else {
    if(typeof configuration.startTime === "string") {
      payload['start_time'] = new Date(configuration.startTime).valueOf();
      payload['end_time'] = new Date(configuration.endTime).valueOf();
    } else {
      payload['start_time'] = configuration.startTime.valueOf();
      payload['end_time'] = configuration.endTime.valueOf();
    }
    
  }
  return payload;
}

const getCommonPayload = () => {
  const { configuration } = store.getState();
  const payload = getIndexTime();
  if(configuration.filterType === "filter_by_tag") {
    if(configuration.tagKey && configuration.tagValue) {
      const tag = configuration.tagKey + '=' + configuration.tagValue;
      payload['tag'] =  tag;
    }
  } else if(configuration.filterType === "filter_by_log_group") {
    if(configuration.logGroups) {
      payload['loggroups'] = configuration.logGroups;
    }
  }
  return payload;
}



export const getLogGroupNamesApi = (id = getUUID()) => {
  const payload = getIndexTime();
  payload['op'] = "loggroups";
  return {
    type: actions.getLogGroupNames,
    payload: Service.complete(payload),
    meta: {id},
  };
};

export const getLogGroupTagsApi = (id = getUUID()) => {
  const payload = getIndexTime();
  payload['op'] = "gettags";
  return {
    type: actions.getLogGroupTags,
    payload: Service.complete(payload),
    meta: {id},
  };
};

export const setLogTimeApi = (id = getUUID()) => {
  const payload = getCommonPayload();
  payload['op'] = "settime";
  return {
    type: actions.setLogTime,
    payload: Service.complete(payload),
    meta: { id },
  };
};

export const autocompleteApi = (searchValue: any, id = getUUID()) => {
  const payload = getCommonPayload();
  payload['op'] = "autocomplete";
  payload['term'] = searchValue;
  return {
    type: actions.autocomplete,
    payload: Service.complete(payload),
    meta: {id},
  };
};

export const searchApi = (searchValue: any, id = getUUID()) => {
  const payload = getCommonPayload();
  payload['op'] = "complete";
  payload['term'] = searchValue;
  const { user } = store.getState();
  const userName = user.userName;
  return {
    type: actions.search,
    payload: Service.complete(payload),
    meta: {id, payload, userName},
  };
};

export const chatApi = (question: any, session: any, id = getUUID()) => {
  const payload = getCommonPayload();
  payload['op'] = "chat";
  payload['term'] = question;
  if (session !== "") {
    payload['session'] = session;
  }
  return {
    type: actions.chat,
    payload: Service.complete(payload),
    meta: {id},
  };
};

export const storeChatInfo = (chat: any) => {
  const indexInfo = getCommonPayload();
  const { user } = store.getState();
  const userName = user.userName;
  return {
    type: actions.storeChat,
    payload: chat,
    meta: {indexInfo, userName},
  };
};