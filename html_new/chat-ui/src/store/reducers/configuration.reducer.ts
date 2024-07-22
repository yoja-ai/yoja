import { persistReducer } from "redux-persist";
import storage from "redux-persist/lib/storage";

export const actionTypes = {
  setIndexTime: "SET_INDEX_TIME",
  setLogGroups: "SET_LOG_GROUPS",
  setTags: "SET_TAGS",
  setConfig: "SET_CONFIG",
  SET_CONFIG_sucess: "SET_CONFIG_sucess"
};

let currentDate = new Date();
let year = currentDate.getFullYear();
let month = currentDate.getMonth();
let day = currentDate.getDate();
let hours = currentDate.getHours();

let roundedDownEnd = new Date(year, month, day, hours - 1, 0, 0, 0 );
let roundedDownStart = new Date(year, month, day - 1, hours - 1, 0, 0, 0 );

const initialConfigurationState = {
  startTime: roundedDownStart,
  endTime: roundedDownEnd,
  filterType: 'None',
  logGroups: [],
  tagKey: '',
  tagValue: '',
};


export const reducer = persistReducer(
  { storage, key: "index-config", whitelist: ["startTime", "endTime", "filterType", "logGroups", "tagKey", "tagValue"] },
  (state = initialConfigurationState, action: any) => {
    switch (action.type) {
      case "SET_INDEX_TIME": {
        const config = action.payload;
        return config;
        // return initialConfigurationState;
      }
      case "INIT_CONFIG": {
        return initialConfigurationState;
      }

      case "[Logout] Action": {
        return { };
      }

      default:
        return state;
    }
  }
);
