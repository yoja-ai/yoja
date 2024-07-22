import { combineReducers } from "redux";
// import { entities } from "../home/reduser"
import * as auth from "./reducers/auth.reducer";
import * as user from "./reducers/user.reducer";
import * as apis from "./reducers/api.reducer";
import * as configuration from "./reducers/configuration.reducer";

export const rootReducer = combineReducers({
  auth: auth.reducer,
  user: user.reducer,
  apis: apis.reducer,
  configuration: configuration.reducer,
  // entities,
});
