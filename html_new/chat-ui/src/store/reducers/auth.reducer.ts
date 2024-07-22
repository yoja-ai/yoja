import { persistReducer } from "redux-persist";
import storage from "redux-persist/lib/storage";
import { Service } from "../services";

export const actionTypes = {
  Login: "[Login] Action",
  Logout: "[Logout] Action",
  SetUserAttributes: "[SetUserAttributes] Action",
};

const initialAuthState = {
  idToken: undefined,
  refreshToken: undefined,
  accessToken: undefined,
  tokenTime: undefined,
  userAttributes: {},
};

export const reducer = persistReducer(
    { storage, key: "inf-auth", whitelist: ["idToken", "tokenTime", "refreshToken"] },
    (state: any = initialAuthState, action: any) => {
      switch (action.type) {
        case actionTypes.SetUserAttributes: {
          return {...state, userAttributes: action.payload};
        }

        case actionTypes.Login: {
          return action.payload;
        }

        case actionTypes.Logout: {
          return initialAuthState;
        }

        default:
          return state;
      }
    }
);

export const actions = {
  login: (authInfo: any) => ({ type: actionTypes.Login, payload: authInfo }),
  logout: () => ({ type: actionTypes.Logout }),
  setUserAttributes: (attributes: any) => ({ type: actionTypes.SetUserAttributes, payload: attributes}),
  getUser: (data: any) => ({ type: "GET_CUSTOMER_INFO_API", payload: Service.getCustomerInfo(data) }),
  setInitConfiguration: () => ({ type: "INIT_CONFIG" }),
  setlastActiviteAsConfiguration: (payload: any) => ({ type: "SET_INDEX_TIME", payload: payload }),
};
