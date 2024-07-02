import { fulfilled, rejected } from "./utils";

export const LOGIN_API = "LOGIN_API";

const initialUserState = {
  fetching: true,
};

export const reducer = (state = initialUserState, action: any) => {
  switch (action.type) {
    
    case fulfilled(LOGIN_API): {
      const user = action.payload.userName ? action.payload : state;
      user.fetching = false;
      user.authorized = true;
      return user;
    }

    case rejected(LOGIN_API): {
      return {fetching: false};
    }

    case "[Logout] Action": {
      return { fetching: false };
    }

    default:
      return state;
  }
}
