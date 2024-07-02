import {
  isFulfilledApi,
  isPendingApi,
  isRejectedApi,
} from './utils';

export const reducer = (state = {}, action: any) => {
  if (isPendingApi(action)) {
    if (!action?.meta?.id) {
      return state;
    }
    return {
      ...state,
      [action.meta.id]: { id: action.meta.id, active: true },
    };
  } else if (isFulfilledApi(action)) {
    if (!action?.meta?.id) {
      return state;
    }
    return {
      ...state,
      [action.meta.id]: { id: action.meta.id, active: false },
    };
  } else if (isRejectedApi(action)) {
    if (!action?.meta?.id) {
      return state;
    }
    return {
      ...state,
      [action.meta.id]: { id: action.meta.id, active: false, error: action.payload },
    };
  } else {
    return state;
  }
};

export const getApis = (requestIds: any, state: any) => {
  return requestIds.map((id: any) => state.apis[id] || {});
};
