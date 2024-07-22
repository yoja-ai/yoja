import { applyMiddleware, compose, Store, createStore } from 'redux';
import { persistStore } from "redux-persist";
import { rootReducer } from "./rootReducer";
import { thunk } from 'redux-thunk';

const store: Store = createStore(rootReducer, {}, applyMiddleware(thunk));

export const persistor = persistStore(store);
export default store;
