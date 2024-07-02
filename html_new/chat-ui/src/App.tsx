import React from 'react';
import './App.css';
import { Provider } from "react-redux";
import { HashRouter as Router } from "react-router-dom";
import { PersistGate } from "redux-persist/integration/react";
import AppRoot from './AppRoot';

function App({ store, persistor }: any) {
  return (
    <Provider store={store}>
      <PersistGate persistor={persistor}>
        <React.Suspense fallback={<div> loading </div>}>
          <Router>
            <div className="App">
              <AppRoot />
            </div>
          </Router>
        </React.Suspense>
      </PersistGate>
    </Provider>
  );
}

declare global {
  const google: any;
}

export default App;
