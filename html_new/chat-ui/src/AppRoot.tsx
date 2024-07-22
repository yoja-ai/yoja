import React, { Component, useEffect } from "react";
import { BallTriangle } from 'react-loader-spinner'
import Layout from "./components/layout/Layout";
import Login from "./Login";
import { loginApi } from "./services/ChatService";

const AppRoot = () => {
  const [isAuthorized, setIsAuthorized] = React.useState(false);
  const [isLoading, setIsLoading] = React.useState(true);

  useEffect(() => {
    loginApi().then((res: any) => {
      if(res.status === 200 ) {
        setIsAuthorized(true);
      } else {
        setIsAuthorized(false);
      }
    }).catch((e: any) => {
      setIsAuthorized(false);
    }).finally(()=> {
      setIsLoading(false);
    })
  }, []);

  if(isLoading) {
    return(
      <div className="loadingSpiner">
        <BallTriangle
          height={100}
          width={100}
          radius={5}
          color="#f47e60"
          ariaLabel="ball-triangle-loading"
          wrapperStyle={{}}
          wrapperClass=""
          visible={true}
        />          
      </div>
    );
  } else {
    if(!isAuthorized) {
      // if login page add here..
      //return <Login />
      return <Layout />
    } else {
      return (
        <Layout />
      );
    }
  }
}

export default AppRoot;