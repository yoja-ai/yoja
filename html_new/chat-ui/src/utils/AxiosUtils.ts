import Axios, { AxiosRequestConfig } from "axios";
import _ from 'lodash';

export const HTTPRetryStatuses = [429, 556];
export const HTTPMethods = {
  get: 'get',
  post: 'post',
  put: 'put',
};

const filterUndefinedFields = (data: any[]) => {
  if (!Array.isArray(data)) {
    return _.pickBy(data, (v) => v !== undefined);
  } else {
    return data.filter((v) => v !== undefined);
  }
};

const generateJsonBody = (data: any) => {
  if (typeof data === 'string') {
    return data;
  } else if (typeof data === 'object') {
    return JSON.stringify(filterUndefinedFields(data));
  } else {
    throw new Error(
      'Unexpected type of input. The REST api payload type must be either an object or a string, got ' +
        typeof data,
    );
  }
};

export const getJson = (props: { relativeUrl: any; data: any; }) => {
  const { relativeUrl, data } = props;
  // filterUndefinedFields(data)
  const queryParams = new URLSearchParams().toString();
  const combinedUrl = queryParams ? `${relativeUrl}?${queryParams}` : relativeUrl;
  return callAxios({ url: combinedUrl, method: HTTPMethods.get });  
};

export const postJson = async(props: { relativeUrl: any; data: any; }) => {
  const { relativeUrl, data } = props;
  return callAxios({
    method: HTTPMethods.post,
    data: generateJsonBody(data),
    url: relativeUrl,
    headers: {
      'Content-Type': 'application/json;charset=UTF-8',
    }
  });
};

export const callAxios = async (config: AxiosRequestConfig<any>) => {
  const response = await Axios(config);
  return response.data? response.data : response;
};
