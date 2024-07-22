import { postJson } from "../utils/AxiosUtils";

const servicesConfig = (window as any).ServiceConfig;
export class Service {
  static complete = (data: any) =>
    postJson({ relativeUrl: 'https:///Prod/1.0/cwsearchvdb/complete', data });

  static getCustomerInfo = (data: any) =>
    postJson({ relativeUrl: + '/customerinfo', data });

  static login = (data: any) =>
    postJson({ relativeUrl: servicesConfig.envAPIEndpoint + '/login', data });

}
