import cfnresponse
import os
from typing import Any, Dict, List, Tuple, Union
import typing
import boto3
import botocore
import botocore.client
if typing.TYPE_CHECKING:
    from mypy_boto3_dynamodb.client import DynamoDBClient
else:
    DynamoDBClient = object
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mypy_boto3_dynamodb import DynamoDBClient
    from mypy_boto3_dynamodb.type_defs import ExecuteStatementOutputTypeDef
    from mypy_boto3_dynamodb.service_resource import Table
    
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_s3.type_defs import GetBucketLocationOutputTypeDef
else:
    DynamoDBClient = object
    ExecuteStatementOutputTypeDef = object
    Table = object
    
    CognitoIdentityProviderClient = object
    AdminListGroupsForUserResponseTypeDef = object; GroupTypeTypeDef=object; AdminGetUserResponseTypeDef=object; AttributeTypeTypeDef = object; AdminCreateUserResponseTypeDef = object; UserTypeTypeDef = object; EmptyResponseMetadataTypeDef = object
    
    S3Client = object
    GetBucketLocationOutputTypeDef = object
#from cryptography.fernet import Fernet

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.getLevelName(os.getenv("LAMBDA_LOG_LEVEL", "INFO")))

def _insert_serviceconf_partiql(serviceconf_table_name:str, index_bucket_name:str) -> Tuple[bool, str]:
    try:
        # # https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ql-reference.update.html
        # # UPDATE  table  
        # # [SET | REMOVE]  path  [=  data] […]
        # # WHERE condition [RETURNING returnvalues]
        # # <returnvalues>  ::= [ALL OLD | MODIFIED OLD | ALL NEW | MODIFIED NEW] *
        # set_clauses:str = ""
        # for key,val in update_dict.items():
        #     set_clauses = set_clauses + f" SET {key}='{val}' "
        # # Note: table name must be double quoted due to the use of '-' in the name: infinstor-Subscribers
        # # Error: ValidationException: Where clause does not contain a mandatory equality on all key attributes
        # update_stmt:str = f"UPDATE \"{os.environ['SUBSCRIBERS_TABLE']}\" {set_clauses} WHERE customerId = '{customer_id}' AND productCode = '{product_code}' RETURNING ALL NEW *"
        # print(f"Executing update_stmt for subscriber {cognito_username} = {update_stmt}")
        
        ddb_client:DynamoDBClient = boto3.client('dynamodb')
        if not ddb_client: return False, "Unable to get ddb_client for service"
        
        # https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ql-reference.insert.html
        insert_stmt:str = f"INSERT INTO \"{serviceconf_table_name}\" VALUE " + "{" + f"'configVersion':1, 'bucket':'{index_bucket_name}', 'prefix':'index2' " +  "}"
        
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb.html#DynamoDB.Client.execute_statement
        # https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/API_ExecuteStatement.html
        # response = {'Items': [{...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, ...], 'ResponseMetadata': {'RequestId': 'CHBDU0TEJ6UH82VNOMPA9F8I3NVV4KQNSO5AEMVJF66Q9ASUAAJG', 'HTTPStatusCode': 200, 'HTTPHeaders': {...}, 'RetryAttempts': 0}}
        exec_stmt_resp:ExecuteStatementOutputTypeDef = ddb_client.execute_statement(Statement=insert_stmt)
        ddb_http_status_code:int = exec_stmt_resp.get('ResponseMetadata').get('HTTPStatusCode')
        if ddb_http_status_code != 200: return False, f"dynamodb http status code={ddb_http_status_code} headers={exec_stmt_resp['ResponseMetadata'].get('HTTPHeaders')}"
        
        return True, ""
    except Exception as e:
        logger.error(f"Exception caught during ddb serviceconf statment: {e}", exc_info=e)
        return False, str(e)

def _delete_serviceconf_partiql(serviceconf_table_name:str) -> Tuple[bool, str]:
    try:
        # # https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ql-reference.update.html
        # # UPDATE  table  
        # # [SET | REMOVE]  path  [=  data] […]
        # # WHERE condition [RETURNING returnvalues]
        # # <returnvalues>  ::= [ALL OLD | MODIFIED OLD | ALL NEW | MODIFIED NEW] *
        # set_clauses:str = ""
        # for key,val in update_dict.items():
        #     set_clauses = set_clauses + f" SET {key}='{val}' "
        # # Note: table name must be double quoted due to the use of '-' in the name: infinstor-Subscribers
        # # Error: ValidationException: Where clause does not contain a mandatory equality on all key attributes
        # update_stmt:str = f"UPDATE \"{os.environ['SUBSCRIBERS_TABLE']}\" {set_clauses} WHERE customerId = '{customer_id}' AND productCode = '{product_code}' RETURNING ALL NEW *"
        # print(f"Executing update_stmt for subscriber {cognito_username} = {update_stmt}")
        
        ddb_client:DynamoDBClient = boto3.client('dynamodb')
        if not ddb_client: return False, "Unable to get ddb_client for service"
        
        # https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ql-reference.delete.html
        delete_stmt:str = f"DELETE FROM \"{serviceconf_table_name}\" WHERE \"configVersion\" = 1" 
        
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb.html#DynamoDB.Client.execute_statement
        # https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/API_ExecuteStatement.html
        # response = {'Items': [{...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, ...], 'ResponseMetadata': {'RequestId': 'CHBDU0TEJ6UH82VNOMPA9F8I3NVV4KQNSO5AEMVJF66Q9ASUAAJG', 'HTTPStatusCode': 200, 'HTTPHeaders': {...}, 'RetryAttempts': 0}}
        exec_stmt_resp:ExecuteStatementOutputTypeDef = ddb_client.execute_statement(Statement=delete_stmt)
        ddb_http_status_code:int = exec_stmt_resp.get('ResponseMetadata').get('HTTPStatusCode')
        if ddb_http_status_code != 200: return False, f"dynamodb http status code={ddb_http_status_code} headers={exec_stmt_resp['ResponseMetadata'].get('HTTPHeaders')}"
        
        return True, ""
    except Exception as e:
        logger.error(f"Exception caught during ddb serviceconf statment: {e}", exc_info=e)
        return False, str(e)

def lambda_handler(event, context):
  print('evt=' + str(event))
  request_type:str = event['RequestType']
  serviceConfTableName:str = event['ResourceProperties']['serviceConfTableName']
  try:
    # 'Create' | 'Update' | 'Delete'
    if request_type.lower() == 'create':
      indexBucketName:str = event['ResourceProperties']['indexBucketName']
      # don't populate the 'key' in serviceconf; autopopulated by the backend in the api if it doesn't exist
      # keyStr:str  = '' #Fernet.generate_key().decode()
     
      success, errmsg = _insert_serviceconf_partiql(serviceConfTableName, indexBucketName)
        
      responseData = {}
      print(responseData)
      cfnresponse.send(event, context, cfnresponse.SUCCESS if success else cfnresponse.FAILED, responseData if success else {'error':errmsg} , serviceConfTableName)
    # delete
    elif request_type.lower() == 'delete':
      serviceConfTableName:str = event['ResourceProperties']['serviceConfTableName']
      success, errmsg = _delete_serviceconf_partiql(serviceConfTableName)
        
      responseData = {}
      print(responseData)
      cfnresponse.send(event, context, cfnresponse.SUCCESS if success else cfnresponse.FAILED, responseData if success else {'error':errmsg} , serviceConfTableName)
    # update
    else:
      #pass
      cfnresponse.send(event, context, cfnresponse.SUCCESS, {},serviceConfTableName)
  except Exception as e:
    logging.error(f'populate_ddb_custom_resource lambda handler: caught exception={e}', exc_info=e)
    cfnresponse.send(event, context, cfnresponse.FAILED, { 'error': str(e) }, serviceConfTableName)

if __name__ == '__main__':
  print ( lambda_handler( {
    'RequestType':'Create',
    'ResourceProperties': {
      'indexBucketName':'indexBucketName',
      'serviceConfTableName': 'yoja-ServiceConf'
    }
  }, None ) )
  print ( lambda_handler( {
    'RequestType':'Delete',
    'ResourceProperties': {
      'indexBucketName':'indexBucketName',
      'serviceConfTableName': 'yoja-ServiceConf'
    }
  }, None ) )
  
