import json
import cfnresponse
import re
import logging
import boto3

def lambda_handler(event, context):
  print('evt=' + str(event))
  request_type:str = event['RequestType']
  yojaServiceDnsName:str = event['ResourceProperties']['yojaServiceDnsName']
  try:
    # 'Create' | 'Update' | 'Delete'
    if request_type.lower() == 'create' or request_type.lower() == 'update':
      htmlBucketPrefix:str = event['ResourceProperties']['htmlBucketPrefix']
      indexBucketPrefix:str = event['ResourceProperties']['indexBucketPrefix']
      scratchBucketPrefix:str = event['ResourceProperties']['scratchBucketPrefix']
      
      yojaServiceDnsNameWithDashes:str = yojaServiceDnsName.replace('.', '-')
      
      htmlBucketName:str; indexBucketName:str; scratchBucketName:str
      htmlBucketName, indexBucketName, scratchBucketName = [ bucket_name_pre + '-' + yojaServiceDnsNameWithDashes for bucket_name_pre in [htmlBucketPrefix, indexBucketPrefix, scratchBucketPrefix] ]
  
      # https://docs.python.org/3/library/re.html
      # https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html: Bucket names must be between 3 (min) and 63 (max) characters long; Bucket names can consist only of lowercase letters, numbers, dots (.), and hyphens (-)
      # output_bucket_name,count_subs = re.subn('[^a-z0-9.-]','-', output_bucket_name.lower())
  
      responseData = {}
      responseData['yojaServiceDnsNameWithDashes'] = yojaServiceDnsNameWithDashes
      responseData['htmlBucketName'] = htmlBucketName
      responseData['indexBucketName'] = indexBucketName
      responseData['scratchBucketName'] = scratchBucketName
      #print(responseData)
      cfnresponse.send(event, context, cfnresponse.SUCCESS, responseData, yojaServiceDnsName)
    # delete 
    else:
      #pass
      cfnresponse.send(event, context, cfnresponse.SUCCESS, {},yojaServiceDnsName)
  except Exception as e:
    logging.error(f'custom_resource lambda handler: caught exception={e}', exc_info=e)
    cfnresponse.send(event, context, cfnresponse.FAILED, { 'error': str(e) }, yojaServiceDnsName)

if __name__ == '__main__':
  print ( lambda_handler( {
    'RequestType':'Create',
    'ResourceProperties': {
      'yojaServiceDnsName':'awsstaging6.yoja.ai',
      'htmlBucketPrefix': 'htmlBucketPrefix',
      'indexBucketPrefix': 'indexBucketPrefix',
      'scratchBucketPrefix': 'scratchBucketPrefix',
    }
  }, None ) )
