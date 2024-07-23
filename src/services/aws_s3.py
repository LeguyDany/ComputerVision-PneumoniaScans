import logging
import os
from botocore.exceptions import ClientError
import boto3
from dotenv import load_dotenv


class AWS_S3:
    def __init__(self):
        load_dotenv()
        self.client = boto3.client('s3')

        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        self.client = session.client('s3', region_name=os.getenv('AWS_DEFAULT_REGION'))

    def upload_file(self, file_name, bucket, object_name=None):
        if object_name is None:
            object_name = os.path.basename(file_name)

        try:
            self.client.upload_file(
                file_name, 
                bucket, 
                object_name
            )
        except ClientError as e:
            logging.error(e)
            return False
        return True

    def download_file(self, bucket, object_name, file_name):
        try:
            self.client.download_file(bucket, object_name, file_name)
        except ClientError as e:
            logging.error(e)
            return False
        return True