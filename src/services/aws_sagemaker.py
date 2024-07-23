import os
import boto3
from dotenv import load_dotenv

class AWS_SAGEMAKER:
    def __init__(self):
        load_dotenv()
        self.client = boto3.client('sagemaker')
        self.client_runtime = boto3.client('sagemaker-runtime') 

        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        self.client = session.client('sagemaker', region_name=os.getenv('AWS_DEFAULT_REGION'))

        self.client_runtime = session.client('sagemaker-runtime', region_name=os.getenv('AWS_DEFAULT_REGION'))



    def setup_sagemaker(self, model_name:str, bucket:str, tarball_key:str = None):
        if tarball_key == None:
            tarball_key = os.path.join(os.getcwd(), "src", "processed", "deployment.tar.gz")

        model_url = f"s3://{bucket}/{tarball_key}"

        response = self.client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': '763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-inference:2.3.0-cpu-py311-ubuntu20.04-sagemaker',
                'ModelDataUrl': model_url
            },
            ExecutionRoleArn='arn:aws:iam::590183748926:role/Zoidberg'
        )

        print(response)

    def setup_endpoint(self, config_name, instance_type, model_name):
        response = self.client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'default',
                    'ModelName': model_name,
                    'InstanceType': instance_type,
                    'InitialInstanceCount': 1
                }
            ]
        )

        print(response)

    def create_endpoint(self, endpoint_name, config_name):
        response = self.client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )

        print(response)

    def use_endpoint(self, endpoint_name, data):
        response = self.client_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=data,
            ContentType='multipart/form-data'
        )

        print(response['Body'].read())