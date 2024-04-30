import boto3
import json

#
# You need this policy to invoke the model
# 
# {
#    "Version": "2012-10-17",
#    "Statement": [
#        {
#            "Effect": "Allow",
#            "Action": "bedrock:InvokeModel",
#            "Resource": "*"
#        }
#    ]
# }
#
bedrock = boto3.client(
  service_name='bedrock-runtime', 
  region_name="us-east-1"
)

prompt = """
Write a medium blog post on how to use 
Amazon Bedrock to write an article on how to use Bedrock.
"""

body = json.dumps({
    "prompt": prompt,
    "temperature": 0.75,
})

modelId = 'meta.llama3-8b-instruct-v1:0'
accept = 'application/json'
contentType = 'application/json'

response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
response_body = json.loads(response.get('body').read())
keys = response_body.keys()

print(f"Avaliable result keys: {keys}")
print(f"Result from LLM: {response_body['generation']}")