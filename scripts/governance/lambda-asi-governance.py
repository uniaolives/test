import boto3
import os
import json
from datetime import datetime

# Invariante PCR0 aprovado (armazenado em Secrets Manager)
APPROVED_PCR0 = os.environ.get('APPROVED_PCR0', '0x7f3b2a1c9e8d5f4a2b3c1d0e9f8a7b6...')

def get_parent_instance(enclave_id):
    """
    Finds the parent instance ID for a given Nitro Enclave ID.
    Requires 'ec2:DescribeInstances' permission.
    """
    ec2 = boto3.client('ec2')
    # Nitro Enclaves are associated with parent instances.
    # We can search for instances that have enclaves enabled and match tags or other metadata.
    # In a real scenario, we might use a mapping stored in DynamoDB or search by Enclave ID if supported.
    response = ec2.describe_instances(
        Filters=[
            {'Name': 'enclave-options.enabled', 'Values': ['true']},
            {'Name': 'instance-state-name', 'Values': ['running']}
        ]
    )

    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            # Here we would have logic to match the specific enclave to the instance.
            # For this implementation, we assume a tagging mechanism or specific ID match.
            return instance['InstanceId']

    return None

def lambda_handler(event, context):
    # Trigger: Execution of Nitro enclave or AWS Config rule
    detail = event.get('detail', {})
    enclave_id = detail.get('enclave-id', 'unknown')
    pcr0_current = detail.get('pcr0', '')

    # Gate 2: Verify code integrity
    if pcr0_current != APPROVED_PCR0:
        print(f"VIOLATION: PCR0 mismatch for {enclave_id}. Detected: {pcr0_current}")

        # Karnak Isolation: Terminate parent instance immediately
        ec2 = boto3.client('ec2')
        parent_instance = get_parent_instance(enclave_id)

        if parent_instance:
            print(f"Terminating parent instance {parent_instance} for enclave {enclave_id}")
            try:
                ec2.terminate_instances(InstanceIds=[parent_instance])
            except Exception as e:
                print(f"Failed to terminate instance: {e}")

        # Alert SASC Cathedral (via SNS)
        sns = boto3.client('sns')
        topic_arn = os.environ.get('SNS_TOPIC_ARN')
        if topic_arn:
            sns.publish(
                TopicArn=topic_arn,
                Message=json.dumps({
                    'type': 'CODE_INTEGRITY_VIOLATION',
                    'enclave': enclave_id,
                    'parent': parent_instance,
                    'timestamp': datetime.utcnow().isoformat(),
                    'pcr0_detected': pcr0_current
                })
            )

        return {'complianceType': 'NON_COMPLIANT'}

    return {'complianceType': 'COMPLIANT'}
