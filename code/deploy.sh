#!/bin/bash
while test $# -gt 0; do
        case "$1" in
                -h|--help)
                        echo "Deploying WhatRThose Severless Application"
                        echo " "
                        echo "bash deploy.sh [options]"
                        echo " "
                        echo "options:"
                        echo "-h, --help		show brief help"
                        echo "-bucket       		specify S3 bucket name to log user input"
                        echo "-region     		specify AWS region (us-east-1, us-west-2, etc)"
			                  echo "-ebname 			specify the ElasticBeanstalk environment name"
                        exit 0
                        ;;
                -bucket)
			                 shift
                       BUCKETNAME=$1
			                 shift
                        ;;
                -region)
			                  shift
                        REGION=$1
			                  shift
                        ;;
		            -ebname)
			                 shift
			                 ENVNAME=$1
			                 shift
        esac
done

source activate whatrthose
echo "Creating s3 bucket"
if [[ $(aws s3 ls | grep $BUCKETNAME) ]]; then
	echo "Using existing bucket"
else
	OUTPUT=$((aws s3 mb s3://$BUCKETNAME >&1) 2>&1)
	if [[ $OUTPUT == *"BucketAlreadyExists"* ]]; then
		echo "Bucket name "$BUCKETNAME" exists - choose a different bucket"
		exit 0
	fi

	if [[ $OUTPUT == $"failed"* ]]; then
		echo "AWS S3 failure: "$OUTPUT
		echo "Terminating deployment"
		exit 0
	else
		echo "Bucket "$BUCKETNAME" created successfully"
	fi
fi
echo "bucket = 's3://"$BUCKETNAME"'" > ~/whatrthose/code/config.py

echo "Creating EB"
mkdir ~/.elasticbeanstalk
echo "deploy:
   artifact: /home/ec2-user/whatrthose/code/code.zip" > ~/.elasticbeanstalk/config.yml
ERR2=$((eb init -p python-3.6 $ENVNAME --region $REGION >&1) 2>&1)
ERR=$((eb create $ENVNAME >&1) 2>&1)
if [[ $ERR == *"already exists"* ]]; then
	eb terminate $ENVNAME --force
	eb create $ENVNAME
fi
echo "Successfully created eb environment "$ENVNAME" in region "$REGION

CNAME=$((aws elasticbeanstalk describe-environments --environment-names whatrthose-dev --region us-west-2 |  python3 -c "import sys, json; print(json.load(sys.stdin)['Environments'][0]['CNAME'])">&1) 2>&1)

echo "App can be found at "$CNAME
