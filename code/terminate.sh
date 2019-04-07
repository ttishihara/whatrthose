#!/bin/bash
while test $# -gt 0; do
        case "$1" in
                -h|--help)
                        echo "Deploying WhatRThose Severless Application"
                        echo " "
                        echo "bash deploy.sh [options]"
                        echo " "
                        echo "options:"
                        echo "-h, --help                show brief help"
                        echo "-bucket                   specify S3 bucket name to log user input"
                        echo "-region                   specify AWS region (us-east-1, us-west-2, etc)"
                        echo "-ebname                   specify the ElasticBeanstalk environment name"
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

aws s3 rb s3://$BUCKETNAME --force
aws elasticbeanstalk terminate-environment --environment-name $ENVNAME --region $REGION
