#!/bin/bash 
eb init -p python-3.6 whatrthose-dev --region us-west-2
ERR=$((eb create whatrthose-dev >&1) 2>&1)
if [[ $ERR == *"already exists"* ]]; then
	eb terminate whatrthose-dev --force
	eb create whatrthose-dev	
fi
