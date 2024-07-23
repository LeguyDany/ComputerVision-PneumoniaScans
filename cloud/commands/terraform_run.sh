#!/bin/bash

export $(grep -v '^#' ../../.env | xargs)

cd ../terraform/gcp_instance_setup
terraform apply