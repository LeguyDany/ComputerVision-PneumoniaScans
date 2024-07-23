# Terraform setup

You can setup a vm instance on GCP from this terraform folder. You first need to create a credential file from your GCP portal and place it at the project's root. Rename it as follow: gcp_api_key.json.

Make sure you have terraform installed on your machine first. You can install it on a macOS machine with this command:

```bash
brew install terraform
```

Head to the cloud/terraform/gcp_instance_setup folder and execute the following command:

```bash
terraform init
```

Make sure your ssh keys are setupin a id_ed25519 file and id_ed25519.pub file. Once this is done, select a region in the ./cloud/terraform/gcp_instance_setup/gcp_instance_setup.tf file by commenting or decommenting lines. Once it's done, make sure you have your .env file at the project's root with this line filled:

```.env
TF_VAR_USER_PASSWORD="<Your user password>"
```

Then run the following command to setup your instance:

```terraform
terraform apply
```

Once it's done, follow the instructions in the ansible readme.md folder.