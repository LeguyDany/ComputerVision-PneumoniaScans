import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

from services.azure_bucket import AzureBucket

bucket = AzureBucket()
bucket.download_locally()