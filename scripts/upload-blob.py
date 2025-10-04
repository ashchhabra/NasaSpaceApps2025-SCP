import requests
from azure.storage.blob import BlobServiceClient

# NASA Exoplanet dataset URL
url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv"
filename = "toi.csv"

# Azure connection
connect_str = "BlobEndpoint=https://datasetsnasa2025.blob.core.windows.net/;QueueEndpoint=https://datasetsnasa2025.queue.core.windows.net/;FileEndpoint=https://datasetsnasa2025.file.core.windows.net/;TableEndpoint=https://datasetsnasa2025.table.core.windows.net/;SharedAccessSignature=sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-11-05T01:07:05Z&st=2025-10-04T15:52:05Z&spr=https&sig=pkjW4WcTAalhdlrYIFN2qu14xeLVJPTgYuZWsnnZwVk%3D"
container_name = "nasa-exoplanets"

# Init clients
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(container_name)

# Create container if not exists
try:
    container_client.create_container()
except:
    pass

# Stream download from URL → upload to blob
print("Streaming NASA dataset into Azure Blob...")
response = requests.get(url, stream=True)
blob_client = container_client.get_blob_client(filename)
blob_client.upload_blob(response.content, overwrite=True)

print(f"Uploaded '{filename}' to Blob container '{container_name}'")
