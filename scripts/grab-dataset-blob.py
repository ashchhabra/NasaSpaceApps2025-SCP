import pandas as pd
from azure.storage.blob import BlobServiceClient
import io

connect_str = "BlobEndpoint=https://datasetsnasa2025.blob.core.windows.net/;QueueEndpoint=https://datasetsnasa2025.queue.core.windows.net/;FileEndpoint=https://datasetsnasa2025.file.core.windows.net/;TableEndpoint=https://datasetsnasa2025.table.core.windows.net/;SharedAccessSignature=sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-11-05T01:07:05Z&st=2025-10-04T15:52:05Z&spr=https&sig=pkjW4WcTAalhdlrYIFN2qu14xeLVJPTgYuZWsnnZwVk%3D"
container_name = "nasa-exoplanets"
blob_name = "toi.csv"

# Connect
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
blob_client = blob_service_client.get_container_client(container_name).get_blob_client(blob_name)

# Download blob into memory
stream = blob_client.download_blob().readall()

# Load into DataFrame directly
df = pd.read_csv(io.BytesIO(stream))

print(df.head())
