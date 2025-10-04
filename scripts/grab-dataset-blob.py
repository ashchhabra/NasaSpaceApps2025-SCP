import pandas as pd
from azure.storage.blob import BlobServiceClient
import io

# ---------- Config ----------
connect_str = "BlobEndpoint=https://datasetsnasa2025.blob.core.windows.net/;QueueEndpoint=https://datasetsnasa2025.queue.core.windows.net/;FileEndpoint=https://datasetsnasa2025.file.core.windows.net/;TableEndpoint=https://datasetsnasa2025.table.core.windows.net/;SharedAccessSignature=sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-11-05T01:07:05Z&st=2025-10-04T15:52:05Z&spr=https&sig=pkjW4WcTAalhdlrYIFN2qu14xeLVJPTgYuZWsnnZwVk%3D"
container_name = "nasa-exoplanets-split"

# ---------- Connect ----------
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(container_name)

# ---------- List and filter blobs starting with 'consolidated' ----------
blobs = [b for b in container_client.list_blobs() if b.name.startswith("consolidated")]
if not blobs:
    raise ValueError("No blobs starting with 'consolidated' found!")

# ---------- Pick the latest modified ----------
latest_blob = sorted(blobs, key=lambda b: b.last_modified, reverse=True)[0]
print(f"Latest consolidated blob: {latest_blob.name}")

# ---------- Download into memory ----------
blob_client = container_client.get_blob_client(latest_blob.name)
stream = blob_client.download_blob().readall()

# ---------- Load into pandas ----------
df = pd.read_csv(io.BytesIO(stream), engine='python', on_bad_lines='skip')
print(df.head())
