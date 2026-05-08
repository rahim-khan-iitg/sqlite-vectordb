from src.client import Client
from src.models import CollectionMeta,DistanceMetric
import  numpy as np
from src.models import Point
from uuid import uuid4


client=Client("./test_db")
# client.create_collection("test_collection2",CollectionMeta(collection_name="test_collection2",embedding_size=128,distance_metric=DistanceMetric.COSINE))

collection=client.get_collection("test_collection2")
collection.store_points([Point(id=str(uuid4()),content="hello",payload={"content_type":"text",},embedding=np.random.rand(128).tolist())])
print(collection.embeddings)
print(collection.ids)