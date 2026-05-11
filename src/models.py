from pydantic import BaseModel
from typing import  Dict,List,Any,Optional
from enum import Enum
class Point(BaseModel):
    id:str
    content:str
    payload:Optional[Dict[str,Any]]
    embedding:Optional[List[float]]

class DistanceMetric(Enum):
    COSINE="cosine"
    MANHATTAN="cityblock"
    EUCLIDEAN="euclidean"

class CollectionMeta(BaseModel):
    collection_name:str
    embedding_size:int
    distance_metric:DistanceMetric

class Metadata(BaseModel):
    collections:Dict[str,CollectionMeta]