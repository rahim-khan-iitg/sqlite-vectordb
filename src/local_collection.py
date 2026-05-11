from src.db_handler import DBHandler
import os
from typing import List
from src.models import Point,DistanceMetric
import numpy as np
from src.cuda_similarity import Index


class LocalCollection:
    def __init__(self,collection:str,location:str,use_gpu:bool=True):
        self.location=location
        self.collection=collection
        self.use_gpu=use_gpu
        self.db=DBHandler(os.path.join(location,collection))
        self.ids,self.embeddings=self.db.get_embeddings()
        if use_gpu:
            self.index=Index(self.embeddings)

    def store_points(self,points:List[Point])->None:
        """Store points"""
        for point in points:
            self.db.store_point(point)
        self.ids,self.embeddings=self.db.get_embeddings()
        if self.use_gpu:
            del self.index
            self.index=Index(self.embeddings)


    def delete_points(self,ids:List[str])->None:
        """Delete points"""
        for id_ in ids:
            self.db.delete_point(id_)
        self.ids,self.embeddings=self.db.get_embeddings()
        if self.use_gpu:
            del self.index
            self.index=Index(self.embeddings)

    def get_point(self,point_id:str)->Point:
        """Get point by id"""
        return self.db.load_point(point_id)
    
    def embedding_retrieve(self,query_embedding:List[float],k:int)->List[Point]:
        """Retrieve points by embedding"""
        if self.use_gpu:
            scores = self.index.search(query_embedding)
        else:
            embeddings = np.array(self.embeddings)
            query_embedding = np.array(query_embedding)
            scores = np.dot(embeddings, query_embedding)/np.linalg.norm(embeddings,axis=1,keepdims=True)/np.linalg.norm(query_embedding)
        top_k = np.argsort(scores)[-k:]
        points = []
        for i in top_k:
            points.append(self.db.load_point(self.ids[i]))
        return points
    
    def knn_retrieve(self,query_embedding:List[float],k:int,metric:DistanceMetric)->List[Point]:
        """Retrieve points by embedding"""
        if self.use_gpu:
            scores,indices = self.index.search_knn(query_embedding,k,metric)
            indices=indices.tolist()
        else:
            embeddings = np.array(self.embeddings)
            query_embedding = np.array(query_embedding)
            scores = np.dot(embeddings, query_embedding)/np.linalg.norm(embeddings,axis=1,keepdims=True)/np.linalg.norm(query_embedding)
            indices = np.argsort(scores)[-k:]
        points = []
        for i in indices:
            points.append(self.db.load_point(self.ids[i]))
        return points