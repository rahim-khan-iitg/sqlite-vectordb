import os
from pathlib import Path
from src.models import Metadata,CollectionMeta
from src.local_collection import LocalCollection
import json

META_FILE_NAME="meta.json"

class Client:
    def __init__(self,location:str,use_gpu:bool=True):
        self.db_location=location
        self.use_gpu=use_gpu
        self.meta_file_path=Path(self.db_location)/META_FILE_NAME
        os.makedirs(self.db_location,exist_ok=True)
        self.metadata = self.load_metadata()

    def load_metadata(self):
        if os.path.exists(self.meta_file_path):
            with open(self.meta_file_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            return Metadata(**metadata)
        return Metadata(collections={})

    def save_metadata(self):
        with open(self.meta_file_path, "w", encoding="utf-8") as file:
            json.dump(self.metadata.model_dump(mode="json"), file, indent=4)

    def get_collection(self, collection_name: str):
        collection_exist = self.metadata.collections.get(collection_name)

        if not collection_exist:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        return LocalCollection(collection_name, self.db_location,self.use_gpu)

    def create_collection(self, collection_name: str, config: CollectionMeta):
        collection_exist = self.metadata.collections.get(collection_name)

        if collection_exist is not None:
            raise ValueError(f"Collection '{collection_name}' already exists")

        self.metadata.collections[collection_name] = config
        os.makedirs(Path(self.db_location)/collection_name,exist_ok=True)
        self.save_metadata()

