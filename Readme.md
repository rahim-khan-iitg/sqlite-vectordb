# 🚀 SQLite-VectorDB

A lightweight, persistent vector database built on top of **SQLite**, featuring **GPU-accelerated** similarity search and **HNSW** indexing.

---

## ✨ Key Features

- 🗄️ **Persistent Storage**: Leverages SQLite for robust, reliable storage of embeddings, metadata, and payloads.
- ⚡ **GPU Acceleration**: Blazing fast cosine similarity search powered by custom CUDA kernels.
- 🧊 **Hybrid Architecture**: Combines the ease of a local database with the performance of high-end vector search engines.
- 🧠 **HNSW Support**: Includes an implementation of the Hierarchical Navigable Small World algorithm for efficient approximate nearest neighbor (ANN) search.
- 🐍 **Pythonic API**: Simple, intuitive client interface for managing collections and points.

---

## 🏗️ Architecture

The project is structured to separate concerns between storage, indexing, and acceleration:

- **`src/client.py`**: The primary interface for database management.
- **`src/local_collection.py`**: Manages individual vector collections and search logic.
- **`src/db_handler.py`**: Handles SQLite interactions and data persistence.
- **`src/cuda_similarity.py`**: Python wrapper for CUDA-based similarity computation.
- **`src/cosine.cu`**: Raw CUDA kernels for high-performance distance calculations.
- **`hnsw.py`**: Standalone implementation of the HNSW algorithm.

---

## 🚀 Quick Start

### 1. Initialize the Client

```python
from src.client import Client
from src.models import CollectionMeta, DistanceMetric

# Initialize client with database location
client = Client("./my_vector_db")

# Create a new collection
config = CollectionMeta(
    collection_name="knowledge_base",
    embedding_size=128,
    distance_metric=DistanceMetric.COSINE
)
client.create_collection("knowledge_base", config)
```

### 2. Store Points

```python
import numpy as np
from src.models import Point
from uuid import uuid4

collection = client.get_collection("knowledge_base")

# Create points with embeddings and optional payload
points = [
    Point(
        id=str(uuid4()),
        content="Artificial Intelligence is fascinating.",
        payload={"category": "tech", "author": "Antigravity"},
        embedding=np.random.rand(128).tolist()
    )
]

collection.store_points(points)
```

### 3. Search

```python
query_vector = np.random.rand(128).tolist()
results = collection.embedding_retrieve(query_vector, k=5)

for point in results:
    print(f"Found point: {point.id} | Content: {point.content}")
```

---

## ⚙️ Requirements

- **Python**: 3.8+
- **CUDA**: Required for GPU acceleration (`libcosine.so`)
- **Dependencies**:
    - `numpy`
    - `pydantic`
    - `sqlite3` (Standard Library)

---

## 🛠️ Build & Installation

Ensure you have the CUDA toolkit installed if you wish to use GPU acceleration. The current implementation relies on a compiled `libcosine.so` located in the `src` directory.

```bash
# Example compilation command (if needed)
nvcc -shared -o src/libcosine.so src/cosine.cu -Xcompiler -fPIC
```

---

## 📜 License

This project is licensed under the MIT License.
