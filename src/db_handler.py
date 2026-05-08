import sqlite3
from typing import List
from pathlib import Path
from src.models import Point
import pickle

DB_FILE_NAME = "storage.sqlite"


class DBHandler:
    def __init__(self, location: str) -> None:
        self.location = Path(location)
        self.conn = sqlite3.connect(self.location / DB_FILE_NAME)
        self.cur = self.conn.cursor()
        self.create_table()

    def create_table(self) -> None:
        self.cur.execute(
            "CREATE TABLE IF NOT EXISTS points (id TEXT PRIMARY KEY, point BLOB)"
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def store_point(self, point: Point) -> None:
        key = point.id
        value = pickle.dumps(point)
        self.cur.execute(
            "INSERT OR REPLACE INTO points VALUES (?, ?)",
            (
                key,
                sqlite3.Binary(value),
            ),
        )
        self.conn.commit()

    def delete_point(self, point_id: str) -> None:
        self.cur.execute("DELETE FROM points WHERE id = ?", (point_id,))
        self.conn.commit()

    def load_point(self, point_id: str) -> Point:
        self.cur.execute("SELECT point FROM points WHERE id=?", (point_id,))
        point = self.cur.fetchone()
        point = pickle.loads(point[0])
        return point

    def get_embeddings(self) -> tuple[List[str],List[List[float]]]:
        """Get all embeddings and ids"""
        embeddings = []
        ids=[]
        rows=self.cur.execute("SELECT id,point FROM points")
        for id_,point in rows:
            p = pickle.loads(point)
            embeddings.append(p.embedding)
            ids.append(id_)
        return ids,embeddings
