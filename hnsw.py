"""
HNSW - Hierarchical Navigable Small World Graphs
=================================================
Pure Python + NumPy implementation based on the original paper:
  "Efficient and robust approximate nearest neighbor search using
   Hierarchical Navigable Small World graphs"
   by Yu. A. Malkov, D. A. Yashunin (2018)

Algorithms implemented (matching paper numbering):
  Algorithm 1  - INSERT
  Algorithm 2  - SEARCH-LAYER
  Algorithm 3  - SELECT-NEIGHBORS-SIMPLE
  Algorithm 4  - SELECT-NEIGHBORS-HEURISTIC
  Algorithm 5  - K-NN-SEARCH
"""

import numpy as np
import math
import heapq
from collections import defaultdict
from typing import List, Set, Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Standard L2 distance between two vectors."""
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))


# ---------------------------------------------------------------------------
# Priority-queue wrappers
# (Python's heapq is a min-heap; we sometimes need a max-heap)
# ---------------------------------------------------------------------------


class MinHeap:
    """Min-heap of (distance, element_id) pairs."""

    def __init__(self):
        self._data: List[Tuple[float, int]] = []

    def push(self, dist: float, elem: int):
        heapq.heappush(self._data, (dist, elem))

    def pop(self) -> Tuple[float, int]:
        return heapq.heappop(self._data)

    def peek(self) -> Tuple[float, int]:
        return self._data[0]

    def __len__(self):
        return len(self._data)

    def to_set(self) -> Set[int]:
        return {e for _, e in self._data}

    def to_list(self) -> List[Tuple[float, int]]:
        return list(self._data)


class MaxHeap:
    """Max-heap of (distance, element_id) pairs (negate dist trick)."""

    def __init__(self):
        self._data: List[Tuple[float, int]] = []

    def push(self, dist: float, elem: int):
        heapq.heappush(self._data, (-dist, elem))

    def pop(self) -> Tuple[float, int]:
        neg_d, e = heapq.heappop(self._data)
        return -neg_d, e

    def peek(self) -> Tuple[float, int]:
        neg_d, e = self._data[0]
        return -neg_d, e

    def __len__(self):
        return len(self._data)

    def to_set(self) -> Set[int]:
        return {e for _, e in self._data}


# ---------------------------------------------------------------------------
# Algorithm 3 – SELECT-NEIGHBORS-SIMPLE
# ---------------------------------------------------------------------------


def select_neighbors_simple(
    q: int,
    candidates: Set[int],
    M: int,
    data: np.ndarray,
) -> List[int]:
    """
    Algorithm 3: simply return the M nearest elements from candidates to q.

    Args:
        q          : query element index
        candidates : set of candidate element indices
        M          : number of neighbors to return
        data       : array of all element vectors, shape (N, dim)

    Returns:
        List of up to M nearest element indices from candidates.
    """
    # Sort candidates by distance to q and take the M closest
    dists = [(euclidean_distance(data[q], data[c]), c) for c in candidates]
    dists.sort(key=lambda x: x[0])
    return [e for _, e in dists[:M]]


# ---------------------------------------------------------------------------
# Algorithm 4 – SELECT-NEIGHBORS-HEURISTIC
# ---------------------------------------------------------------------------


def select_neighbors_heuristic(
    q: int,
    candidates: Set[int],
    M: int,
    layer: int,
    data: np.ndarray,
    graph: Dict[int, Dict[int, List[int]]],  # graph[layer][node] -> neighbor list
    extend_candidates: bool = True,
    keep_pruned_connections: bool = True,
) -> List[int]:
    """
    Algorithm 4: heuristic neighbor selection that tries to diversify
    the neighborhood - each selected neighbor should be closer to q than
    to any already-selected neighbor.

    Args:
        q                      : query element index
        candidates             : initial set of candidates
        M                      : number of neighbors to return
        layer                  : current graph layer
        data                   : all element vectors
        graph                  : adjacency lists per layer
        extend_candidates      : if True, extend candidates with their neighbors
        keep_pruned_connections: if True, fill up to M with pruned candidates

    Returns:
        List of up to M element indices.
    """
    R: List[int] = []  # result set
    W: List[Tuple[float, int]] = []  # working candidates (min-heap by dist)

    # Build initial working set
    for c in candidates:
        d = euclidean_distance(data[q], data[c])
        heapq.heappush(W, (d, c))

    # (Optional) extend candidates with their layer-lc neighbors
    if extend_candidates:
        extra: Set[int] = set()
        for c in candidates:
            for neighbor in graph.get(layer, {}).get(c, []):
                if neighbor not in candidates:
                    extra.add(neighbor)
        for e_adj in extra:
            d = euclidean_distance(data[q], data[e_adj])
            heapq.heappush(W, (d, e_adj))

    W_d: List[Tuple[float, int]] = []  # discarded candidates

    while W and len(R) < M:
        dist_e, e = heapq.heappop(W)  # nearest in W to q

        # If e is closer to q than to every element already in R → keep it
        closer_to_q = True
        for r in R:
            if euclidean_distance(data[e], data[r]) < dist_e:
                closer_to_q = False
                break

        if closer_to_q:
            R.append(e)
        else:
            W_d.append((dist_e, e))

    # (Optional) fill up remaining slots with pruned connections
    if keep_pruned_connections:
        W_d.sort(key=lambda x: x[0])
        for dist_e, e in W_d:
            if len(R) >= M:
                break
            R.append(e)

    return R


# ---------------------------------------------------------------------------
# Algorithm 2 – SEARCH-LAYER
# ---------------------------------------------------------------------------


def search_layer(
    q_vec: np.ndarray,
    entry_points: List[int],
    ef: int,
    layer: int,
    data: np.ndarray,
    graph: Dict[int, Dict[int, List[int]]],
) -> List[Tuple[float, int]]:
    """
    Algorithm 2: greedy beam-search on a single graph layer.

    Args:
        q_vec        : query vector (numpy array)
        entry_points : starting element indices
        ef           : size of the dynamic candidate list (beam width)
        layer        : which layer to search
        data         : all element vectors
        graph        : adjacency lists per layer

    Returns:
        List of (distance, element_id) for the ef nearest found neighbors,
        sorted nearest-first.
    """
    v: Set[int] = set(entry_points)  # visited nodes

    # C – candidates (min-heap, nearest at top)
    C: List[Tuple[float, int]] = []
    # W – found nearest neighbors (max-heap, furthest at top so we can prune)
    W: List[Tuple[float, int]] = []  # stored as (-dist, elem)

    for ep in entry_points:
        d = euclidean_distance(q_vec, data[ep])
        heapq.heappush(C, (d, ep))
        heapq.heappush(W, (-d, ep))  # max-heap trick

    while C:
        dist_c, c = heapq.heappop(C)  # nearest candidate to q

        # f = furthest element in W
        neg_dist_f, _ = W[0]
        dist_f = -neg_dist_f

        # If nearest candidate is farther than worst result → stop
        if dist_c > dist_f:
            break

        # Explore neighbors of c on this layer
        for e in graph.get(layer, {}).get(c, []):
            if e not in v:
                v.add(e)
                dist_e = euclidean_distance(q_vec, data[e])

                neg_dist_f2, _ = W[0]
                dist_f2 = -neg_dist_f2

                # Add e to C and W if it improves W or W is not full yet
                if dist_e < dist_f2 or len(W) < ef:
                    heapq.heappush(C, (dist_e, e))
                    heapq.heappush(W, (-dist_e, e))

                    # Keep W trimmed to ef elements
                    if len(W) > ef:
                        heapq.heappop(W)  # removes the furthest (max-heap)

    # Convert W back to (dist, elem) sorted nearest-first
    result = [(-neg_d, e) for neg_d, e in W]
    result.sort(key=lambda x: x[0])
    return result


# ---------------------------------------------------------------------------
# Main HNSW class  (Algorithm 1 = INSERT, Algorithm 5 = K-NN-SEARCH)
# ---------------------------------------------------------------------------


class HNSW:
    """
    Hierarchical Navigable Small World graph for approximate nearest
    neighbor search.

    Parameters
    ----------
    M              : number of established connections per inserted element
                     (neighbors per layer, except layer 0 which uses M_max0)
    ef_construction: size of the dynamic candidate list during construction
    M_max          : max connections per element per layer  (defaults to M)
    M_max0         : max connections per element at layer 0 (defaults to 2*M)
    m_L            : level normalization factor              (defaults to 1/ln(M))
    use_heuristic  : if True use Algorithm 4, else Algorithm 3
    """

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        M_max: Optional[int] = None,
        M_max0: Optional[int] = None,
        m_L: Optional[float] = None,
        use_heuristic: bool = True,
    ):
        self.M = M
        self.ef_construction = ef_construction
        self.M_max = M_max if M_max is not None else M
        self.M_max0 = M_max0 if M_max0 is not None else 2 * M
        self.m_L = m_L if m_L is not None else 1.0 / math.log(M)
        self.use_heuristic = use_heuristic

        # Storage
        self.data: List[np.ndarray] = []  # element vectors
        self.graph: Dict[int, Dict[int, List[int]]] = defaultdict(dict)
        # graph[layer][node] = [neighbors]
        self.enter_point: Optional[int] = None  # global entry point
        self.max_layer: int = -1  # top occupied layer (L)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_level(self) -> int:
        """Sample a random layer level for a new element (paper eq. 1)."""
        # l = floor(-ln(unif(0,1)) * m_L)
        return int(-math.log(np.random.uniform()) * self.m_L)

    def _select_neighbors(
        self,
        q: int,
        candidates: Set[int],
        M: int,
        layer: int,
    ) -> List[int]:
        """Route to Algorithm 3 or 4 depending on self.use_heuristic."""
        if self.use_heuristic:
            return select_neighbors_heuristic(
                q,
                candidates,
                M,
                layer,
                np.array(self.data),
                self.graph,
            )
        else:
            return select_neighbors_simple(
                q,
                candidates,
                M,
                np.array(self.data),
            )

    def _add_connection(self, layer: int, u: int, v: int):
        """Add undirected edge u-v on given layer."""
        if u not in self.graph[layer]:
            self.graph[layer][u] = []
        if v not in self.graph[layer]:
            self.graph[layer][v] = []
        if v not in self.graph[layer][u]:
            self.graph[layer][u].append(v)
        if u not in self.graph[layer][v]:
            self.graph[layer][v].append(u)

    # ------------------------------------------------------------------
    # Algorithm 1 – INSERT
    # ------------------------------------------------------------------

    def insert(self, q_vec: np.ndarray):
        """
        Algorithm 1: insert a new element into the HNSW graph.

        Args:
            q_vec : vector of the new element (numpy array)
        """
        # ---- Bookkeeping ----
        q = len(self.data)  # new element's integer id
        self.data.append(q_vec.copy())
        data_arr = np.array(self.data)  # snapshot for distance calls

        W: List[Tuple[float, int]] = []  # currently found nearest elements
        ep = self.enter_point  # global entry point
        L = self.max_layer  # top layer

        l = self._get_level()  # new element's assigned level

        # ---- Phase 1: greedily descend from L down to l+1 ----
        #      (ef=1: we just track the single nearest neighbor)
        if ep is not None:
            for l_c in range(L, l, -1):
                W = search_layer(
                    q_vec, [ep], ef=1, layer=l_c, data=data_arr, graph=self.graph
                )
                ep = W[0][1]  # nearest element found

            # ---- Phase 2: insert connections from layer min(L,l) down to 0 ----
            for l_c in range(min(L, l), -1, -1):
                W = search_layer(
                    q_vec,
                    [ep],
                    ef=self.ef_construction,
                    layer=l_c,
                    data=data_arr,
                    graph=self.graph,
                )

                candidates = {e for _, e in W}

                M_limit = self.M_max0 if l_c == 0 else self.M_max
                neighbors = self._select_neighbors(q, candidates, self.M, l_c)

                # Add bidirectional connections
                for nb in neighbors:
                    self._add_connection(l_c, q, nb)

                # Shrink connections if any neighbor exceeds M_max
                for nb in neighbors:
                    nb_conn = self.graph[l_c].get(nb, [])
                    if len(nb_conn) > M_limit:
                        nb_candidates = set(nb_conn)
                        new_conn = self._select_neighbors(
                            nb, nb_candidates, M_limit, l_c
                        )
                        self.graph[l_c][nb] = new_conn

                # Move entry point to nearest found in this layer
                ep = W[0][1]

        # ---- Update global entry point if new level is higher ----
        if l > L:
            self.max_layer = l
            self.enter_point = q

            # Ensure graph dicts exist for new layers
            for l_c in range(L + 1, l + 1):
                self.graph[l_c][q] = []

    # ------------------------------------------------------------------
    # Algorithm 5 – K-NN-SEARCH
    # ------------------------------------------------------------------

    def search(
        self, q_vec: np.ndarray, K: int, ef: int = 50
    ) -> List[Tuple[float, int]]:
        """
        Algorithm 5: find the K approximate nearest neighbors of q_vec.

        Args:
            q_vec : query vector
            K     : number of nearest neighbors to return
            ef    : size of dynamic candidate list (≥ K for best recall)

        Returns:
            List of (distance, element_id) sorted nearest-first, length ≤ K.
        """
        if self.enter_point is None:
            return []

        data_arr = np.array(self.data)
        W: List[Tuple[float, int]] = []
        ep = self.enter_point
        L = self.max_layer

        # Greedy descent to layer 1 with ef=1
        for l_c in range(L, 0, -1):
            W = search_layer(
                q_vec, [ep], ef=1, layer=l_c, data=data_arr, graph=self.graph
            )
            ep = W[0][1]

        # Full beam search at layer 0
        W = search_layer(q_vec, [ep], ef=ef, layer=0, data=data_arr, graph=self.graph)

        # Return the K nearest
        W.sort(key=lambda x: x[0])
        return W[:K]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f"HNSW(n={len(self)}, M={self.M}, "
            f"ef_construction={self.ef_construction}, "
            f"layers={self.max_layer + 1})"
        )


# ---------------------------------------------------------------------------
# Demo / sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    DIM = 128  # vector dimension
    N = 1000  # number of elements to index
    K = 10  # nearest neighbors to retrieve
    EF = 32  # search ef

    print("=" * 60)
    print("HNSW Demo")
    print("=" * 60)
    print(f"  Indexing {N} random {DIM}-d vectors …")

    # Build the index
    index = HNSW(M=32, ef_construction=32, use_heuristic=True)
    vectors = np.random.rand(N, DIM).astype(np.float32)
    for i, vec in enumerate(vectors):
        index.insert(vec)
        if (i + 1) % 500 == 0:
            print(f"    inserted {i + 1}/{N}")

    print(f"\n  Index built: {index}")
    print(f"  Layers: 0 … {index.max_layer}")

    # Query
    q_vec = np.random.rand(DIM).astype(np.float32)
    print(f"\n  Querying for {K}-NN …")
    results = index.search(q_vec, K=K, ef=EF)

    print(f"\n  Top-{K} approximate nearest neighbors:")
    for rank, (dist, elem_id) in enumerate(results, 1):
        print(f"    #{rank:2d}  id={elem_id:5d}  dist={dist:.6f}")

    # Brute-force ground truth for recall evaluation
    print("\n  Computing brute-force ground truth …")
    all_dists = np.linalg.norm(vectors - q_vec, axis=1)
    gt_ids = np.argsort(all_dists)[:K]
    gt_set = set(gt_ids.tolist())

    approx_set = {elem_id for _, elem_id in results}
    recall = len(approx_set & gt_set) / K
    print(f"  Recall@{K} = {recall:.2%}")

    print("\n  Brute-force ground truth top-10:")
    for rank, idx in enumerate(gt_ids, 1):
        marker = "✓" if idx in approx_set else "✗"
        print(f"    #{rank:2d}  id={idx:5d}  dist={all_dists[idx]:.6f}  {marker}")

    print("\nDone.")
