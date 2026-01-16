
### 1. Embedding Foundation (Task 1)
* **Neural Encoder:** Utilizes the `all-MiniLM-L6-v2` transformer model for high-speed, high-accuracy vectorization.
* **Vector Transformation:** Converts raw string inputs into 384-dimensional dense vectors.
* **Storage & Management:** Implements persistent storage using `NumPy` (`.npy` files) to track embedding dimensions and avoid redundant computations.

### 2. Semantic Search Engine (Task 2)
* **Document Indexing:** Capable of storing and retrieving information from a multi-document local repository.
* **Mathematical Matching:** Uses **Cosine Similarity** to calculate the semantic "distance" between queries and stored data.
* **Formula:**
  $$\text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

### 3. Memory & Intelligent Routing (Task 3 & 4)
* **Session Memory:** Implements a tracking system to store query history with timestamps for user behavior analysis.
* **Topic Summarization:** Prepares data for summarizing frequently accessed knowledge points.
* **Dynamic Routing Logic:**
    * **Short Queries (< 5 words):** Optimized for direct, high-speed semantic retrieval.
    * **Long Queries (>= 5 words):** Routed through a refinement and explanation step for deeper contextual understanding.



---

## Technical Comparison

| Feature | Legacy Keyword Search | Our Semantic System |
| :--- | :--- | :--- |
| **Logic** | Exact string matching | Deep semantic understanding |
| **Synonyms** | Fails (e.g., "PC" vs "Computer") | Succeeds (identifies same intent) |
| **Context** | Ignores word order/intent | Captures linguistic nuances |

