from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Khởi tạo Server
app = FastAPI()

# 1. Load Model (Chỉ load 1 lần khi server khởi động)
print("--- Đang tải model AI... ---")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. Kho dữ liệu
docs = [
    "Hướng dẫn cài đặt Python",
    "Phở bò Nam Định ngon tuyệt",
    "AI và Machine Learning là xu hướng",
    "Du lịch Đà Nẵng: Cầu Rồng, Bà Nà Hills",
    "Học tiếng Anh giao tiếp cấp tốc"
]
# Mã hóa dữ liệu sẵn để tìm cho nhanh
doc_vectors = model.encode(docs)

# 3. Định dạng dữ liệu đầu vào
class SearchQuery(BaseModel):
    text: str

# 4. Tạo đường dẫn tìm kiếm (API Endpoint)
@app.post("/search")
def search(item: SearchQuery):
    query = item.text
    
    # Logic tìm kiếm
    vec = model.encode([query])
    scores = cosine_similarity(vec, doc_vectors)[0]
    best_idx = int(np.argmax(scores))
    
    return {
        "status": "success",
        "question": query,
        "best_match": docs[best_idx],
        "score": float(scores[best_idx])
    }