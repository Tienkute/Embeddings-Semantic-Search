from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Khởi động AI
print("--- Đang tải dữ liệu... ---")
# Dùng model hỗ trợ nhiều ngôn ngữ (bao gồm Tiếng Việt)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# 2. Tạo một kho dữ liệu giả lập (Ví dụ bạn có các tài liệu này)
documents = [
    "Cách cài đặt Python trên Windows rất dễ",   # Doc 0
    "Công thức nấu món phở bò Nam Định chuẩn vị", # Doc 1
    "Học máy (Machine Learning) là tương lai của công nghệ", # Doc 2
    "Du lịch Đà Nẵng nên đi Bà Nà Hills", # Doc 3
]

# 3. Học dữ liệu (Biến tất cả văn bản thành Vector)
doc_vectors = model.encode(documents)

# 4. Người dùng đặt câu hỏi
query = "Làm thế nào để lập trình?"  # <--- Bạn có thể đổi câu hỏi ở đây
print(f"\nCâu hỏi của bạn: {query}")

# 5. Tìm kiếm (So sánh vector câu hỏi với kho dữ liệu)
query_vector = model.encode([query])
scores = cosine_similarity(query_vector, doc_vectors)[0]

# 6. Lấy kết quả tốt nhất
best_idx = np.argmax(scores) # Tìm vị trí có điểm cao nhất
best_score = scores[best_idx]

print(f"--> Kết quả tìm thấy: \"{documents[best_idx]}\"")
print(f"--> Độ chính xác: {best_score:.4f} (Càng gần 1 càng chuẩn)")