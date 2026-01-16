from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Khởi tạo model AI
print("Đang khởi động model AI")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Test thử biến chữ thành số
van_ban = ["Tôi thích học lập trình", "Code là đam mê của tôi"]
vectors = model.encode(van_ban)

print(f"\nKích thước Vector: {vectors.shape[1]} chiều")

# 3. Test độ tương đồng giữa 2 câu
similarity = cosine_similarity([vectors[0]], [vectors[1]])
print(f"Câu 1: {van_ban[0]}")
print(f"Câu 2: {van_ban[1]}")
print(f"Độ giống nhau: {similarity[0][0] * 100:.2f}%")

# 4. Lưu lại
np.save('vectors.npy', vectors)
print("\nĐã lưu file vectors.npy.")