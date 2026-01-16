from sentence_transformers import SentenceTransformer

# 1. Tải mô hình AI (Lần chạy đầu tiên sẽ hơi lâu một chút để tải model về)
print("--- Đang tải mô hình AI... ---")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Ví dụ 2 câu văn bản cần máy hiểu
sentences = ["Hôm nay trời đẹp", "Tôi đang học lập trình AI"]

# 3. Chuyển đổi văn bản thành số (Vector)
embeddings = model.encode(sentences)

# 4. In kết quả
print("Thành công!")
print(f"Máy đã biến 2 câu văn thành vector có kích thước: {embeddings.shape}")
print("Vector của câu đầu tiên trông như thế này (rút gọn):", embeddings[0][:5])