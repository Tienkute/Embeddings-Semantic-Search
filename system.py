from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- PHẦN 1: BỘ NÃO (MEMORY & ROUTING) ---
class SearchBrain:
    def __init__(self):
        self.history = [] # Nơi lưu lịch sử câu hỏi

    def remember(self, text):
        self.history.append(text)
        # In ra lịch sử để kiểm tra
        print(f"[Memory] Đã nhớ {len(self.history)} câu hỏi trong phiên này.")

    def decide(self, text):
        # LOGIC ĐIỀU HƯỚNG:
        # Nếu câu hỏi dài hơn 7 từ -> Coi là phức tạp -> Chuyển sang chế độ Giải thích
        # Nếu ngắn hơn -> Tìm kiếm ngay
        word_count = len(text.split())
        if word_count > 7:
            return "EXPLAIN"
        return "SEARCH"

# --- PHẦN 2: KHỞI TẠO HỆ THỐNG ---
print("--- Đang khởi động hệ thống... (Vui lòng đợi 1 chút) ---")
# Dùng model tiếng Việt xịn mà bạn vừa tải
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
brain = SearchBrain()

# Kho dữ liệu giả lập
docs = [
    "Hướng dẫn cài đặt Python cho người mới bắt đầu",
    "Phở bò Nam Định là món ngon nổi tiếng",
    "Trí tuệ nhân tạo (AI) đang thay đổi thế giới",
    "Đà Nẵng có cầu Rồng và Bà Nà Hills",
    "Lập trình viên cần học tiếng Anh"
]
doc_vectors = model.encode(docs)

# --- PHẦN 3: VÒNG LẶP CHAT (INTERACTIVE LOOP) ---
print("\n" + "="*40)
print("  HỆ THỐNG TÌM KIẾM SẴN SÀNG!")
print("  Gõ 'exit' để thoát chương trình.")
print("="*40 + "\n")

while True:
    # Nhập câu hỏi từ bàn phím
    query = input("Bạn (Gõ câu hỏi): ")
    
    if query.lower() == 'exit':
        print("Tạm biệt!")
        break
    
    # 1. Ghi nhớ
    brain.remember(query)
    
    # 2. Phân loại (Routing)
    action = brain.decide(query)
    
    if action == "EXPLAIN":
        print(f">> AI: Câu hỏi này khá dài ({len(query.split())} từ). Bạn cần tôi giải thích chi tiết hay tóm tắt?")
    
    else: # action == "SEARCH"
        # 3. Tìm kiếm
        vec = model.encode([query])
        scores = cosine_similarity(vec, doc_vectors)[0]
        idx = np.argmax(scores)
        
        print(f">> AI tìm thấy: \"{docs[idx]}\"")
        print(f"   (Độ tin cậy: {scores[idx]:.4f})\n")