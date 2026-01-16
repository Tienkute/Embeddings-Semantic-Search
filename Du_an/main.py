from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import datetime

class SemanticSearchSystem:
    def __init__(self):
        # 1. Khởi tạo Model (Embedding Basics)
        print("--- Đang khởi động hệ thống AI... ---")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.memory = [] # Lưu lịch sử tìm kiếm (Memory)
        
        # 2. Kho dữ liệu mẫu (Document Store)
        self.documents = [
            "Python là ngôn ngữ lập trình mạnh mẽ cho AI và Web.",
            "GitHub giúp quản lý mã nguồn và làm việc nhóm hiệu quả.",
            "Embedding biến văn bản thành các vector số để máy tính hiểu ngữ nghĩa.",
            "Hệ thống RAG kết hợp tìm kiếm dữ liệu và tạo văn bản tự động.",
            "Học máy (Machine Learning) là một tập con của Trí tuệ nhân tạo."
        ]
        # Chuyển kho dữ liệu thành Vector sẵn để tìm kiếm nhanh hơn
        self.doc_embeddings = self.model.encode(self.documents)

    def search(self, query):
        """Thực hiện tìm kiếm tương đồng (Similarity Search)"""
        query_vec = self.model.encode([query])
        # Tính toán Cosine Similarity
        scores = cosine_similarity(query_vec, self.doc_embeddings)[0]
        best_index = np.argmax(scores)
        return self.documents[best_index], scores[best_index]

    def process_query(self, user_query):
        """Điều hướng và xử lý câu hỏi (Memory + Routing)"""
        # Lưu vào lịch sử (Memory)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.memory.append({"time": timestamp, "query": user_query})

        # Logic điều hướng (Routing)
        words = user_query.split()
        
        if len(words) < 4:
            # Câu hỏi ngắn -> Tìm kiếm ngữ nghĩa trực tiếp
            print(f"\n[ROUTING]: Câu hỏi ngắn -> Chế độ Tìm kiếm nhanh")
            result, score = self.search(user_query)
            print(f"🔍 Kết quả: {result} (Độ khớp: {score*100:.2f}%)")
        else:
            # Câu hỏi dài -> Chế độ phân tích/giải thích
            print(f"\n[ROUTING]: Câu hỏi dài -> Chế độ Phân tích chuyên sâu")
            result, score = self.search(user_query)
            print(f"💡 Hệ thống đề xuất: {result}")
            print(f"📝 Giải thích: Dựa trên phân tích, câu hỏi '{user_query}' có liên quan mật thiết nhất đến nội dung này.")

    def show_history(self):
        """Hiển thị lịch sử (Summarize memory)"""
        print("\n--- LỊCH SỬ TÌM KIẾM ---")
        for item in self.memory:
            print(f"[{item['time']}] {item['query']}")

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    app = SemanticSearchSystem()
    
    # Test 1: Câu hỏi ngắn (Routing sang Search)
    app.process_query("Lập trình Python")
    
    # Test 2: Câu hỏi dài (Routing sang Refinement/Explanation)
    app.process_query("Làm thế nào để quản lý mã nguồn một cách hiệu quả nhất?")
    
    # Hiển thị lịch sử
    app.show_history()
