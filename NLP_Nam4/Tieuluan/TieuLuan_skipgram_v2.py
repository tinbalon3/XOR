import numpy as np
import re
import string
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize

from pyvi import ViTokenizer
from tqdm import tqdm
import pickle


# === 1. TIỀN XỬ LÝ DỮ LIỆU ===
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text


with open('dataset/demo-title.txt', 'r', encoding='utf-8') as file:
    text_corpus = file.read()



# Tokenize và loại bỏ stopwords


def preprocess_text(corpus):
    sentences = corpus.split('\n')
    # Kết quả chứa các mảng từ của từng câu
    processed_sentences = []
    with open("stopword/stopwords-vi.txt", "r", encoding="utf-8") as f:
        stop_words = set(f.read().splitlines())
    for sentence in sentences:
        if sentence.strip():  # Bỏ qua các dòng trống
            # Chuyển thành chữ thường và loại bỏ dấu câu
            cleaned_sentence = re.sub(r'[^\w\s]', '', sentence.lower())
            # Tách từ bằng ViTokenizer
            tokens = ViTokenizer.tokenize(cleaned_sentence).split()
            # Loại bỏ stopwords
            filtered_tokens = [word for word in tokens if word not in stop_words]
            processed_sentences.append(filtered_tokens)

    return processed_sentences


# Tiền xử lý văn bản
# sentences = tokenize_text(text_corpus)
sentences = preprocess_text(text_corpus)
print(f"Số câu: {len(sentences)}")


def build_vocab(sentences):
    """
    Tạo từ điển từ vựng giữ nguyên thứ tự xuất hiện và không loại bỏ từ trùng lặp.

    :param sentences: Danh sách các câu, mỗi câu là danh sách các từ.
    :return: Từ điển từ vựng và từ điển đảo ngược.
    """
    vocab = {}
    reverse_vocab = {}
    idx = 0

    for sentence in sentences:
        for word in sentence:
            if word not in vocab:  # Chỉ thêm từ chưa có trong từ điển
                vocab[word] = idx  # Mỗi từ được ánh xạ với chỉ số duy nhất
                reverse_vocab[idx] = word  # Đảo ngược ánh xạ chỉ số về từ
                idx += 1

    return vocab, reverse_vocab

# Tạo từ điển từ vựng
# Gọi hàm
vocabulary, reverse_vocab = build_vocab(sentences)
print(vocabulary)
# Kết quả

vocab_size = len(vocabulary)
print(f"Kích thước từ vựng: {vocab_size}")

# === 2. TẠO DỮ LIỆU HUẤN LUYỆN ===

# Tạo cặp (từ trung tâm, từ ngữ cảnh)

def generate_training_data(sentences, window_size=2, vocabulary=None):
    """
    Tạo dữ liệu huấn luyện dưới dạng cặp chỉ số (center_word_idx, context_word_idx).

    :param sentences: Danh sách các câu, mỗi câu là danh sách các từ.
    :param window_size: Kích thước cửa sổ ngữ cảnh.
    :param vocabulary: Từ điển ánh xạ từ thành chỉ số.
    :return: Danh sách các cặp (center_word_idx, context_word_idx).
    """
    training_data = []
    for sentence in sentences:
        for idx, word in enumerate(sentence):
            for neighbor in range(-window_size, window_size + 1):
                if neighbor == 0 or idx + neighbor < 0 or idx + neighbor >= len(sentence):
                    continue
                center_word = word
                context_word = sentence[idx + neighbor]
                # Chuyển đổi từ sang chỉ số nếu chúng nằm trong từ điển
                if center_word in vocabulary and context_word in vocabulary:
                    training_data.append((vocabulary[center_word], vocabulary[context_word]))
    return training_data

training_data = generate_training_data(sentences, window_size=2,vocabulary=vocabulary)

print(f"Tổng số cặp (từ trung tâm, từ ngữ cảnh): {len(training_data)}")

# === 3. MÔ HÌNH SKIP-GRAM ===

# Hàm softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Hàm huấn luyện Skip-gram
class SkipGramModel:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W1 = np.random.rand(vocab_size, embedding_dim)
        self.W2 = np.random.rand(embedding_dim, vocab_size)

    def forward(self, center_word_idx):
        h = self.W1[center_word_idx]  # Hidden layer
        u = np.dot(h, self.W2)  # Output layer
        y_pred = softmax(u)
        return y_pred, h, u

    def backward(self, error, h, center_word_idx, learning_rate):
        dW2 = np.outer(h, error)
        dW1 = np.dot(self.W2, error)
        self.W1[center_word_idx] -= learning_rate * dW1
        self.W2 -= learning_rate * dW2

    def train(self, training_data, epochs, learning_rate):
        loss_history = []
        for epoch in range(epochs):
            total_loss = 0
            for center_word_idx, context_word_idx in tqdm(training_data):
                y_pred, h, u = self.forward(center_word_idx)
                error = y_pred.copy()
                error[context_word_idx] -= 1
                self.backward(error, h, center_word_idx, learning_rate)
                total_loss += -np.log(y_pred[context_word_idx])
            loss_history.append(total_loss / len(training_data))
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(training_data)}")
        return loss_history



# === 5. ĐÁNH GIÁ EMBEDDING VECTOR ===

def get_vector(word):
    word_idx = vocabulary[word]
    return skip_gram.W1[word_idx]

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# == 6.  MÔ HÌNH
# Lưu mô hình
def save_model(embedding_matrix, word_to_index, index_to_word,loss_history, file_path):
    model_data = {
        "embedding_matrix": embedding_matrix,
        "word_to_index": word_to_index,
        "index_to_word": index_to_word,
        "loss_history": loss_history
    }
    with open(file_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {file_path}")

# Tải mô hình
def load_model(file_path):
    with open(file_path, "rb") as f:
        model_data = pickle.load(f)
    print(f"Model loaded from {file_path}")
    return model_data["embedding_matrix"], model_data["word_to_index"], model_data["index_to_word"], model_data["loss_history"]


# === 4. HUẤN LUYỆN MÔ HÌNH ===

embedding_dim = 100
skip_gram = SkipGramModel(vocab_size, embedding_dim)
loss_history = skip_gram.train(training_data, epochs=100, learning_rate=0.01)
# Lưu mô hình sau khi huấn luyện
save_model(skip_gram.W1, vocabulary, reverse_vocab,loss_history, "skipgram_model_demofull_v1.pkl")
embedding_matrix, word_to_index, index_to_word, loss_history = load_model("skipgram_model_demofull_v2.pkl")

# # Vẽ biểu đồ mất mát
plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss model_v2")
plt.show()



plt.figure(figsize=(10, 6))
plt.bar(range(1, len(loss_history) + 1), loss_history, alpha=0.5, label="Loss per Epoch")
plt.plot(range(1, len(loss_history) + 1), loss_history, label="Loss Line", color="red", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss model_v2")
plt.legend()
plt.grid(True)
plt.show()


# Đánh giá tương đồng từ vựng
# Tải mô hình đã lưu


# Sử dụng embedding_matrix thay vì skip_gram.W1
def get_vector_loaded_model(word):
    if word in word_to_index:
        word_idx = word_to_index[word]
        return embedding_matrix[word_idx]
    else:
        print(f"Word '{word}' not found in vocabulary.")
        return None
if __name__ == '__main__':

    # word_pairs_synonyms = [ ("học", "giáo_dục") , ("phạm_luật","vi_phạm"),("xử_lý","giải_quyết"),("kiểm_tra","xác_minh")]
    # word_pairs_antonyms = [("chấp_hành","vi_phạm"),("tăng","giảm"),("đất","nước"),("đồng_ý","phản_đối")]
    # similarities = []
    # for word1, word2 in word_pairs_antonyms:
    #     vec1 = get_vector_loaded_model(word1)
    #     vec2 = get_vector_loaded_model(word2)
    #     if vec1 is not None and vec2 is not None:
    #         similarity = cosine_similarity(vec1, vec2)
    #         similarities.append((word1, word2, similarity))
    #         print(f"Cosine similarity giữa '{word1}' và '{word2}': {similarity}")
    # # Vẽ biểu đồ cosine similarity
    # import matplotlib.pyplot as plt
    # #
    # words = [f"{word1} ↔ {word2}" for word1, word2, _ in similarities]
    # scores = [sim for _, _, sim in similarities]
    #
    # plt.figure(figsize=(10, 6))
    # plt.barh(words, scores)
    # plt.xlabel("Cosine Similarity")
    # plt.title("Cosine Similarity for Synonyms and Antonyms")
    # plt.tight_layout()  # Tự động căn chỉnh mọi thành phần
    # plt.show()


    # Load mô hình v1 và v2
    model_v1_path = "skipgram_model_demofull_v1.pkl"
    model_v2_path = "skipgram_model_demofull_v2.pkl"

    embedding_matrix_v1, word_to_index_v1, index_to_word_v1, loss_history_v1 = load_model(model_v1_path)
    embedding_matrix_v2, word_to_index_v2, index_to_word_v2, loss_history_v2 = load_model(model_v2_path)

    # === 1. So sánh lịch sử mất mát ===
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history_v1) + 1), loss_history_v1, label="Model v1")
    plt.plot(range(1, len(loss_history_v2) + 1), loss_history_v2, label="Model v2")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Comparison Between Models v1 and v2")
    plt.legend()
    plt.grid(True)
    plt.show()

    # === 2. So sánh độ tương đồng giữa các từ ===

    # word_pairs = [("đất", "nước"), ("học", "giáo_dục"), ("tăng", "giảm"), ("đồng_ý", "phản_đối")]


    word_pairs_synonyms = [ ("học", "giáo_dục") , ("phạm_luật","vi_phạm"),("xử_lý","giải_quyết"),("kiểm_tra","xác_minh")]
    word_pairs_antonyms = [("học","vi_phạm"),("tăng","giảm"),("đất","nước"),("đồng_ý","phản_đối")]
    def calculate_similarity(embedding_matrix, word_to_index, word_pairs):
        similarities = []
        for word1, word2 in word_pairs:
            vec1 = embedding_matrix[word_to_index[word1]] if word1 in word_to_index else None
            vec2 = embedding_matrix[word_to_index[word2]] if word2 in word_to_index else None
            if vec1 is not None and vec2 is not None:
                similarity = cosine_similarity(vec1, vec2)
                similarities.append((word1, word2, similarity))
            else:
                similarities.append((word1, word2, None))  # Trường hợp không tìm thấy từ
        return similarities

    similarities_v1 = calculate_similarity(embedding_matrix_v1, word_to_index_v1, word_pairs_antonyms)
    similarities_v2 = calculate_similarity(embedding_matrix_v2, word_to_index_v2, word_pairs_antonyms)

    # In ra kết quả
    print("Cosine Similarity - Model v1:")
    for word1, word2, sim in similarities_v1:
        print(f"{word1} ↔ {word2}: {sim if sim is not None else 'Not found in vocabulary'}")

    print("\nCosine Similarity - Model v2:")
    for word1, word2, sim in similarities_v2:
        print(f"{word1} ↔ {word2}: {sim if sim is not None else 'Not found in vocabulary'}")

    # Vẽ biểu đồ so sánh cosine similarity giữa v1 và v2
    words = [f"{word1} ↔ {word2}" for word1, word2, _ in similarities_v1]
    scores_v1 = [sim if sim is not None else 0 for _, _, sim in similarities_v1]
    scores_v2 = [sim if sim is not None else 0 for _, _, sim in similarities_v2]

    x = range(len(words))  # Chỉ số cho từng cặp từ

    plt.figure(figsize=(10, 6))
    plt.bar(x, scores_v1, width=0.4, label="Model v1", align="center", alpha=0.7)
    plt.bar([i + 0.4 for i in x], scores_v2, width=0.4, label="Model v2", align="center", alpha=0.7)
    plt.xticks([i + 0.2 for i in x], words, rotation=45, ha="right")
    plt.xlabel("Word Pairs")
    plt.ylabel("Cosine Similarity")
    plt.title("Comparison of Cosine Similarity Between Models v1 and v2")
    plt.legend()
    plt.tight_layout()
    plt.show()



model_v1_path = "skipgram_model_demofull_v1.pkl"
model_v2_path = "skipgram_model_demofull_v2.pkl"

embedding_matrix_v1, word_to_index_v1, index_to_word_v1, loss_history_v1 = load_model(model_v1_path)
embedding_matrix_v2, word_to_index_v2, index_to_word_v2, loss_history_v2 = load_model(model_v2_path)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_history_v1) + 1), loss_history_v1, label="Model v1")
plt.plot(range(1, len(loss_history_v2) + 1), loss_history_v2, label="Model v2")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Comparison Between Models v1 and v2")
plt.legend()
plt.grid(True)
plt.show()




