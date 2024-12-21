# Xây Dựng Mạng Nơ-Ron Đa Lớp Xử Lý Bài Toán XOR

## Giới thiệu
Đồ án này tập trung vào việc nghiên cứu và xây dựng một mạng nơ-ron đa lớp để giải quyết bài toán XOR phi tuyến tính, sử dụng ngôn ngữ Python. Đây là một bài toán kinh điển trong lĩnh vực trí tuệ nhân tạo, giúp củng cố các kiến thức cơ bản về mạng nơ-ron nhân tạo và các thuật toán tối ưu hóa.

### Mục tiêu nghiên cứu
- Tìm hiểu các khái niệm và cấu trúc của mạng nơ-ron nhân tạo.
- Xây dựng và huấn luyện mạng nơ-ron đa lớp để xử lý bài toán XOR.
- Ứng dụng thuật toán lan truyền ngược để tối ưu hóa mô hình.
- Hiểu rõ các vấn đề như hội tụ, overfitting, và cải thiện hiệu suất mô hình.

## Nội dung chính
### 1. Mô hình mạng nơ-ron
- **Lớp đầu vào**: 2 đầu vào.
- **Lớp ẩn**: 2 nút với hàm kích hoạt Sigmoid.
- **Lớp đầu ra**: 1 nút với hàm kích hoạt Sigmoid.

### 2. Các bước thực hiện
#### **Lan truyền tiến (Forward Propagation)**
- Tính toán giá trị đầu ra từng lớp dựa trên hàm kích hoạt Sigmoid.

#### **Hàm mất mát (Loss Function)**
- Sử dụng Binary Cross-Entropy để đánh giá độ lệch giữa đầu ra dự đoán và đầu ra thực tế.

#### **Lan truyền ngược (Backward Propagation)**
- Sử dụng chuỗi đạo hàm (chain rule) để cập nhật trọng số và bias, tối ưu hóa mô hình.

#### **Cập nhật trọng số**
- Áp dụng thuật toán Gradient Descent với learning rate cố định để cải thiện hiệu suất.

### 3. Kết quả thực nghiệm
- Hàm mất mát giảm đều qua các epoch và đạt trạng thái ổn định sau khoảng 8000 epoch.
- Mô hình có khả năng phân loại chính xác dữ liệu XOR với độ chính xác cao.

### 4. Hướng phát triển
- Nâng cấp mô hình với mạng sâu hơn (Deep Neural Network), hoặc thử nghiệm với kiến trúc khác như CNN, RNN.
- Áp dụng các thuật toán tối ưu hiện đại như Adam, RMSProp.
- Mở rộng mô hình sang các bài toán thực tế trong xử lý ngôn ngữ tự nhiên (NLP), nhận diện hình ảnh, và dự đoán tài chính.

## Cách chạy chương trình
1. **Cài đặt môi trường**:
   - Python >= 3.8
   - Các thư viện: NumPy

2. **Chạy chương trình**:
   - Thực hiện lệnh:
     ```bash
     python main.py
     ```
   - Kết quả sẽ hiển thị giá trị hàm mất mát và dự đoán cuối cùng.

3. **Tùy chỉnh tham số**:
   - Có thể thay đổi learning rate, số epoch hoặc cấu trúc mạng trong file `main.py`.

## Tài liệu tham khảo
- [Machine Learning Cơ Bản](https://machinelearningcoban.com/2017/02/24/mlp/#-vi-du-tren-python)
- [NTTuan8 Backpropagation](https://nttuan8.com/bai-4-backpropagation/)
