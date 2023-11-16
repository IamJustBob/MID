# MID
For school stem project, MID is a text classifier model with LSTM architecture

Đây là một dự án khoa học rất thú vị và có ý nghĩa. Tôi sẽ giúp bạn viết tệp README cho dự án của bạn bằng tiếng Việt. Bạn có thể sử dụng mẫu sau đây:

# MID - Máy dò bệnh tâm thần
MID là một phần mềm mã nguồn mở có thể chẩn đoán 8 loại bệnh tâm thần chính (Trầm cảm, Lo âu, PTSD, Lo lắng quá mức, Rối loạn ám ảnh cưỡng chế, Kiệt sức, Tâm thần phân liệt, Mất ngủ) bằng cách sử dụng các đầu vào văn bản về cuộc sống, cảm xúc, khó khăn của bệnh nhân. MID sử dụng sức mạnh của 2 mô hình AI do tôi tự tạo từ đầu, MID-large11m và sub-MID-large3m. MID-large11m được huấn luyện trên bộ dữ liệu do tôi tự tạo từ đầu từ GPT sinh ra (khoảng 1000 mẫu) và các bài đăng Reddit (phần còn lại của các mẫu) và tổng cộng 4445 mẫu dữ liệu. Nó hoạt động tốt nhất với tiếng Việt nhưng cũng hoạt động với tiếng Anh. Điều kiện duy nhất là họ phải cung cấp đầu vào một cách nghiêm túc nhưng không phải là điều gì đó không liên quan đến sức khỏe tâm thần của họ hoặc điều gì đó ngẫu nhiên.

## Cài đặt
Để cài đặt MID, bạn cần có Python 3.11.6 và các thư viện sau, tất cả đều là phiên bản mới nhất:

```py
import tkinter as tk
from tkinter import messagebox
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import threading
from time import sleep
```

Bạn có thể cài đặt các thư viện bằng cách chạy lệnh sau trong terminal:

```terminal
pip install -r requirements.txt
```

Sau đó, bạn cần tải xuống các tệp mô hình AI từ [đây](https://drive.google.com/drive/folders/1joAOCPMctApg-C0a1Q1War5dUAUMxFFG?usp=sharing) và đưa mô hình sub-MID vào mục  ```Grader``` còn MID vào thư mục gốc của dự án.

## Sử dụng
Để chạy MID, bạn có thể mở tệp mã của tệp run.py, sửa đổi lời nhắc của bạn và chạy nó, hoặc nếu bạn muốn một giao diện người dùng, bạn có thể chạy tệp ui.py, chỉ cần đảm bảo bạn đã có các thư viện cần thiết.

Nếu bạn không có Python, MID.exe cũng có sẵn, nhưng chưa hỗ trợ trang web và điện thoại, nhưng dự kiến sẽ có mặt trên Android sớm vì tôi biết Kotlin!

## Đóng góp
MID là một dự án mã nguồn mở và tôi rất hoan nghênh sự đóng góp của cộng đồng. Nếu bạn có bất kỳ ý kiến, góp ý, báo lỗi hoặc yêu cầu tính năng nào, xin vui lòng tạo một vấn đề hoặc một yêu cầu kéo. Tôi sẽ cố gắng phản hồi sớm nhất có thể.

## Giấy phép
MID được phân phối theo giấy phép MIT. Xem tệp [LICENSE] để biết thêm chi tiết.

## Liên hệ
Nếu bạn có bất kỳ câu hỏi nào về dự án này, xin vui lòng liên hệ với tôi qua email: [vlinh2008ls@gmail.com]. Tôi rất mong nhận được phản hồi của bạn..
