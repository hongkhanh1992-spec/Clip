# Phát thanh Măng non – "Em yêu văn hóa Việt Nam – Tự hào bản sắc dân tộc"

Video có lời đọc hưởng ứng Ngày Văn hóa Việt Nam 24/11.

- Video: `output/phat_thanh_mang_non_van_hoa_viet_nam.mp4` (1920×1080, ~5 phút)
- Phụ đề rời: `output/phu_de.srt`
- Kịch bản lời đọc (sửa được): `script.txt`

## Dựng lại video
```
pip install piper-tts numpy soundfile pillow
# tải giọng đọc tiếng Việt (vi_VN-vais1000-medium) và giải nén vào thư mục voice/
curl -L https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-vi_VN-vais1000-medium.tar.bz2 | tar xj
mv vits-piper-vi_VN-vais1000-medium voice
python3 build.py
```
Giọng đọc là giọng nữ tổng hợp (offline); nhạc nền do chương trình tự tạo (không dùng bài hát có bản quyền).
