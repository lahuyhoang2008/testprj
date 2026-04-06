"""
image_to_json.py - OCR đề thi THPT → questions.json
Hỗ trợ: PNG/JPG/WEBP, PDF, DOCX | Xử lý song song
Cài:  pip install google-genai pillow pymupdf docx2pdf
.env: GEMINI_API_KEY=your_key_here
Chạy: python image_to_json.py [file1 file2 ...]
       python image_to_json.py --pages p1.jpg p2.jpg p3.jpg   <- ghép ảnh riêng lẻ thành 1 đề
       python image_to_json.py --pages *.jpg --sort            <- tự sắp xếp theo số trong tên file
"""
import json, logging, os, re, shutil, sys, time, argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── .env ───────────────────────────────────────────────────────────────────────
for _p in [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]:
    if _p.exists():
        for _l in _p.read_text(encoding="utf-8").splitlines():
            if _l.strip() and not _l.startswith("#") and "=" in _l:
                k, _, v = _l.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        break

try:
    from google import genai; from google.genai import types
except ImportError: sys.exit("pip install google-genai")
try:
    from PIL import Image, ImageFilter, ImageEnhance
except ImportError: sys.exit("pip install pillow")

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL, MAX_TOKENS, WORKERS    = "gemma-4-31b-it", 32768, 10
RETRIES, RETRY_DELAY          = 4, 10.0
TIMEOUT                       = 180
# Rate limit: tối đa RATE_LIMIT request/batch, sau đó chờ RATE_WINDOW giây
# tính từ thời điểm request CUỐI CÙNG của batch trước.
RATE_LIMIT  = 15   # số request tối đa mỗi batch
RATE_WINDOW = 62   # giây phải chờ sau khi batch đầy (tính từ request cuối của batch)
CROP_PADDING        = 0.01
CROP_PADDING_BOTTOM = 0.01
CROP_MIN_PX  = 120
IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".gif": "image/gif",  ".webp": "image/webp", ".bmp": "image/bmp"}
SUBJECTS = {
    "toan":      ("Toán", True),      "vat_ly":    ("Vật lý", True),
    "hoa_hoc":   ("Hóa học", True),   "sinh_hoc":  ("Sinh học", False),
    "lich_su":   ("Lịch sử", False),  "dia_ly":    ("Địa lý", False),
    "gdcd":      ("GDCD/KTPL", False), "tieng_anh": ("Tiếng Anh", False),
    "tin_hoc":   ("Tin học", False),  "cong_nghe": ("Công nghệ", False),
}
SUBJECT_UNSUPPORTED = "ngu_van"

_KEYWORDS: dict[str, list[str]] = {
    "toan":      ["toán", "math", "đại số", "hình học", "giải tích", "xác suất", "thống kê"],
    "vat_ly":    ["vật lý", "vật lí", "physics", "lực từ", "điện", "quang học", "cơ học", "nhiệt", "sóng"],
    "hoa_hoc":   ["hóa học", "hoá học", "hóa", "chemistry", "nguyên tố", "phản ứng", "axit", "bazơ"],
    "sinh_hoc":  ["sinh học", "biology", "sinh", "di truyền", "tế bào", "tiến hóa", "sinh thái"],
    "lich_su":   ["lịch sử", "history"],
    "dia_ly":    ["địa lý", "địa lí", "geography", "khí hậu", "dân số"],
    "gdcd":      ["gdcd", "công dân", "kinh tế pháp luật", "pháp luật", "ktpl"],
    "tieng_anh": ["tiếng anh", "english", "anh văn"],
    "tin_hoc":   ["tin học", "informatics", "lập trình", "thuật toán"],
    "cong_nghe": ["công nghệ", "technology", "kỹ thuật"],
    "ngu_van":   ["ngữ văn", "văn học", "literature", "tập làm văn"],
}

# ── Rate limiter (batch window) ────────────────────────────────────────────────
# Mô hình: gửi tối đa RATE_LIMIT request một lúc (batch), sau đó chờ RATE_WINDOW
# giây tính từ thời điểm request CUỐI CÙNG của batch đó. Khi hết thời gian chờ,
# reset batch và gửi tiếp RATE_LIMIT request mới.
# Khác sliding window: không tự động "mở" slot theo thời gian — chờ hẳn 1 phút
# rồi mới gửi cả batch tiếp theo, tận dụng tối đa 15 request/phút.
_batch_count: int = 0                 # số request đã gửi trong batch hiện tại
_batch_last_time: float | None = None # thời điểm monotonic của request cuối batch
_rate_lock = Lock()

def _throttle() -> None:
    """
    Chặn luồng gọi nếu batch hiện tại đã đầy (>= RATE_LIMIT request).
    Sau khi chờ đủ RATE_WINDOW giây từ request cuối của batch, reset và cho phép tiếp.
    Thread-safe: nhiều worker gọi đồng thời vẫn đúng.
    """
    global _batch_count, _batch_last_time
    while True:
        with _rate_lock:
            now = time.monotonic()
            if _batch_count < RATE_LIMIT:
                # Còn slot trong batch hiện tại → cho qua ngay
                _batch_count += 1
                _batch_last_time = now
                return
            # Batch đầy → kiểm tra xem đã đủ RATE_WINDOW từ request cuối chưa
            elapsed = now - (_batch_last_time or now)
            if elapsed >= RATE_WINDOW:
                # Đủ thời gian → reset batch, cho request đầu tiên của batch mới
                _batch_count = 1
                _batch_last_time = now
                log.info("  🔄 Rate limit: reset batch mới (đã chờ %.1fs)", elapsed)
                return
            wait = RATE_WINDOW - elapsed
        # Ngủ bên ngoài lock để không block các luồng khác kiểm tra
        log.info("  ⏳ Rate limit: batch đầy (%d/%d), chờ %.1fs từ request cuối...",
                 RATE_LIMIT, RATE_LIMIT, wait)
        time.sleep(wait + 0.2)   # +0.2s buffer nhỏ tránh lệch đồng hồ


# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
Bạn là hệ thống OCR chuyên biệt cho đề thi THPT Việt Nam.

━━━ VAI TRÒ ━━━
Input : Ảnh đề thi (scan, chụp, screenshot). Có thể có nhiều câu hỏi.
Output: JSON object thuần duy nhất. Không markdown, không giải thích, không code fence.
Chuẩn : Zero tolerance — sai một tọa độ, thiếu một câu, nhầm loại ảnh đều là lỗi nghiêm trọng.

━━━ QUY TẮC OUTPUT ━━━
1. Chỉ trả về JSON bắt đầu { kết thúc }.
2. LaTeX trong $...$, mọi backslash PHẢI doubled: \\\\dfrac \\\\sqrt \\\\pi \\\\vec \\\\overrightarrow
3. Giữ nguyên 100% nội dung — không tóm tắt, không paraphrase.
4. Không thêm, không bỏ sót bất kỳ câu hỏi nào có trong ảnh.

━━━ QUY TẮC FIELD "question" ━━━
Field "question" chỉ chứa NỘI DUNG câu hỏi — KHÔNG chứa nhãn số câu.
Nhãn số câu là: "Câu 1:", "Câu 2:", "Câu 3:", ... hoặc "Câu 1.", "Câu 2.", v.v.
  SAI : "question": "Câu 2: Xét đoạn dây dẫn thẳng..."
  ĐÚNG: "question": "Xét đoạn dây dẫn thẳng..."
Bắt đầu "question" từ ký tự đầu tiên NGAY SAU dấu hai chấm (hoặc dấu chấm) của nhãn câu.
Trim khoảng trắng đầu chuỗi.

━━━ HAI LOẠI ẢNH — KHÔNG ĐƯỢC NHẦM ━━━

LOẠI 1 — ẢNH MINH HỌA ĐỀ BÀI (question illustration):
  Định nghĩa: Hình vẽ/biểu đồ phục vụ nội dung câu hỏi — KHÔNG phải đáp án.
  Ví dụ: hệ tọa độ Oxyz, mạch điện, đồ thị hàm số, hình học phẳng, bảng số liệu.
  Vị trí trong JSON: q["image"] = {"bbox":[...], "alt":"...", "width":"..."}

  VỊ TRÍ HÌNH TRONG ĐỀ — hình có thể xuất hiện ở BẤT KỲ đâu trong câu:
    • Bên phải text câu hỏi (text trái, hình phải — cùng hàng)
    • Bên dưới text câu hỏi (hình nằm giữa câu hỏi và đáp án)
    • Bên trái text câu hỏi (hiếm, nhưng có)
    • Xen kẽ giữa các dòng text
  → Dù hình ở đâu, bbox PHẢI bao trọn toàn bộ vùng visual đó.

  VÙNG CẦN BAO GỒM VÀO BBOX:
    ✅ Toàn bộ hình vẽ / biểu đồ / sơ đồ
    ✅ Nhãn chú thích gắn liền với hình: tên trục (x, y, z), ký hiệu vật lý
       (→B, →F, →I, ⊕, ⊙), tên đỉnh hình học (A, B, C, D khi là tên điểm/đỉnh),
       số đo góc, kích thước cạnh, chú thích trong/ngoài hình
    ✅ Dòng caption ngay dưới hình nếu thuộc về hình đó
    ✅ Khoảng trắng tối thiểu bao quanh để hình không bị cắt sát

  VÙNG TUYỆT ĐỐI KHÔNG ĐƯA VÀO BBOX:
    ❌ Text câu hỏi phía trên hình ("Câu N: ...", nội dung đề bài)
    ❌ Nhãn đáp án trắc nghiệm: "A.", "B.", "C.", "D."
    ❌ Nhãn mệnh đề đúng/sai: "a.", "b.", "c.", "d." / "a)", "b)", "c)", "d)"
    ❌ Text của đáp án/mệnh đề
    ❌ Phần thuộc câu hỏi khác

  CÁCH PHÂN BIỆT nhãn tên điểm/đỉnh (lấy) vs nhãn đáp án (không lấy):
    • Tên điểm/đỉnh: nằm TRONG hoặc KỀ SÁT hình vẽ, không có dấu chấm theo sau text dài.
    • Nhãn đáp án: nằm NGOÀI hình, theo sau là text/hình đáp án riêng biệt.

LOẠI 2 — ẢNH MINH HỌA ĐÁP ÁN (answer illustration):
  Định nghĩa: Mỗi đáp án A/B/C/D là một hình vẽ riêng biệt.
  Vị trí trong JSON: q["options"][i]["image"] = {"bbox":[...], "alt":"...", "width":"..."}
  Crop: mỗi đáp án được crop HOÀN TOÀN ĐỘC LẬP.
  TUYỆT ĐỐI KHÔNG đặt ảnh đáp án vào q["image"] và ngược lại.

  VÙNG CẦN BAO GỒM:
    ✅ Toàn bộ hình vẽ của đáp án đó
    ✅ Nhãn ký hiệu trong hình (→B, →F, ⊕, ⊙, N, S, tên điểm gắn liền hình)
  VÙNG KHÔNG ĐƯA VÀO:
    ❌ Nhãn "A.", "B.", "C.", "D." đứng ngoài hình
    ❌ Text câu hỏi phía trên
    ❌ Hình của đáp án khác (x_max[A] < x_min[B] < x_max[B] < x_min[C] < x_max[C] < x_min[D])

━━━ HỆ TỌA ĐỘ BBOX ━━━
bbox = [x_min, y_min, x_max, y_max], giá trị 0.0→1.0 (normalized theo kích thước ảnh).
Gốc (0,0) = góc TRÊN-TRÁI. Điểm (1,1) = góc DƯỚI-PHẢI. x tăng →phải, y tăng →dưới.

QUY TRÌNH XÁC ĐỊNH BBOX (bắt buộc từng bước):
  B1. Xác định pixel nội dung ngoài cùng theo 4 hướng.
  B2. x_min = x pixel nội dung trái nhất.
  B3. x_max = x pixel nội dung phải nhất (kể cả đầu mũi tên, ký hiệu phải nhất).
  B4. y_min = y pixel nội dung trên nhất — KHÔNG bao text câu hỏi phía trên.
  B5. y_max = y pixel nội dung dưới nhất — BAO GỒM ký hiệu/caption thấp nhất gắn với hình.
  B6. Kiểm tra: không lấn sang nhãn đáp án hay text đáp án. Chỉ ghi sau khi pass.\
"""

# ── Prompt tầng 1: phân tích cấu trúc, KHÔNG OCR nội dung ─────────────────────
SCAN_PROMPT = """\
Quan sát toàn bộ ảnh đề thi. Trả về JSON duy nhất, không markdown, không giải thích.

NHIỆM VỤ: Liệt kê tất cả câu hỏi có trong ảnh. KHÔNG OCR nội dung câu hỏi.

{
  "subject": "<toan|vat_ly|hoa_hoc|sinh_hoc|lich_su|dia_ly|gdcd|tieng_anh|tin_hoc|cong_nghe|ngu_van|unknown>",
  "questions": [
    {
      "q":       <số thứ tự câu — đúng số ghi trong đề, ví dụ 1, 2, 3...>,
      "part":    <1 = trắc nghiệm 1 đáp án | 2 = đúng/sai mệnh đề | 3 = trả lời ngắn>,
      "type":    "<TEXT_ONLY|Q_IMG+TEXT|OPT_IMG_ONLY|Q_IMG+OPT_IMG|PART2|PART3>",
      "label_y": <y normalized 0.0–1.0 của dòng "Câu N:" trong ảnh>,
      "cut":     <true nếu câu bị cắt ở cuối trang (đáp án/nội dung tiếp tục sang trang sau), false nếu hoàn chỉnh>
    }
  ]
}

QUY TẮC PHÂN LOẠI TYPE:
  TEXT_ONLY    : câu hỏi text + 4 đáp án A/B/C/D là text, không có hình nào
  Q_IMG+TEXT   : câu hỏi có hình minh họa đề bài + 4 đáp án là text
  OPT_IMG_ONLY : câu hỏi text + 4 đáp án A/B/C/D mỗi cái là một hình riêng
  Q_IMG+OPT_IMG: có cả hình đề bài lẫn 4 đáp án là hình
  PART2        : câu đúng/sai với các mệnh đề a/b/c/d
  PART3        : câu trả lời ngắn / điền số

QUY TẮC PHÁT HIỆN CÂU BỊ CẮT ("cut": true):
  Câu bị cắt khi nhãn "Câu N:" xuất hiện nhưng phần đáp án/nội dung bên dưới
  bị cắt mất (chạy ra ngoài đáy ảnh, không thấy đủ A/B/C/D hoặc mệnh đề).
  Dấu hiệu chính: label_y > 0.75 VÀ không thấy đủ nội dung câu trong ảnh.

QUY TẮC ĐẾM:
  • Đếm CHÍNH XÁC — câu 1 dòng ngắn vẫn là 1 câu đầy đủ
  • Đọc đúng SỐ CÂU ghi trong đề — không tự đánh số lại
  • KHÔNG bỏ sót bất kỳ câu nào dù ngắn hay không có hình\
"""

SCAN_PROMPT_CONT = """\
Quan sát toàn bộ ảnh. Đây là một trang tiếp theo của đề thi THPT — có thể không có tiêu đề môn học.
Trả về JSON duy nhất, không markdown, không giải thích.

NHIỆM VỤ: Liệt kê tất cả câu hỏi có trong ảnh. KHÔNG OCR nội dung câu hỏi.

{
  "subject": "<toan|vat_ly|hoa_hoc|sinh_hoc|lich_su|dia_ly|gdcd|tieng_anh|tin_hoc|cong_nghe|ngu_van|unknown>",
  "questions": [
    {
      "q":       <số thứ tự câu — đúng số ghi trong đề, ví dụ 5, 6, 7...>,
      "part":    <1 | 2 | 3>,
      "type":    "<TEXT_ONLY|Q_IMG+TEXT|OPT_IMG_ONLY|Q_IMG+OPT_IMG|PART2|PART3>",
      "label_y": <y normalized 0.0–1.0>,
      "cut":     <true nếu bị cắt ở cuối trang, false nếu hoàn chỉnh>
    }
  ]
}

DẤU HIỆU CÂU HỎI TRÊN TRANG NÀY: dòng bắt đầu "Câu N:" với N bất kỳ (không nhất thiết từ 1).
QUY TẮC TYPE, ĐẾM, PHÁT HIỆN CẮT: như trang đầu.\
"""

# ── Prompt tầng 2: xử lý đúng 1 câu ──────────────────────────────────────────
def make_single_prompt(q_num: int, q_type: str, part: int) -> str:
    part_schema = {
        1: """\
Schema trả về (part1 — trắc nghiệm):
{
  "question": "<nội dung câu hỏi — KHÔNG chứa 'Câu N:' hoặc 'Câu N.' ở đầu>",
  "options": ["<text A>", "<text B>", "<text C>", "<text D>"],
  "answer": "A|B|C|D|\\"\\"",
  "image": {"bbox": [x,y,x,y], "alt": "<mô tả>", "width": "260px"}
}
Nếu đáp án là hình thay vì text, KHÔNG có q["image"], dùng:
  "options": [
    {"image": {"bbox": [x,y,x,y], "alt": "<mô tả A>", "width": "220px"}},
    {"image": {"bbox": [x,y,x,y], "alt": "<mô tả B>", "width": "220px"}},
    {"image": {"bbox": [x,y,x,y], "alt": "<mô tả C>", "width": "220px"}},
    {"image": {"bbox": [x,y,x,y], "alt": "<mô tả D>", "width": "220px"}}
  ]
Chỉ có q["image"] khi câu có ảnh minh họa ĐỀ BÀI (LOẠI 1). Bỏ field "image" nếu không có hình đề bài.""",

        2: """\
Schema trả về (part2 — đúng/sai):
{
  "question": "<câu dẫn — KHÔNG chứa 'Câu N:' hoặc 'Câu N.' ở đầu>",
  "subItems": [
    {"statement": "<mệnh đề a>", "answer": "true|false|\\"\\""} ,
    {"statement": "<mệnh đề b>", "answer": "true|false|\\"\\""} ,
    {"statement": "<mệnh đề c>", "answer": "true|false|\\"\\""} ,
    {"statement": "<mệnh đề d>", "answer": "true|false|\\"\\""} 
  ]
}
Nếu có hình minh họa đề bài, thêm: "image": {"bbox": [x,y,x,y], "alt": "...", "width": "260px"}""",

        3: """\
Schema trả về (part3 — trả lời ngắn):
{
  "question": "<full text — KHÔNG chứa 'Câu N:' hoặc 'Câu N.' ở đầu>",
  "hint": "<chọn 1: 'Nhập kết quả là một số nguyên' | 'Nhập kết quả (làm tròn 2 chữ số thập phân, ví dụ: 4.00)' | 'Nhập kết quả là một số nguyên hoặc phân số' | 'Nhập biểu thức rút gọn' | 'Nhập từ hoặc cụm từ'>",
  "answer": "<đáp án nếu biết, không thì \\"\\">"
}
Nếu có hình minh họa đề bài, thêm: "image": {"bbox": [x,y,x,y], "alt": "...", "width": "260px"}""",
    }

    return f"""\
Ảnh này chứa nhiều câu hỏi. Nhiệm vụ của bạn: CHỈ XỬ LÝ CÂU {q_num}.
Bỏ qua hoàn toàn tất cả câu khác — không đọc, không nhắc đến.

Loại câu: {q_type}

{part_schema[part]}

━━━ QUY TẮC FIELD "question" ━━━
Nhãn câu hỏi như "Câu {q_num}:", "Câu {q_num}." KHÔNG được đưa vào field "question".
Bắt đầu "question" từ ký tự đầu tiên của NỘI DUNG sau nhãn. Trim khoảng trắng đầu.
  SAI : "question": "Câu {q_num}: Trong không gian có hệ trục..."
  ĐÚNG: "question": "Trong không gian có hệ trục..."

━━━ QUY TẮC BBOX (áp dụng cho mọi image field) ━━━
bbox = [x_min, y_min, x_max, y_max], normalized 0.0→1.0 theo ảnh gốc.
Gốc (0,0) = góc TRÊN-TRÁI. x tăng →phải, y tăng →dưới.

  B1. x_min = x pixel nội dung trái nhất của vùng hình.
  B2. x_max = x pixel nội dung phải nhất (kể cả đầu mũi tên, ký hiệu xa nhất bên phải).
  B3. y_min = y pixel nội dung trên nhất — KHÔNG bao text câu hỏi hay nhãn "Câu {q_num}:".
  B4. y_max = y pixel nội dung dưới nhất — BAO GỒM nhãn/caption thấp nhất gắn với hình.
  B5. Với ảnh đáp án (LOẠI 2): x_max[A] < x_min[B] < x_max[B] < x_min[C] < x_max[C] < x_min[D].
  B6. Kiểm tra không lấn sang nhãn "A." "B." "C." "D." hay text đáp án. Chỉ ghi sau khi pass.

Output: chỉ JSON thuần bắt đầu {{ kết thúc }}. Không markdown, không giải thích.\
"""

FIX_PROMPT = "JSON sau lỗi cú pháp. Sửa thành JSON hợp lệ (không markdown, không text khác):\n\n"


# ── JSON parse & validate ──────────────────────────────────────────────────────
_VALID_ESC = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}

def _fix_escapes(s: str) -> str:
    res, i, in_s = [], 0, False
    while i < len(s):
        c = s[i]
        if c == '"':
            n = 0; j = i - 1
            while j >= 0 and s[j] == '\\': n += 1; j -= 1
            if n % 2 == 0: in_s = not in_s
            res.append(c); i += 1
        elif in_s and c == '\\':
            nc = s[i + 1] if i + 1 < len(s) else ''
            if nc in _VALID_ESC: res += [c, nc]; i += 2
            else: res.append('\\\\'); i += 1
        else: res.append(c); i += 1
    return ''.join(res)

def parse_json(raw: str) -> dict:
    t = raw.strip()
    if t.startswith("```"):
        ls = t.splitlines()
        t = "\n".join(ls[1:(-1 if ls[-1].strip() == "```" else len(ls))]).strip()
    for fn in (lambda x: x, _fix_escapes):
        try:
            return json.loads(fn(t))
        except (json.JSONDecodeError, ValueError):
            pass
    raise ValueError("Không parse được JSON")

_VALID_P1_ANSWERS = {"A", "B", "C", "D", ""}
_VALID_P2_ANSWERS = {"true", "false", ""}

def validate(d: dict) -> list[str]:
    if not isinstance(d, dict): return ["Root không phải dict"]
    errs = []
    for p in ("part1", "part2", "part3"):
        items = d.get(p)
        if not isinstance(items, list):
            errs.append(f"'{p}' thiếu/sai kiểu"); continue
        for i, q in enumerate(items):
            if not isinstance(q, dict) or "question" not in q:
                errs.append(f"{p}[{i}] thiếu question"); continue
            if p == "part1":
                opts = q.get("options")
                if not isinstance(opts, list) or len(opts) != 4:
                    errs.append(f"{p}[{i}] options phải có 4 phần tử")
                ans = q.get("answer", "")
                if ans not in _VALID_P1_ANSWERS:
                    errs.append(f"{p}[{i}] answer '{ans}' không hợp lệ (phải là A/B/C/D hoặc '')")
            elif p == "part2":
                if not q.get("subItems"):
                    errs.append(f"{p}[{i}] subItems rỗng")
                else:
                    for j, sub in enumerate(q["subItems"]):
                        ans = sub.get("answer", "")
                        if ans not in _VALID_P2_ANSWERS:
                            errs.append(f"{p}[{i}].subItems[{j}] answer '{ans}' phải là true/false/''")
            elif p == "part3":
                if "hint" not in q:
                    errs.append(f"{p}[{i}] thiếu hint")
    return errs

# ── File → images ──────────────────────────────────────────────────────────────
def stitch_images(p1: Path, p2: Path, out: Path) -> Path:
    img1 = Image.open(p1).convert("RGB")
    img2 = Image.open(p2).convert("RGB")
    w = max(img1.width, img2.width)
    if img1.width != w:
        img1 = img1.resize((w, int(img1.height * w / img1.width)), Image.LANCZOS)
    if img2.width != w:
        img2 = img2.resize((w, int(img2.height * w / img2.width)), Image.LANCZOS)
    canvas = Image.new("RGB", (w, img1.height + img2.height), (255, 255, 255))
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (0, img1.height))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, format="PNG", optimize=True)
    log.debug("  🧩 Ghép %s + %s → %s (%dx%dpx)",
              p1.name, p2.name, out.name, w, img1.height + img2.height)
    return out

def pdf_to_imgs(pdf: Path, out: Path) -> list[Path]:
    try: import fitz
    except ImportError: sys.exit("pip install pymupdf")
    out.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf); paths = []
    for i, pg in enumerate(doc):
        p = out / f"{pdf.stem}_p{i+1}.png"
        pg.get_pixmap(matrix=fitz.Matrix(3, 3)).save(p)
        paths.append(p)
    doc.close(); return paths

def expand(src: Path, tmp: Path) -> list[Path]:
    ext = src.suffix.lower()
    if ext in IMG_EXTS: return [src]
    if ext == ".pdf":   return pdf_to_imgs(src, tmp)
    if ext == ".docx":
        try:
            import docx2pdf
        except ImportError:
            sys.exit(
                "pip install docx2pdf\n"
                "Lưu ý: docx2pdf yêu cầu Microsoft Word trên Windows/macOS, "
                "hoặc LibreOffice trên Linux."
            )
        tmp.mkdir(parents=True, exist_ok=True)
        docx2pdf.convert(str(src), str(tmp))
        pdf = tmp / f"{src.stem}.pdf"
        if not pdf.exists():
            raise RuntimeError(f"docx2pdf không tạo được file: {pdf}")
        return pdf_to_imgs(pdf, tmp)
    raise ValueError(f"Không hỗ trợ định dạng: {ext}")

def sort_images_by_name(images: list[Path]) -> list[Path]:
    def _key(p: Path):
        nums = re.findall(r'\d+', p.stem)
        return (int(nums[-1]) if nums else 0, p.stem)
    return sorted(images, key=_key)

# ── Crop ───────────────────────────────────────────────────────────────────────
def _normalize_image_mode(img: Image.Image) -> Image.Image:
    if img.mode == "CMYK":          return img.convert("RGB")
    if img.mode in ("P", "PA"):     return img.convert("RGBA")
    if img.mode == "LA":            return img.convert("RGBA")
    if img.mode not in ("RGB", "RGBA", "L"): return img.convert("RGB")
    return img

def crop_save(src: Path, bbox: list, out: Path) -> bool:
    try:
        img = Image.open(src)
        img = _normalize_image_mode(img)
        W, H = img.size
        xn, yn, xx, yx = [max(0.0, min(float(v), 1.0)) for v in bbox]
        xn = max(0.0, xn - CROP_PADDING)
        yn = max(0.0, yn - CROP_PADDING)
        xx = min(1.0, xx + CROP_PADDING)
        yx = min(1.0, yx + CROP_PADDING_BOTTOM)
        if xn >= xx or yn >= yx:
            log.warning("Crop: bbox không hợp lệ sau padding [%.3f,%.3f,%.3f,%.3f]", xn, yn, xx, yx)
            return False
        left, top, right, bottom = int(xn*W), int(yn*H), int(xx*W), int(yx*H)
        right  = max(right,  left + 1)
        bottom = max(bottom, top  + 1)
        cropped = img.crop((left, top, right, bottom))
        cw, ch  = cropped.size
        if cw < CROP_MIN_PX or ch < CROP_MIN_PX:
            scale    = max(CROP_MIN_PX / cw, CROP_MIN_PX / ch)
            new_size = (max(1, int(cw * scale)), max(1, int(ch * scale)))
            cropped  = cropped.resize(new_size, Image.LANCZOS)
            log.debug("  Upscale crop: %dx%d → %dx%d", cw, ch, *new_size)
        cropped = cropped.filter(ImageFilter.SHARPEN)
        cropped = ImageEnhance.Contrast(cropped).enhance(1.08)
        if cropped.mode not in ("RGB", "RGBA", "L"):
            cropped = cropped.convert("RGB")
        out.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(out, format="PNG", optimize=True)
        log.debug("  Crop %s: [%.3f,%.3f,%.3f,%.3f] → %dx%d px",
                  out.name, xn, yn, xx, yx, *cropped.size)
        return True
    except Exception as e:
        log.error("Crop '%s': %s", out.name, e)
        return False

def apply_crops(result: dict, src: Path, img_dir: Path) -> None:
    stem = src.stem

    def _do_crop(node: dict, fname: str) -> dict:
        ok = crop_save(src, node["bbox"], img_dir / fname)
        if ok:  log.info("  ✂️  %s", fname)
        else:   log.warning("  ✂️  Crop thất bại: %s", fname)
        return {"src": f"images/{fname}" if ok else "", "alt": node.get("alt", ""), "width": node.get("width", "260px")}

    for pk in ("part1", "part2", "part3"):
        for qi, q in enumerate(result.get(pk, [])):
            base = f"{stem}-{pk.replace('part', 'phan')}-cau{qi + 1}"
            if isinstance(q.get("image"), dict) and "bbox" in q["image"]:
                q["image"] = _do_crop(q["image"], f"{base}.png")
            if pk == "part1":
                for oi, opt in enumerate(q.get("options", [])):
                    if isinstance(opt, dict) and isinstance(opt.get("image"), dict) and "bbox" in opt["image"]:
                        opt["image"] = _do_crop(opt["image"], f"{base}-opt{chr(65+oi)}.png")
            if pk == "part2":
                for si, sub in enumerate(q.get("subItems", [])):
                    if isinstance(sub, dict) and isinstance(sub.get("image"), dict) and "bbox" in sub["image"]:
                        sub["image"] = _do_crop(sub["image"], f"{base}-item{si+1}.png")

def preprocess_for_ai(img_path: Path) -> tuple[bytes, str]:
    import io
    try:
        img = Image.open(img_path)
        img = _normalize_image_mode(img)
        W, H = img.size
        long_side = max(W, H)
        if long_side < 1200:
            scale = 1200 / long_side
            img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
            log.debug("  preprocess: upscale %dx%d → %dx%d", W, H, *img.size)
        elif long_side > 2400:
            scale = 2400 / long_side
            img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
            log.debug("  preprocess: downscale %dx%d → %dx%d", W, H, *img.size)
        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = ImageEnhance.Sharpness(img).enhance(1.8)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88, optimize=True)
        return buf.getvalue(), "image/jpeg"
    except Exception as e:
        log.warning("preprocess_for_ai thất bại, dùng file gốc: %s", e)
        mime = MIME.get(img_path.suffix.lower(), "image/jpeg")
        return img_path.read_bytes(), mime

# ── API helpers ────────────────────────────────────────────────────────────────
def _call(client, contents, system=None, max_tok=MAX_TOKENS) -> str:
    _throttle()
    kw = {"temperature": 0.0, "max_output_tokens": max_tok}
    if system: kw["system_instruction"] = system
    with ThreadPoolExecutor(1) as ex:
        fut = ex.submit(
            lambda: client.models.generate_content(
                model=MODEL, config=types.GenerateContentConfig(**kw), contents=contents)
        )
        try:
            resp = fut.result(timeout=TIMEOUT)
        except TimeoutError:
            fut.cancel()
            raise RuntimeError(f"API timeout sau {TIMEOUT}s")
    if getattr(resp, "candidates", []) and str(getattr(resp.candidates[0], "finish_reason", "")) == "MAX_TOKENS":
        log.warning("⚠️ Response bị cắt (MAX_TOKENS) — cân nhắc chia nhỏ ảnh")
    return resp.text

def _img_parts(img_bytes: bytes, mime: str, text: str):
    return [types.Content(role="user", parts=[
        types.Part.from_bytes(data=img_bytes, mime_type=mime),
        types.Part.from_text(text=text),
    ])]

def _text_parts(text: str):
    return [types.Content(role="user", parts=[types.Part.from_text(text=text)])]

def _keyword_match(text: str) -> str:
    t = text.lower()
    for code, words in _KEYWORDS.items():
        if any(w in t for w in words):
            return code
    return "unknown"

# ── Tầng 1: phân tích cấu trúc ────────────────────────────────────────────────
def detect_questions(client, img_bytes: bytes, mime: str,
                     subject_hint: str = "",
                     skip_q: set[int] | None = None) -> dict:
    def _make_prompt(base: str) -> str:
        prefix_lines = []
        if subject_hint and subject_hint not in ("unknown", ""):
            prefix_lines.append(
                f"LƯU Ý: Đây là trang thuộc đề thi môn '{subject_hint}' "
                f"(không cần tìm tiêu đề, chỉ liệt kê câu hỏi)."
            )
        if skip_q:
            prefix_lines.append(
                f"CÁC CÂU ĐÃ XỬ LÝ TỪ TRANG TRƯỚC (bỏ qua, không liệt kê): {sorted(skip_q)}"
            )
        return ("\n".join(prefix_lines) + "\n" + base) if prefix_lines else base

    for prompt_name, prompt in [("scan", SCAN_PROMPT), ("cont", SCAN_PROMPT_CONT)]:
        full = _make_prompt(prompt)
        for attempt in range(1, RETRIES + 1):
            try:
                raw  = _call(client, _img_parts(img_bytes, mime, full))
                data = parse_json(raw)
                qs   = data.get("questions", [])
                if skip_q:
                    qs = [q for q in qs if q["q"] not in skip_q]
                    data["questions"] = qs
                if not qs:
                    raise ValueError("Không tìm thấy câu hỏi nào")
                log.info("  📋 Tầng 1 [%s]: %d câu — %s", prompt_name,
                         len(qs), [f"Câu{q['q']}({q['type']})" for q in qs])
                return data
            except Exception as e:
                log.warning("  Tầng 1 [%s] lần %d: %s", prompt_name, attempt, e)
                if attempt < RETRIES:
                    time.sleep(RETRY_DELAY * 2 ** (attempt - 1))

    raise RuntimeError("Tầng 1 thất bại sau tất cả lần thử (cả 2 prompt)")

# ── Tầng 2: OCR một câu ───────────────────────────────────────────────────────
def extract_one(client, img_bytes: bytes, mime: str,
                q_info: dict) -> tuple[int, int, dict] | None:
    q_num  = q_info["q"]
    part   = q_info.get("part", 1)
    q_type = q_info.get("type", "TEXT_ONLY")
    prompt = make_single_prompt(q_num, q_type, part)
    last_raw = ""

    for attempt in range(1, RETRIES + 1):
        try:
            raw      = _call(client, _img_parts(img_bytes, mime, prompt), SYSTEM_PROMPT)
            last_raw = raw
            result   = parse_json(raw)
            if "question" not in result:
                raise ValueError("Thiếu field 'question'")
            if errs := validate({"part1": [result], "part2": [], "part3": []} if part == 1
                                 else {"part1": [], "part2": [result], "part3": []} if part == 2
                                 else {"part1": [], "part2": [], "part3": [result]}):
                log.warning("  Câu %d cấu trúc: %s", q_num, "; ".join(errs))
            log.info("  ✅ Câu %d [%s] OK", q_num, q_type)
            return q_num, part, result

        except ValueError:
            if last_raw and attempt < RETRIES:
                log.warning("  Câu %d lần %d: JSON lỗi → thử sửa...", q_num, attempt)
                try:
                    fixed  = _call(client, _text_parts(FIX_PROMPT + last_raw))
                    result = parse_json(fixed)
                    if "question" not in result:
                        raise ValueError("Thiếu field 'question' sau fix")
                    log.info("  ✅ Câu %d OK (sau fix JSON)", q_num)
                    return q_num, part, result
                except Exception as fe:
                    log.warning("  Fix JSON câu %d thất bại: %s", q_num, fe)

        except Exception as e:
            log.warning("  Câu %d lần %d: %s", q_num, attempt, e)

        if attempt < RETRIES:
            time.sleep(RETRY_DELAY * 2 ** (attempt - 1))

    log.error("  ❌ Câu %d thất bại sau %d lần thử", q_num, RETRIES)
    return None

# ── Orchestrator: 1 ảnh → (subject, result) ───────────────────────────────────
def extract_questions(client, img_path: Path,
                      subject_hint: str = "",
                      skip_q: set[int] | None = None) -> tuple[str, dict]:
    img_bytes, mime = preprocess_for_ai(img_path)
    scan    = detect_questions(client, img_bytes, mime, subject_hint, skip_q)
    subject = scan.get("subject", "unknown").strip().lower()
    q_list  = scan.get("questions", [])

    valid = set(SUBJECTS) | {SUBJECT_UNSUPPORTED, "unknown"}
    if subject not in valid:
        subject = _keyword_match(scan.get("subject", ""))
    if subject in ("unknown", "") and subject_hint and subject_hint not in ("unknown", ""):
        subject = subject_hint
        log.info("  ↳ subject kế thừa từ trang trước: %s", subject)

    if subject == SUBJECT_UNSUPPORTED:
        return subject, {"part1": [], "part2": [], "part3": []}
    if not q_list:
        raise RuntimeError("Tầng 1 không tìm thấy câu hỏi nào")

    part_map = {1: "part1", 2: "part2", 3: "part3"}
    result   = {"part1": [], "part2": [], "part3": []}
    answers: dict[int, tuple[int, dict]] = {}

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(extract_one, client, img_bytes, mime, q): q for q in q_list}
        for fut in as_completed(futures):
            out = fut.result()
            if out:
                q_num, part_num, q_data = out
                answers[q_num] = (part_num, q_data)

    for q_num in sorted(answers):
        part_num, q_data = answers[q_num]
        result[part_map.get(part_num, "part1")].append(q_data)

    p1, p2, p3 = len(result["part1"]), len(result["part2"]), len(result["part3"])
    log.info("  📊 Tổng: part1=%d part2=%d part3=%d (thất bại: %d câu)",
             p1, p2, p3, len(q_list) - p1 - p2 - p3)
    return subject, result

# ── Xử lý cross-page: câu bị cắt giữa 2 trang ────────────────────────────────
def handle_cross_page(client, img_dir: Path, tmp_dir: Path,
                      page_a: Path, page_b: Path,
                      cut_questions: list[dict],
                      subject_hint: str) -> tuple[dict, set[int]]:
    stitched_path = tmp_dir / f"stitched_{page_a.stem}__{page_b.stem}.png"
    try:
        stitch_images(page_a, page_b, stitched_path)
    except Exception as e:
        log.warning("  Ghép trang thất bại: %s", e)
        return {"part1": [], "part2": [], "part3": []}, set()

    log.info("  🔗 Cross-page: xử lý %d câu bị cắt từ ảnh ghép [%s + %s]",
             len(cut_questions), page_a.name, page_b.name)

    img_bytes, mime = preprocess_for_ai(stitched_path)
    part_map = {1: "part1", 2: "part2", 3: "part3"}
    result   = {"part1": [], "part2": [], "part3": []}
    handled: set[int] = set()
    answers: dict[int, tuple[int, dict]] = {}

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(extract_one, client, img_bytes, mime, q): q for q in cut_questions}
        for fut in as_completed(futures):
            out = fut.result()
            if out:
                q_num, part_num, q_data = out
                answers[q_num] = (part_num, q_data)
                handled.add(q_num)

    for q_num in sorted(answers):
        part_num, q_data = answers[q_num]
        result[part_map.get(part_num, "part1")].append(q_data)

    apply_crops(result, stitched_path, img_dir)
    return result, handled

# ── Process một nhóm trang tuần tự ───────────────────────────────────────────
def process_group(client, group: list[Path], img_dir: Path,
                  tmp_dir: Path) -> list[tuple[str, dict | None]]:
    results: list[tuple[str, dict | None]] = []
    subject_hint    = ""
    cross_page_skip: set[int] = set()

    for idx, img_path in enumerate(group):
        try:
            subject, result = extract_questions(
                client, img_path, subject_hint, skip_q=cross_page_skip
            )
            cross_page_skip = set()

            if subject == SUBJECT_UNSUPPORTED:
                log.warning("⏭️  %s — Ngữ Văn không hỗ trợ (tự luận)", img_path.name)
                results.append((subject, None))
                continue

            label = SUBJECTS.get(subject, ("?",))[0]
            log.info("📚 %s → %s", img_path.name, label)
            apply_crops(result, img_path, img_dir)

            if idx + 1 < len(group):
                img_bytes, mime = preprocess_for_ai(img_path)
                try:
                    scan     = detect_questions(client, img_bytes, mime, subject)
                    cut_qs   = [q for q in scan.get("questions", []) if q.get("cut", False)]
                    if cut_qs:
                        log.info("  ✂️  Phát hiện %d câu bị cắt trang: %s",
                                 len(cut_qs), [q["q"] for q in cut_qs])
                        cross_result, handled = handle_cross_page(
                            client, img_dir, tmp_dir,
                            img_path, group[idx + 1],
                            cut_qs, subject
                        )
                        cross_page_skip = handled
                        if any(cross_result[k] for k in ("part1", "part2", "part3")):
                            results.append((subject, cross_result))
                except Exception as e:
                    log.warning("  Kiểm tra cross-page thất bại: %s", e)

            p1, p2, p3 = (len(result.get(k, [])) for k in ("part1", "part2", "part3"))
            log.info("✅ %s [%s] part1=%d part2=%d part3=%d", img_path.name, label, p1, p2, p3)
            results.append((subject, result))

            if subject and subject not in ("unknown", SUBJECT_UNSUPPORTED, ""):
                subject_hint = subject

        except Exception as e:
            log.error("❌ %s: %s", img_path.name, e)
            results.append(("unknown", None))

    return results

# ── Lọc câu hỏi lỗi ───────────────────────────────────────────────────────────
def _is_valid_q(q: dict, part: str) -> tuple[bool, str]:
    """
    Kiểm tra câu hỏi có hợp lệ không.
    Trả về (True, "") nếu hợp lệ, hoặc (False, lý_do) nếu lỗi.
    """
    # ── Kiểm tra chung: phải có đề bài ──
    question = q.get("question", "").strip()
    if not question:
        return False, "thiếu đề bài (question rỗng)"

    if part == "part1":
        opts = q.get("options", [])
        # Phải có đúng 4 đáp án
        if not isinstance(opts, list) or len(opts) != 4:
            return False, f"options không đủ 4 (có {len(opts) if isinstance(opts, list) else 0})"
        # Mỗi option phải có nội dung (text hoặc image)
        for i, opt in enumerate(opts):
            letter = chr(65 + i)
            if isinstance(opt, str):
                if not opt.strip():
                    return False, f"đáp án {letter} rỗng"
            elif isinstance(opt, dict):
                # Đáp án dạng hình: phải có image với src hoặc bbox
                img = opt.get("image", {})
                if not img:
                    return False, f"đáp án {letter} là dict nhưng không có image"
                if not img.get("src", "").strip() and not img.get("bbox"):
                    return False, f"đáp án {letter} image thiếu src và bbox"
            else:
                return False, f"đáp án {letter} sai kiểu ({type(opt).__name__})"

    elif part == "part2":
        subs = q.get("subItems", [])
        if not isinstance(subs, list) or len(subs) == 0:
            return False, "subItems rỗng hoặc thiếu"
        for i, sub in enumerate(subs):
            if not isinstance(sub, dict) or not sub.get("statement", "").strip():
                return False, f"mệnh đề {chr(97+i)} rỗng hoặc thiếu statement"

    return True, ""


def filter_invalid(data: dict) -> dict:
    """
    Lọc bỏ các câu hỏi lỗi khỏi data (sửa in-place, trả về data).
    Ghi log từng câu bị loại và tổng kết cuối.
    Điều kiện lỗi:
      - Không có đề bài (question rỗng / chỉ khoảng trắng)
      - part1: options không đủ 4, hoặc có option rỗng
      - part2: subItems rỗng hoặc có mệnh đề không có statement
      - part3: thiếu hint
    Câu có answer="" (chưa biết đáp án) vẫn hợp lệ — không bị lọc.
    """
    total_removed = 0
    for part in ("part1", "part2", "part3"):
        items = data.get(part, [])
        valid, removed = [], []
        for q in items:
            ok, reason = _is_valid_q(q, part)
            if ok:
                valid.append(q)
            else:
                removed.append((q.get("question", "")[:40] or "<no text>", reason))
                total_removed += 1
        if removed:
            log.warning("  🗑️  [%s] Loại %d câu lỗi:", part, len(removed))
            for preview, reason in removed:
                log.warning("       • \"%s...\" → %s", preview, reason)
        data[part] = valid
    if total_removed:
        log.info("  ✅ Đã lọc tổng %d câu lỗi", total_removed)
    else:
        log.info("  ✅ Không có câu lỗi")
    return data


# ── Merge ──────────────────────────────────────────────────────────────────────
def merge(base: dict, new: dict):
    for k in ("part1", "part2", "part3"):
        seen = {q.get("question", "").strip() for q in base[k] if q.get("question", "").strip()}
        img_only_count = sum(1 for q in base[k] if not q.get("question", "").strip())
        for q in new.get(k, []):
            key = q.get("question", "").strip()
            if key:
                if key not in seen:
                    base[k].append(q); seen.add(key)
            else:
                img_only_count += 1
                base[k].append(q)

# ── Main ───────────────────────────────────────────────────────────────────────

# ── Web Server ─────────────────────────────────────────────────────────────────
def create_web_app():
    try:
        from flask import Flask, request, jsonify, send_file, send_from_directory
    except ImportError:
        sys.exit("pip install flask")

    import uuid, threading

    app     = Flask(__name__)
    BASE_DIR = Path(__file__).parent.resolve()
    IMG_DIR  = BASE_DIR / "images"
    TMP_DIR  = BASE_DIR / ".tmp"
    IMG_DIR.mkdir(exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)

    _jobs: dict      = {}
    _jobs_lock_w     = Lock()
    ALL_EXTS_WEB     = IMG_EXTS | {".pdf", ".docx"}

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        log.warning("⚠️  GEMINI_API_KEY chưa được đặt — OCR sẽ thất bại")
    client = genai.Client(api_key=api_key) if api_key else None

    @app.route("/")
    def index():
        return send_file(BASE_DIR / "index.html")

    @app.route("/images/<path:filename>")
    def serve_image(filename):
        return send_from_directory(IMG_DIR, filename)

    @app.route("/api/ocr", methods=["POST"])
    def ocr_start():
        if not client:
            return jsonify({"error": "Thiếu GEMINI_API_KEY trong .env"}), 500
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "Không có file nào được gửi lên"}), 400

        job_id = uuid.uuid4().hex[:10]
        # Lưu file xuống disk TRONG request context — trước khi Flask đóng stream
        tmp = TMP_DIR / job_id
        tmp.mkdir(parents=True, exist_ok=True)
        saved = []
        for f in files:
            dst = tmp / Path(f.filename).name
            f.save(dst)
            saved.append(dst)

        with _jobs_lock_w:
            _jobs[job_id] = {"status": "pending", "progress": 0, "total": len(saved), "result": None, "error": None}

        def _run():
            try:
                page_groups: list[list[Path]] = []
                for src in saved:
                    if src.suffix.lower() not in ALL_EXTS_WEB:
                        continue
                    if src.suffix.lower() in IMG_EXTS:
                        page_groups.append([src])
                    else:
                        try:
                            page_groups.append(expand(src, tmp))
                        except Exception as e:
                            log.error("expand %s: %s", src.name, e)

                if not page_groups:
                    raise RuntimeError("Không có file hợp lệ để xử lý")

                with _jobs_lock_w:
                    _jobs[job_id].update({"status": "processing", "total": len(page_groups)})

                combined = {"subject": "unknown", "part1": [], "part2": [], "part3": []}
                subjects_found: list[str] = []
                done = 0

                with ThreadPoolExecutor(max_workers=max(1, WORKERS // 2)) as pool:
                    futs = {pool.submit(process_group, client, grp, IMG_DIR, tmp): grp for grp in page_groups}
                    for fut in as_completed(futs):
                        try:
                            for subject, result in fut.result():
                                if result:
                                    subjects_found.append(subject)
                                    merge(combined, result)
                        except Exception as e:
                            log.error("Job %s group error: %s", job_id, e)
                        done += 1
                        with _jobs_lock_w:
                            _jobs[job_id]["progress"] = done

                if subjects_found:
                    combined["subject"] = SUBJECTS.get(
                        Counter(subjects_found).most_common(1)[0][0], ("unknown",)
                    )[0]
                filter_invalid(combined)

                # Chuyển đường dẫn ảnh sang URL tương đối để browser truy cập được
                def _fix_src(node):
                    if isinstance(node, dict):
                        if node.get("src"):
                            node["src"] = "/" + str(node["src"]).replace("\\", "/").lstrip("/")
                        for v in node.values(): _fix_src(v)
                    elif isinstance(node, list):
                        for item in node: _fix_src(item)
                _fix_src(combined)

                with _jobs_lock_w:
                    _jobs[job_id].update({"status": "done", "result": combined})
            except Exception as e:
                log.error("Job %s thất bại: %s", job_id, e)
                with _jobs_lock_w:
                    _jobs[job_id].update({"status": "error", "error": str(e)})
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        threading.Thread(target=_run, daemon=True).start()
        return jsonify({"job_id": job_id})

    @app.route("/api/status/<job_id>")
    def job_status(job_id):
        with _jobs_lock_w:
            job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job không tồn tại"}), 404
        return jsonify(job)

    return app


def main():
    parser = argparse.ArgumentParser(
        description="OCR đề thi THPT → questions.json | --web để chạy giao diện web",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python app.py --web                              # khởi động web server (port 5000)
  python app.py                                    # xử lý tất cả file trong thư mục
  python app.py de1.pdf de2.jpg                   # chỉ định file cụ thể
  python app.py --pages p1.jpg p2.jpg p3.jpg      # ghép ảnh riêng lẻ thành 1 đề
  python app.py --pages *.jpg --sort              # tự sắp xếp theo số trong tên file
        """
    )
    parser.add_argument("files", nargs="*", help="File ảnh/PDF/DOCX cần xử lý")
    parser.add_argument("--pages", nargs="+", metavar="IMG",
                        help="Ghép nhiều ảnh riêng lẻ thành 1 đề (xử lý theo thứ tự chỉ định)")
    parser.add_argument("--sort", action="store_true",
                        help="Dùng với --pages: tự động sắp xếp ảnh theo số trong tên file")
    parser.add_argument("--web", action="store_true",
                        help="Khởi động web server thay vì chạy CLI")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port cho web server (mặc định: 5000)")
    args = parser.parse_args()

    # ── Web mode: mặc định khi không có file/--pages, hoặc khi dùng --web ─────
    if args.web or (not args.files and not args.pages):
        import webbrowser, threading as _th
        url = f"http://localhost:{args.port}"
        log.info("🌐 Khởi động web server tại %s", url)
        _th.Timer(1.2, lambda: webbrowser.open(url)).start()
        create_web_app().run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
        return

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key: sys.exit("❌ Thiếu GEMINI_API_KEY trong .env")

    client   = genai.Client(api_key=api_key)
    base_dir = Path(__file__).parent.resolve()
    img_dir  = base_dir / "images"
    tmp_dir  = base_dir / ".tmp"
    img_dir.mkdir(exist_ok=True)
    tmp_dir.mkdir(exist_ok=True)

    ALL_EXTS = IMG_EXTS | {".pdf", ".docx"}

    if args.pages:
        page_paths = [Path(p) if Path(p).is_absolute() else base_dir / p for p in args.pages]
        if missing := [p for p in page_paths if not p.exists()]:
            sys.exit(f"❌ Không tìm thấy: {[str(p) for p in missing]}")
        if args.sort:
            page_paths = sort_images_by_name(page_paths)
            log.info("📑 Thứ tự sau khi sắp xếp: %s", [p.name for p in page_paths])
        else:
            log.info("📑 Thứ tự ảnh (theo chỉ định): %s", [p.name for p in page_paths])
        expanded: list[Path] = []
        for p in page_paths:
            expanded.extend(expand(p, tmp_dir))
        page_groups: list[list[Path]] = [expanded]
    else:
        if args.files:
            srcs = [Path(n) if Path(n).is_absolute() else base_dir / n for n in args.files]
            if missing := [p for p in srcs if not p.exists()]:
                sys.exit(f"❌ Không tìm thấy: {[str(p) for p in missing]}")
        else:
            srcs = sorted(p for p in base_dir.iterdir()
                          if p.is_file() and p.suffix.lower() in ALL_EXTS and p.parent == base_dir)
        if not srcs: sys.exit(f"⚠️ Không có file nào trong: {base_dir}")

        all_images: list[Path] = []
        for src in srcs:
            try:
                imgs = expand(src, tmp_dir)
                if src.suffix.lower() not in IMG_EXTS:
                    log.info("📄 %s → %d trang ảnh", src.name, len(imgs))
                all_images.extend(imgs)
            except Exception as e:
                log.error("Không convert được %s: %s", src.name, e)

        if not all_images: sys.exit("⚠️ Không có ảnh nào để xử lý")

        page_groups = []
        for src in srcs:
            ext = src.suffix.lower()
            if ext in IMG_EXTS:
                page_groups.append([src])
            else:
                grp = sorted(p for p in all_images
                             if p.stem.startswith(src.stem + "_p"))
                if grp:
                    page_groups.append(grp)

    log.info("🖼️  %d nhóm | workers=%d | model=%s | rate=%d req/min | timeout=%ds\n",
             len(page_groups), WORKERS, MODEL, RATE_LIMIT, TIMEOUT)

    combined = {"subject": "unknown", "part1": [], "part2": [], "part3": []}
    subjects_found: list[str] = []

    try:
        with ThreadPoolExecutor(max_workers=max(1, WORKERS // 2)) as pool:
            group_futures = {
                pool.submit(process_group, client, grp, img_dir, tmp_dir): grp
                for grp in page_groups
            }
            for fut in as_completed(group_futures):
                try:
                    for subject, result in fut.result():
                        if result:
                            subjects_found.append(subject)
                            merge(combined, result)
                except Exception as e:
                    grp = group_futures[fut]
                    log.error("❌ nhóm %s: %s", [p.name for p in grp], e)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            log.debug("🗑️  Đã xóa thư mục tạm: %s", tmp_dir)

    if subjects_found:
        combined["subject"] = SUBJECTS.get(
            Counter(subjects_found).most_common(1)[0][0], ("unknown",)
        )[0]

    # Lọc câu hỏi lỗi trước khi ghi file
    log.info("\n🔍 Lọc câu hỏi lỗi...")
    filter_invalid(combined)

    out = base_dir / "questions.json"
    out.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(combined[k]) for k in ("part1", "part2", "part3"))
    log.info("\n💾 %d câu → %s", total, out)
    log.info("   Môn: %s | part1=%d part2=%d part3=%d",
             combined["subject"], len(combined["part1"]), len(combined["part2"]), len(combined["part3"]))

if __name__ == "__main__":
    main()