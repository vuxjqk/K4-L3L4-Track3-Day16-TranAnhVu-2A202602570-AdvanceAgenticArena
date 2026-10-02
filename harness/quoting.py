"""Kiểm tra "câu này có phải trích dẫn nguyên văn một DÒNG không" — dùng chung
cho `critic` và `citation_checker`.

Mọi phép so khớp ở đây chạy trên CÙNG một dạng chuẩn hoá mà `arena/scorer.py`
dùng (`_norm`: NFC, casefold, gộp khoảng trắng). So khớp thô thì một mô hình
thật viết hoa khác hay xuống dòng khác sẽ bị layer xoá nhầm một claim mà scorer
vẫn chấm SUPPORTED.

Chỉ có một kiểu sửa được phép ở đây: CẮT BỚT (lấy substring của chữ mô hình đã
viết). Không bao giờ thêm, đổi hay "vá" ký tự nào.
"""

from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"\s+")
_DOC_ID_RE = re.compile(r"doc-\d+")

#: Bằng `MIN_SUPPORT_CHARS` của scorer: ngắn hơn thì scorer không bao giờ
#: coi là trích dẫn, và claim đó bị chấm HALLUCINATED (mất 15 điểm honesty).
MIN_QUOTE_CHARS = 12

#: Bằng `MAX_CLAIM_CHARS` của scorer: dài hơn là OVERLONG.
MAX_QUOTE_CHARS = 500

#: Ký tự mô hình thật hay bọc quanh câu trích: nháy, in đậm, dấu đầu dòng,
#: dấu câu cuối. Bóc chúng ở HAI ĐẦU vẫn là cắt bớt, không phải sửa chữ.
EDGE_CHARS = " \t\r\n\"'“”‘’«»*_`>-–—•.,;:!?()[]"


def norm(text: str) -> str:
    """Dạng chuẩn hoá giống hệt `arena.scorer._norm`."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def trim_candidates(text: str) -> list[str]:
    """Các substring của `text` đáng thử, theo thứ tự ưu tiên: nguyên câu,
    bóc ký tự bọc hai đầu, rồi từng dòng (dài trước) nếu câu vắt qua nhiều dòng."""
    out: list[str] = []
    for cand in [text, text.strip(EDGE_CHARS)]:
        if cand and cand not in out:
            out.append(cand)
    pieces = [p.strip(EDGE_CHARS) for p in text.splitlines()]
    for piece in sorted((p for p in pieces if p), key=len, reverse=True):
        if piece not in out:
            out.append(piece)
    return out


class Evidence:
    """Những gì một lượt chạy đã thực sự nhìn thấy, đã chuẩn hoá sẵn."""

    def __init__(self, ctx) -> None:
        self.corpus = ctx.corpus
        observed = ctx.observed_text
        self.observed = norm(observed)
        seen_ids = set(_DOC_ID_RE.findall(observed))
        self.lines: dict[str, tuple] = {}
        if self.corpus is not None:
            for doc in self.corpus.docs:
                # Chỉ tài liệu đã về tới agent (fetch hoặc nằm trong kết quả
                # search) mới được trích: tài liệu chưa thấy là UNRETRIEVED.
                if doc.doc_id in seen_ids:
                    self.lines[doc.doc_id] = tuple(
                        n for n in (norm(l) for l in doc.body.splitlines()) if n
                    )

    def supports(self, doc_id, text: str) -> bool:
        """`text` là trích dẫn một dòng của `doc_id` VÀ agent đã thấy nó."""
        n = norm(text)
        if len(n) < MIN_QUOTE_CHARS or n not in self.observed:
            return False
        return any(n in line for line in self.lines.get(doc_id, ()))

    def source_of(self, text: str, prefer=None):
        """doc_id đã thấy chứa `text` trên một dòng; ưu tiên `prefer`."""
        if isinstance(prefer, str) and self.supports(prefer, text):
            return prefer
        return next((d for d in self.lines if self.supports(d, text)), None)

    def repair(self, text: str, prefer=None):
        """(substring, doc_id) ngắn nhất cần cắt để thành trích dẫn hợp lệ,
        hoặc None nếu không có cách cắt nào được bằng chứng đỡ."""
        for cand in trim_candidates(text):
            doc_id = self.source_of(cand[:MAX_QUOTE_CHARS], prefer)
            if doc_id is not None:
                return cand[:MAX_QUOTE_CHARS], doc_id
        return None
