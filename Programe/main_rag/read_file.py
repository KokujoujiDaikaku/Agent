import os
import PyPDF2
import docx


class ReadFile():
    def __init__(self, file_path: str):
        self._file_path = file_path

    def read_text_file(self) -> str:
        """
        加载text文件
        """
        with open(self._file_path, 'r', encoding='utf-8') as fp:
            return fp.read()

    def read_pdf_file(self) -> str:
        """
        加载pdf文件
        """
        texts = []
        with open(self._file_path, 'rb') as fp:
            reader = PyPDF2.PdfReader(fp)
            for pg in reader.pages:
                # 确保pg非空
                page_txt = pg.extract_text() or ""
                texts.append(page_txt)
        # 合并文件内容
        return "\n".join(texts)

    def read_docx_file(self) -> str:
        """
        加载docx文件
        """
        doc = docx.Document(self._file_path)
        paras = [p.text for p in doc.paragraphs]
        return "\n".join(paras)

    def load_document(self) -> str:
        """
        根据文件的不同类型对文件进行加载
        目前支持的文件类型：txt、pdf、docx
        """
        _, extension = os.path.splitext(self._file_path)
        extension = extension.lower()

        if extension.endswith(".txt"):
            return self.read_text_file()
        elif extension.endswith(".pdf"):
            return self.read_pdf_file()
        elif extension.endswith(".docx"):
            return self.read_docx_file()
        else:
            raise ValueError(f"不支持的文件格式：{extension}")

    def chunk_sentences(self, text: str, max_length: int = 500) -> list[str]:
        """
        将字符串按照最低长度进行切分，仅在句子边界处换行
        """
        segments = text.replace('\n', ' ').split('.')
        blocks = []
        buffer = []
        buffer_len = 0

        for segment in segments:
            seg = segment.strip()
            if not seg:
                continue  # 空字符串跳过

            # 确保segment以结尾标点符号结束
            if not seg.endswith('.'):
                seg += '.'

            seg_len = len(seg)

            # 若单次增加的字段超过了最长字段，刷新buffer
            if buffer and buffer_len + seg_len > max_length:
                blocks.append(''.join(buffer))
                buffer = [seg]
                buffer_len = seg_len
            else:
                buffer.append(seg)
                buffer_len += seg_len

        # 添加所有剩余的句子
        if buffer:
            blocks.append(''.join(buffer))

        return blocks
