import chromadb
from chromadb.utils import embedding_functions
import os

from read_file import ReadFile


class DealChromaDB:
    def __init__(self):
        # 持久化存储，在会话之间保存数据
        self.client = chromadb.PersistentClient(path="chroma_bd")
        # 定义嵌入函数，把文本转换为数学向量，支持相似度搜索
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"  # embedding模型
        )
        # 创建或获取集合
        self.collection = self.client.get_or_create_collection(
            name="documents_collection",  # 该向量知识库可存储文档
            embedding_function=sentence_transformer_ef
        )

    def build_knowledge_units(self, path: str):
        """
        该函数用于处理文档，将文档分块并添加元数据
        """
        try:
            file_text = ReadFile(path)
            raw = file_text.load_document()
            # 对文档进行切分
            segments = file_text.chunk_sentences(raw)
            # 提取文件名
            name = os.path.basename(path)

            # 为每个片段添加元数据字典
            metadata_records = [
                {"source_file": name, "segment_index": idx}
                for idx in range(len(segments))
            ]

            # 为每个片段添加唯一标识符
            unique_keys = [
                f"{name}_seg_{idx}"
                for idx in range(len(segments))
            ]

            return unique_keys, segments, metadata_records

        except Exception as e:
            print(f"Failed to process '{path}':{e}")

    def batch_insert_into_store(self, store, record_ids, contents, metadata_list):
        """
        优化批量写入向量库，分片批量提交，不一次性把所有数据塞进 store.add()，适配 ChromaDB 性能阈值，提升写入速度、降低内存占用
        """
        batch_size = 100  # 并发量，针对Chromdb吞吐量进行优化
        for start_idx in range(0, len(contents), batch_size):
            stop_idx = min(start_idx + batch_size, len(contents))  # 防止下标越界
            store.add(
                documents=contents[start_idx:stop_idx],  # 当前批次文本切片
                metadatas=metadata_list[start_idx:stop_idx],  # 当前批次每条分片的元数据
                ids=record_ids[start_idx:stop_idx]  # 当前批次分片唯一ID
            )

    def ingest_folder(self, store, directory: str):
        """
        遍历目标文件夹所有文件，逐个解析文件、切割文本分片，调用上面的批量函数入库，是对外调用的顶层入口
        """
        # 列表推导式遍历文件夹，只保留普通文件，过滤文件夹、软链接等非文件对象，得到完整文件路径列表
        entries = [
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))
        ]

        for path in entries:
            filename = os.path.basename(path)
            print(f"► Processing {filename} …")
            ids, contents, metadata_list = self.build_knowledge_units(path)
            if contents:
                self.batch_insert_into_store(store, ids, contents, metadata_list)
                print(f"✔ Loaded {len(contents)} chunks from {filename}")
