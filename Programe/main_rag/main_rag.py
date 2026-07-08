from ..main_llm.main_llm import MainLLM


class MainRAG():
    @staticmethod
    def build_prompt(context: str, question: str) -> str:
        return f"""你是一个文档助手。仅通过提供的文档进行回答。若文档中不含有答案，则直接回复：“我没有相关的信息。”。

        Context:
        {context}

        Question:
        {question}

        Answer:"""

    @staticmethod
    def ask_openai(question: str, context: str) -> str:
        """
        发送prompt给OpenAI，并接收其返回值
        """
        prompt = MainRAG.build_prompt(context=context, question=question)

        try:
            llm_client = MainLLM()

            messages = [
                {"role": "system", "content": "You answer based strictly on the context provided."},
                {"role": "user", "content": prompt}
            ]

            print("--- 调用LLM ---")
            # 0.0最保守只输出事实，0.5平衡事实和表达，1.0最有创造性。对RAG来说，0.0到0.3比较合适，保证回答基于文档内容。
            response_text = llm_client.think(messages=messages, temperature=0.3)
            if response_text:
                print("\n\n--- 完整模型响应 ---")
                return response_text

        except ValueError as e:
            return e

    @staticmethod
    def run_semantic_query(collection, query: str, top_k: int = 2):
        """
        基于向量库执行语义相似度查询，返回相似度最高的 top_k 个文本分片
        """
        return collection.query(
            query_texts=[query],
            n_results=top_k
        )

    @staticmethod
    def build_context_and_citations(results):
        """
        接收向量库检索原始结果，输出两份内容：
        combined_text：拼接所有匹配分片，作为大模型输入上下文
        references：每条分片对应的来源引用（文件名 + 块编号），用于回答末尾标注参考文档
        """
        combined_text = "\n\n".join(results['documents'][0])
        references = [
            f"{meta['source']} (chunk {meta['chunk']})"
            for meta in results['metadatas'][0]
        ]
        return combined_text, references

    @staticmethod
    def display_search_hits(results):
        """
        搜索结果会显示相似度分数和来源文档，帮你判断检索质量
        """
        print("\nTop Matches\n" + "=" * 50)

        hits = results['documents'][0]
        metadata = results['metadatas'][0]
        scores = results['distances'][0]

        for idx in range(len(hits)):
            snippet = hits[idx]
            info = metadata[idx]
            score = scores[idx]

            print(f"\nMatch #{idx + 1}")
            print(f"From: {info['source']} — Chunk {info['chunk']}")
            print(f"Similarity Score: {1 - score:.2f} / 1.00")
            print(f"Excerpt: {snippet[:150]}...\n")
