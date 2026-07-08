from history_chat import HistoryChat
from anaphora_resolution import Anaphora_Resolution
from ..main_rag.main_rag import MainRAG


class main_chat():
    def __init__(self):
        self.chat_sessions = HistoryChat()

    def handle_conversational_query(self, collection, query: str, session_id: str, n_chunks: int = 3):
        """
        编排聊天对话中基于RAG的完整回答流程
        """

        # Step1: 拉取会话历史并准备进行上下文注入
        chat_log = self.chat_sessions.fetch_recent_messages(session_id=session_id)
        prior_messages = self.chat_sessions.prepare_history_for_model(chat_log)

        # Step2: 解析查询中的代词或不明确的引用
        refined_query = Anaphora_Resolution.rewrite_query_with_context(query=query, chat_log=prior_messages, client=collection)
        print(f"[消除指代后提问] {refined_query}")

        # Step3: 从向量数据库中检索相关知识
        research_results = MainRAG.run_semantic_query(collection=collection, query=refined_query, top_k=n_chunks)
        retrieved_text, citations = MainRAG.build_context_and_citations(results=research_results)

        # Step4: 基于检索到的内容生成答案
        answer = MainRAG.ask_openai(question=retrieved_text, context=citations)
        MainRAG.display_search_hits(results=answer)

        # Step5: 将用户输入和AI回复都保存至内存中
        self.chat_sessions.log_message(session_id=session_id, sender="user", message=query)
        self.chat_sessions.log_message(session_id=session_id, sender="assistant", message=answer)

    def main_chat():
        """
        作为交互的最上层入口，用户可选择导入文件或开始对话.
        """
