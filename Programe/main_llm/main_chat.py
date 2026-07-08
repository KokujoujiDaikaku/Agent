import os

from history_chat import HistoryChat
from anaphora_resolution import Anaphora_Resolution
from main_rag.main_rag import MainRAG
from main_rag.deal_chromabd import DealChromaDB


class MainChat:
    def __init__(self):
        self.chat_sessions = HistoryChat()
        self.rag_client = DealChromaDB()
        self.collection = self.rag_client.collection

    def handle_conversational_query(self, collection, query: str, session_id: str, n_chunks: int = 3):
        """
        编排聊天对话中基于RAG的完整回答流程
        """
        # 基于是否传入session_id判断是否为新对话
        session_id = self.chat_sessions.start_new_session() if session_id is None else session_id

        # Step1: 拉取会话历史并准备进行上下文注入
        chat_log = self.chat_sessions.fetch_recent_messages(session_id=session_id)
        prior_messages = self.chat_sessions.prepare_history_for_model(chat_log)

        # Step2: 解析查询中的代词或不明确的引用
        anaphora_resolution = Anaphora_Resolution()
        refined_query = anaphora_resolution.rewrite_query_with_context(query=query, chat_log=prior_messages, client=collection)
        print(f"[消除指代后提问] {refined_query}")

        # Step3: 从向量数据库中检索相关知识
        research_results = MainRAG.run_semantic_query(collection=collection, query=refined_query, top_k=n_chunks)
        retrieved_text, citations = MainRAG.build_context_and_citations(results=research_results)

        # Step4: 基于检索到的内容生成答案
        answer = MainRAG.ask_openai(question=refined_query, context=retrieved_text)
        MainRAG.display_search_hits(results=citations)

        # Step5: 将用户输入和AI回复都保存至内存中
        self.chat_sessions.log_message(session_id=session_id, sender="user", message=query)
        self.chat_sessions.log_message(session_id=session_id, sender="assistant", message=answer)

    def chat_session(self, session_id):
        while True:
            query = input("【用户】 ").strip()
            if query.lower() == 'exit':
                print("退出对话。")
                break
            if not query:
                continue

            self.handle_conversational_query(
                collection=self.collection,
                query=query,
                session_id=session_id
            )

    def main_chat(self):
        """
        作为交互的最上层入口，用户可选择导入文件或开始对话
        """
        print("=" * 50)
        print("欢迎使用 RAG 对话系统")
        print("=" * 50)
        print("\n请选择操作：")
        print("1. 导入RAG文件（将文档内容入库）")
        print("2. 新增对话（开始新的对话会话）")
        print("3. 继续对话（输入已有的对话 ID 继续）")
        print("4. 推出系统")

        choice = input("\n请输入选项（1/2/3/4）：").strip()

        if choice == "1":
            # 选项1：导入RAG文件
            directory = input("请输入要导入的文件夹路径：").strip()
            if os.path.isdir(directory):
                print(f"\n开始导入文件加：{directory}")
                self.rag_client.ingest_folder(store=self.collection, directory=directory)
                print("\n✔ 文件导入完成！")
            else:
                print(f"\n❌ 指定的文件夹不存在：{directory}")

        elif choice == "2":
            # 选项2：新增对话
            print("\n开始新对话（输入'exit'可退出对话）")
            session_id = self.chat_sessions.start_new_session()
            print(f"会话ID：{session_id}\n")

            self.chat_session(session_id=session_id)

        elif choice == "3":
            # 选项3：继续对话
            session_id = input("请输入要继续的对话ID：").strip
            # 验证session_id是否存在
            if session_id not in self.chat_sessions.chat_sessions:
                print(f"❌ 未找到对应的会话ID：{session_id}")
                return

            print("\n继续对话（输入'exit'可退出对话）\n")

            self.chat_session(session_id=session_id)

        elif choice == "4":
            # 选项4：退出
            print("感谢使用，再见！")

        else:
            print("❌ 无效的选项，请重新运行并选择 1/2/3/4")


if __name__ == '__main__':
    main_chat = MainChat()
    main_chat.main_chat()
