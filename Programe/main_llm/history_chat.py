import uuid
from datetime import datetime


class HistoryChat():
    def __init__(self):
        # 内存会话机制（生产环境可改为持久化存储）
        self.chat_sessions = {}

    def start_new_session(self) -> str:
        """
        为新对话创建一个新的唯一标识符
        """
        session_id = str(uuid.uuid4())
        self.chat_sessions[session_id] = []
        return session_id

    def log_message(self, session_id: str, sender: str, message: str):
        """
        将信息添加至历史聊天
        """
        if session_id not in self.chat_sessions:
            self.start_new_session()

        self.chat_sessions[session_id].append({
            "role": sender,
            "content": message,
            "timestamp": datetime.now().isoformat()
        })

    def fetch_recent_messages(self, session_id: str, limit: int = 5):
        """返回历史聊天中的最近历史消息"""
        msgs = self.chat_sessions.get(session_id, [])
        return msgs[-limit:]

    def prepare_history_for_model(self, messages: list) -> str:
        """将多条消息转换为单一格式化字符串"""
        return "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in messages
        )
