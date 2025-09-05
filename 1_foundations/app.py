from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
import gradio as gr
from pathlib import Path
import glob
from pypdf import PdfReader


load_dotenv(override=True)


def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        },
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_user_requirements(requirements, is_suitability):
    push(f"Recording {requirements} with is_suitability {is_suitability}")
    return {"recorded": "ok"}


def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}


record_user_details_json = {
    "name": "record_user_details",
    "description": "ユーザーが連絡希望でメールアドレスを提供したことを記録するためのツールです",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "このユーザーのメールアドレス"},
            "name": {"type": "string", "description": "ユーザー名（提供がある場合）"},
            "notes": {
                "type": "string",
                "description": "会話の文脈として記録しておきたい追加情報",
            },
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_user_requirements_json = {
    "name": "record_user_requirements",
    "description": "ユーザーが職業上の要望を提供したことを記録するためのツールです",
    "parameters": {
        "type": "object",
        "properties": {
            "requirements": {"type": "string", "description": "このユーザーの職業上の要望"},
            "is_suitability": {"type": "boolean", "description": "この要望が自身の経験と適切かどうか(true: 適切, false: 不適切)"},
        },
        "required": ["requirements"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "回答が分からず答えられなかった質問を必ず記録するためのツールです",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "答えられなかった質問内容"},
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_user_requirements_json},
    {"type": "function", "function": record_unknown_question_json},
]


class Me:
    def __init__(self):
        self.openai = OpenAI()
        self.name = "李静静"

        # RAG 用設定
        self.knowledge_dir = Path("me/knowledge")
        self.index_path = Path("me/knowledge_index.json")
        self.embedding_model = "text-embedding-3-small"
        self._ensure_knowledge_ready()

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append(
                {
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": tool_call.id,
                }
            )
        return results

    def system_prompt(self):
        system_prompt = (
            f"あなたは {self.name} として振る舞います。{self.name} のウェブサイト上での質問に回答してください。"
            "RAG により、関連情報は都度 system メッセージとして追加されます。追加コンテキストを優先的に活用し、"
            "事実はコンテキストに含まれる内容に限定してください。情報が無い場合は無理に推測せず、"
            "record_unknown_question ツールで未回答の質問を記録してください。会話が続く際は、メール連絡へ誘導し、"
            "メールアドレスを尋ねて record_user_details ツールで記録してください。"
            "職業上の要望を尋ねて、 自身の経験と適切かどうかを確認して record_user_requirements ツールで記録してください。"
            "常に丁寧かつ親しみやすい日本語で回答してください。"
        )
        return system_prompt

    def chat(self, message, history):
        # 追加コンテキスト（RAG）
        additional_context = self._retrieve_context(message, top_k=5)
        messages = [{"role": "system", "content": self.system_prompt()}]
        if additional_context:
            messages.append(
                {
                    "role": "system",
                    "content": "以下は参考用の追加コンテキストです。必要に応じて活用してください。\n\n"
                    + additional_context,
                }
            )
        messages.extend(history)
        messages.append({"role": "user", "content": message})
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model="gpt-5-nano", messages=messages, tools=tools
            )
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content

    # =============================
    # RAG: インデックス構築と検索
    # =============================
    def _ensure_knowledge_ready(self):
        # ディレクトリ作成
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        # インデックスが無い、または再構築指定時はビルド
        if os.getenv("REBUILD_KNOWLEDGE") == "1" or not self.index_path.exists():
            self._build_knowledge_index()

    def _chunk_text(self, text, max_chars=800, overlap=120):
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + max_chars, length)
            chunk = text[start:end]
            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)
            if end == length:
                break
            start = end - overlap
            if start < 0:
                start = 0
        return chunks

    def _list_knowledge_files(self):
        pattern_txt = str(self.knowledge_dir / "**" / "*.txt")
        pattern_pdf = str(self.knowledge_dir / "**" / "*.pdf")
        files = glob.glob(pattern_txt, recursive=True) + glob.glob(
            pattern_pdf, recursive=True
        )
        return sorted(files)

    def _read_text_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _read_pdf_file(self, file_path):
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception:
            return ""
        return text

    def _build_knowledge_index(self):
        files = self._list_knowledge_files()
        records = []
        for file_path in files:
            try:
                suffix = Path(file_path).suffix.lower()
                if suffix == ".txt":
                    text = self._read_text_file(file_path)
                elif suffix == ".pdf":
                    text = self._read_pdf_file(file_path)
                else:
                    continue
            except Exception:
                continue
            chunks = self._chunk_text(text)
            for idx, chunk in enumerate(chunks):
                records.append(
                    {
                        "id": f"{file_path}:{idx}",
                        "source": file_path,
                        "chunk_index": idx,
                        "text": chunk,
                    }
                )

        if not records:
            # 空でもインデックスファイルは作成
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"model": self.embedding_model, "chunks": []}, f, ensure_ascii=False
                )
            return

        # 埋め込み作成（バッチ）
        inputs = [r["text"] for r in records]
        emb = self.openai.embeddings.create(model=self.embedding_model, input=inputs)
        for r, e in zip(records, emb.data):
            r["embedding"] = e.embedding

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(
                {"model": self.embedding_model, "chunks": records},
                f,
                ensure_ascii=False,
            )

    def _load_knowledge_index(self):
        if not self.index_path.exists():
            return {"model": self.embedding_model, "chunks": []}
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"model": self.embedding_model, "chunks": []}

    def _cosine_similarity(self, a, b):
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        na = na**0.5
        nb = nb**0.5
        eps = 1e-12
        if na <= eps or nb <= eps:
            return 0.0
        return dot / (na * nb)

    def _retrieve_context(self, query, top_k=5):
        index = self._load_knowledge_index()
        chunks = index.get("chunks", [])
        if not chunks:
            return ""
        q_emb = (
            self.openai.embeddings.create(model=self.embedding_model, input=[query])
            .data[0]
            .embedding
        )
        scored = []
        for r in chunks:
            sim = self._cosine_similarity(q_emb, r.get("embedding", []))
            scored.append((sim, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [r for _, r in scored[:top_k]]
        # フォーマットして返却
        formatted = []
        for r in top:
            src = r.get("source", "")
            txt = r.get("text", "")
            formatted.append(f"[source: {src}]\n{txt}")
        return "\n\n---\n\n".join(formatted)


if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
