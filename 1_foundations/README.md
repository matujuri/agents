---
title: career_conversation
app_file: app.py
sdk: gradio
sdk_version: 5.34.2
---

## 実行方法

- 前提: `.env` で `OPENAI_API_KEY`（必須）、`PUSHOVER_TOKEN`/`PUSHOVER_USER`（任意）を設定
- アプリ起動（Gradio）:

```bash
python 1_foundations/app.py
```

## ナレッジ DB（RAG）構築

- 取り込み対象: `1_foundations/me/knowledge/**/*.{txt,pdf}` に配置
- 再構築方法（どちらか）:

```bash
python 1_foundations/rebuild_knowledge.py
```

```bash
REBUILD_KNOWLEDGE=1 python 1_foundations/app.py
```

- 生成物: `1_foundations/me/knowledge_index.json`

## デプロイ

```bash
uv run gradio deploy
```
