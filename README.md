# MRagAgent

这实际上是VR虚拟导游系统的后端部分。为了更好的测试，额外写了一个前端界面。以下是其单独使用时的说明。

多模态 RAG 智能体 — 面向文化遗产细粒度 POI 识别。支持用户上传照片，自动识别具体文物（佛像、壁龛、壁画等）并提供中文知识解答。

## 环境要求

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)（Python 包管理器）
- Docker 和 Docker Compose

## 快速启动

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

编辑 `.env`，填入你的 API 密钥（已内置默认值，大部分无需修改）。

关键配置项：
- `OPENAI_API_KEY` — DeepSeek API 密钥

### 3. 启动基础服务

```bash
docker compose up -d
```

启动 PostgreSQL 15、Redis 7、Milvus 2.5 等服务。

### 4. 运行

**命令行模式：**
```bash
uv run python main.py
```

**Web 服务模式：**
```bash
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
打开 http://localhost:8000 使用 Web 界面。

### 5. 停止服务

```bash
docker compose down
```
