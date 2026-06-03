# MRagAgent

VR 虚拟导游系统的后端。支持两种接入方式：**Web 前端**（浏览器访问）和 **Unity 客户端**（VR 头盔语音+图片交互）。

多模态 RAG 智能体 — 面向文化遗产细粒度 POI 识别，支持上传照片自动识别具体文物（佛像、壁龛、壁画等）并提供中文知识解答。

## 环境要求

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)（Python 包管理器）
- Docker 和 Docker Compose

## 通用步骤

以下两种接入方式都需要先完成。

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

然后编辑 `.env`，填入你自己的 API 密钥等敏感信息。`.env.example` 中已内置所有配置项的默认值，关键需要修改的：

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | DeepSeek API 密钥（**必填**） |
| `DOUBAO_API_KEY` | 豆包视觉模型 API 密钥（图片识别用） |
| `STT_APPID` / `STT_ACCESS_KEY` | 火山引擎语音识别（Unity 接入需要） |
| `TTS_APPID` / `TTS_ACCESS_TOKEN` | 火山引擎语音合成（Unity 接入需要） |

其余配置（Milvus、PostgreSQL、Redis、嵌入模型等）已内置默认值，一般无需修改。

### 3. 启动基础服务

```bash
docker compose up -d
```

启动 PostgreSQL 15、Redis 7、Milvus 2.5 等服务。

### 4. 启动后端

```bash
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 停止

```bash
docker compose down
```

---

## 接入方式 A：Web 前端

打开 http://localhost:8000 ，在浏览器中直接使用。支持文字聊天、图片上传和文档管理。


---

## 接入方式 B：Unity 客户端

### 端点

| 方法 | 路径 | 用途 |
|------|------|------|
| POST | `/unity/chat` | 主对话（SSE 流式） |
| POST | `/unity/stt` | 语音识别（调试用） |

### 请求（/unity/chat）

```json
{
    "audio_base64": "<Base64 WAV 音频>",
    "image_base64": "<可选，Base64 JPEG/PNG 图片>"
}
```

### 响应（SSE 事件流）

| 事件 | 说明 | 主要字段 |
|------|------|---------|
| `thinking` | Agent 推理/工具调用 | `text`, `tool_name` |
| `token` | 逐字输出 | `content` |
| `text` | 完整句子 | `content`, `index` |
| `audio` | TTS 语音 (base64 WAV, 24kHz) | `content`, `index`, `audio_base64`, `sample_rate` |
| `done` | 对话结束 | `finish_reason` |

`text` 和 `audio` 通过 `index` 配对，可实现文字与语音同步播放。

### 数据流

```
Unity → POST /unity/chat {audio_base64, image_base64?}
      → STT 语音识别 → Agent 推理 → TTS 语音合成
      → SSE: token → text → audio → done
```
