"""
句子缓冲器 — 将 LLM token 流聚合为自然语言完整句子。

供 Unity 流式接口使用，确保 TTS 语音合成以完整句子为单位，
避免碎片化的短句导致语音不自然。

算法:
  - 强分隔符（。！？…\\n）：立即切割，形成自然断句点。
  - 弱分隔符（，；：、）：仅当缓冲区累计超过 20 字符时切割，
    避免 "大家好，" 这类短片段被单独切出。
"""


class SentenceBuffer:
    """
    将 LLM token 流聚合成完整句子。

    用法:
        buf = SentenceBuffer()
        async for chunk in llm.astream(...):
            for sentence in buf.add(chunk.content):
                await tts_synthesize(sentence)  # 完整句子 → TTS
        remaining = buf.flush()  # 处理末尾残留文字
    """

    # 强分隔符：遇到立即切割
    STRONG: set[str] = set("。！？…\n")

    # 弱分隔符：仅当累计长度 >= MIN_WEAK_LEN 时才切割
    WEAK: set[str] = set("，；：、")
    MIN_WEAK_LEN: int = 20

    def __init__(self) -> None:
        self._buf: str = ""

    def add(self, token: str) -> list[str]:
        """
        添加 token 到缓冲区，返回新完成的句子列表。

        返回的句子已包含标点符号。未触发切割时返回空列表。
        """
        self._buf += token
        sentences: list[str] = []

        while True:
            found = False
            for i, ch in enumerate(self._buf):
                if ch in self.STRONG:
                    sentence = self._buf[: i + 1].strip()
                    self._buf = self._buf[i + 1 :]
                    found = True
                    if sentence:
                        sentences.append(sentence)
                    break
                elif ch in self.WEAK and i >= self.MIN_WEAK_LEN:
                    sentence = self._buf[: i + 1].strip()
                    self._buf = self._buf[i + 1 :]
                    found = True
                    if sentence:
                        sentences.append(sentence)
                    break
            if not found:
                break

        return sentences

    def flush(self) -> str:
        """
        返回并清空缓冲区中的残留文字。

        在 LLM 流结束后调用，确保末尾无标点的文字也会被处理。
        返回空字符串表示缓冲区已空。
        """
        remaining = self._buf.strip()
        self._buf = ""
        return remaining
