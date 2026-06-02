const { createApp } = Vue;

createApp({
    data() {
        return {
            messages: [],
            userInput: '',
            isLoading: false,
            isStreaming: false,
            streamText: '',
            abortController: null,
            selectedImage: null,
            selectedImageFile: null,
            activePanel: 'chat',
            connected: true,
            // Document management
            documents: [],
            documentsLoading: false,
            selectedDoc: null,
            isUploading: false,
            useLLMNaming: false,
            uploadStatus: null,
            // Auth
            token: localStorage.getItem('mragagent_token') || 'mragagent-admin-token-2026',
        };
    },
    mounted() {
        localStorage.setItem('mragagent_token', this.token);
        this.configureMarked();
        this.autoResize();
    },
    methods: {
        configureMarked() {
            if (typeof marked === 'undefined') return;
            marked.setOptions({
                highlight: function(code, lang) {
                    if (typeof hljs !== 'undefined') {
                        const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                        return hljs.highlight(code, { language }).value;
                    }
                    return code;
                },
                breaks: true,
                gfm: true,
            });
        },
        parseMarkdown(text) {
            if (typeof marked === 'undefined') return this.escapeHtml(text);
            return marked.parse(text || '');
        },
        escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text || '';
            return div.innerHTML;
        },
        authHeaders(extra = {}) {
            const headers = { ...extra };
            headers.Authorization = `Bearer ${this.token}`;
            return headers;
        },

        // ── Chat ─────────────────────────────────────────────────

        async sendMessage() {
            const text = this.userInput.trim();
            if (!text && !this.selectedImage) return;
            if (this.isStreaming) return;

            // 如果有图片，先上传获取服务端路径。不再在消息中下"请识别"指令，
            // 而是以中性方式传递图片路径，让 Agent 根据对话历史自行判断是否需要识图。
            let imagePath = null;
            if (this.selectedImageFile) {
                try {
                    const formData = new FormData();
                    formData.append('file', this.selectedImageFile);
                    const uploadResp = await fetch('/images/upload', {
                        method: 'POST',
                        headers: this.authHeaders({}),
                        body: formData,
                    });
                    if (uploadResp.ok) {
                        const uploadResult = await uploadResp.json();
                        imagePath = uploadResult.image_path;
                    } else {
                        this.messages.push({ role: 'bot', text: '【错误】图片上传失败，请重试' });
                        return;
                    }
                } catch (err) {
                    this.messages.push({ role: 'bot', text: '【错误】图片上传失败: ' + err.message });
                    return;
                }
            }

            this.messages.push({ role: 'user', text: text || '[图片]', image: this.selectedImage });
            this.userInput = '';
            this.selectedImage = null;
            this.selectedImageFile = null;

            this.isStreaming = true;
            this.abortController = new AbortController();

            // Create a placeholder bot message for streaming.
            // Fields: text (final answer), thinking (collapsible), toolCalls (list)
            this.messages.push({
                role: 'bot',
                text: '',
                thinking: '',
                toolCalls: [],
                _streaming: true,
            });
            const botMsgIdx = this.messages.length - 1;

            try {
                const requestBody = { message: text || '' };
                if (imagePath) requestBody.image_path = imagePath;

                const response = await fetch('/chat/stream', {
                    method: 'POST',
                    headers: this.authHeaders({ 'Content-Type': 'application/json' }),
                    body: JSON.stringify(requestBody),
                    signal: this.abortController.signal,
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });

                    // SSE events are separated by double newlines.
                    let eventEndIndex;
                    while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
                        const eventStr = buffer.slice(0, eventEndIndex);
                        buffer = buffer.slice(eventEndIndex + 2);

                        if (eventStr.startsWith('data: ')) {
                            const dataStr = eventStr.slice(6);
                            if (dataStr === '[DONE]') continue;
                            try {
                                const data = JSON.parse(dataStr);
                                const botMsg = this.messages[botMsgIdx];
                                if (data.type === 'content') {
                                    // Final answer text
                                    botMsg.text += data.text;
                                } else if (data.type === 'thinking') {
                                    // Internal reasoning — shown in collapsible section
                                    botMsg.thinking += data.text;
                                } else if (data.type === 'tool_call') {
                                    // Tool invocation record
                                    botMsg.toolCalls.push({ name: data.name });
                                } else if (data.type === 'error') {
                                    botMsg.text += '\n\n【错误】' + data.text;
                                }
                            } catch (_) {}
                        }
                    }
                }
            } catch (err) {
                if (err.name !== 'AbortError') {
                    this.messages[botMsgIdx].text = '【连接错误】' + err.message;
                }
            } finally {
                this.isStreaming = false;
                // Clean up streaming flag — thinking section will auto-collapse.
                const botMsg = this.messages[botMsgIdx];
                if (botMsg) {
                    delete botMsg._streaming;
                    if (!botMsg.text && !botMsg.thinking) botMsg.text = '[空回复]';
                }
                this.$nextTick(() => this.scrollToBottom());
            }
        },

        stopStreaming() {
            if (this.abortController) {
                this.abortController.abort();
            }
            this.isStreaming = false;
        },

        clearChat() {
            this.messages = [];
            this.streamText = '';
            this.isStreaming = false;
        },

        handleKeyDown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        },

        autoResize() {
            this.$nextTick(() => {
                const ta = this.$refs.textarea;
                if (!ta) return;
                ta.style.height = 'auto';
                ta.style.height = Math.min(ta.scrollHeight, 120) + 'px';
            });
        },

        scrollToBottom() {
            const container = this.$refs.chatContainer;
            if (container) {
                container.scrollTop = container.scrollHeight;
            }
        },

        // ── Image upload ─────────────────────────────────────────

        handleImageSelect(e) {
            const file = e.target.files[0];
            if (!file) return;
            this.selectedImageFile = file;
            const reader = new FileReader();
            reader.onload = (ev) => {
                this.selectedImage = ev.target.result;
            };
            reader.readAsDataURL(file);
        },

        // ── Document management ──────────────────────────────────

        handleDocSelect(e) {
            this.selectedDoc = e.target.files[0];
            this.uploadStatus = null;
        },

        async uploadDocument() {
            if (!this.selectedDoc) return;
            this.isUploading = true;
            this.uploadStatus = null;

            const formData = new FormData();
            formData.append('file', this.selectedDoc);
            if (this.useLLMNaming) {
                formData.append('use_llm_naming', 'true');
            }

            try {
                const response = await fetch('/documents/upload', {
                    method: 'POST',
                    headers: this.authHeaders({}),
                    body: formData,
                });
                const result = await response.json();
                if (response.ok) {
                    this.uploadStatus = { type: 'success', text: result.message };
                    this.selectedDoc = null;
                    this.loadDocuments();
                } else {
                    this.uploadStatus = { type: 'error', text: result.detail || '上传失败' };
                }
            } catch (err) {
                this.uploadStatus = { type: 'error', text: '网络错误: ' + err.message };
            } finally {
                this.isUploading = false;
            }
        },

        async loadDocuments() {
            this.documentsLoading = true;
            try {
                const response = await fetch('/documents', {
                    headers: this.authHeaders(),
                });
                const data = await response.json();
                this.documents = data.documents || [];
            } catch (_) {
                this.documents = [];
            } finally {
                this.documentsLoading = false;
            }
        },

        async deleteDocument(filename) {
            if (!confirm(`确定要删除 ${filename} 吗？这将同时删除其所有索引数据。`)) return;
            try {
                const response = await fetch(`/documents/${encodeURIComponent(filename)}`, {
                    method: 'DELETE',
                    headers: this.authHeaders(),
                });
                if (response.ok) {
                    this.loadDocuments();
                }
            } catch (_) {}
        },

        hasThinkingContent(msg) {
            return !!(msg.thinking || (msg.toolCalls && msg.toolCalls.length > 0));
        },

        formatSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        },
    },
}).mount('#app');
