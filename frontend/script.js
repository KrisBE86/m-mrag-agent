const { createApp } = Vue;
const CHAT_HISTORY_KEY = 'mragagent_chat_conversations_v1';

function createChatSessionId() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        return `web-${window.crypto.randomUUID()}`;
    }
    return `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function createConversation() {
    const now = Date.now();
    const id = createChatSessionId();
    return {
        id,
        sessionId: id,
        title: '新对话',
        messages: [],
        createdAt: now,
        updatedAt: now,
    };
}

createApp({
    data() {
        const initialConversation = createConversation();
        return {
            conversations: [initialConversation],
            activeConversationId: initialConversation.id,
            messages: initialConversation.messages,
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
            selectedDocs: [],
            uploadResults: [],
            isUploading: false,
            useLLMNaming: false,
            useVLMDescription: true,
            uploadStatus: null,
            urlInput: '',
            isImportingUrl: false,
            // Voice input / output
            isRecording: false,
            isTranscribing: false,
            mediaRecorder: null,
            recordingStream: null,
            audioChunks: [],
            voiceStatus: '',
            playingMessageIndex: null,
            ttsLoadingIndex: null,
            currentAudio: null,
            // Auth
            token: localStorage.getItem('mragagent_token') || 'mragagent-admin-token-2026',
            sessionId: initialConversation.sessionId,
        };
    },
    mounted() {
        localStorage.setItem('mragagent_token', this.token);
        this.loadConversations();
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
        blobToBase64(blob) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onloadend = () => {
                    const result = reader.result || '';
                    resolve(String(result).split(',')[1] || '');
                };
                reader.onerror = reject;
                reader.readAsDataURL(blob);
            });
        },
        stopCurrentAudio() {
            if (this.currentAudio) {
                this.currentAudio.pause();
                this.currentAudio.currentTime = 0;
                this.currentAudio = null;
            }
            this.playingMessageIndex = null;
        },
        createSessionId() {
            return createChatSessionId();
        },
        sanitizeMessages(messages) {
            return (messages || []).map(msg => {
                const copy = { ...msg };
                delete copy._streaming;
                return copy;
            });
        },
        conversationTitle(messages) {
            const firstUser = (messages || []).find(msg => msg.role === 'user' && msg.text);
            if (!firstUser) return '新对话';
            return firstUser.text.replace(/\s+/g, ' ').trim().slice(0, 24) || '图片对话';
        },
        activeConversation() {
            return this.conversations.find(conv => conv.id === this.activeConversationId);
        },
        persistConversations() {
            const payload = this.conversations
                .map(conv => ({
                    ...conv,
                    messages: this.sanitizeMessages(conv.messages),
                }))
                .sort((a, b) => b.updatedAt - a.updatedAt)
                .slice(0, 50);
            localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(payload));
        },
        loadConversations() {
            try {
                const raw = localStorage.getItem(CHAT_HISTORY_KEY);
                const parsed = raw ? JSON.parse(raw) : [];
                if (Array.isArray(parsed) && parsed.length > 0) {
                    this.conversations = parsed
                        .filter(conv => conv && conv.id && conv.sessionId)
                        .map(conv => ({
                            id: conv.id,
                            sessionId: conv.sessionId,
                            title: conv.title || this.conversationTitle(conv.messages),
                            messages: Array.isArray(conv.messages) ? conv.messages : [],
                            createdAt: conv.createdAt || Date.now(),
                            updatedAt: conv.updatedAt || conv.createdAt || Date.now(),
                        }))
                        .sort((a, b) => b.updatedAt - a.updatedAt);
                }
            } catch (_) {
                this.conversations = [createConversation()];
            }

            if (this.conversations.length === 0) {
                this.conversations = [createConversation()];
            }
            this.selectConversation(this.conversations[0].id);
        },
        persistActiveConversation() {
            const conv = this.activeConversation();
            if (!conv) return;
            conv.messages = this.messages;
            conv.title = this.conversationTitle(this.messages);
            conv.updatedAt = Date.now();
            this.conversations.sort((a, b) => b.updatedAt - a.updatedAt);
            this.persistConversations();
        },
        selectConversation(conversationId) {
            if (this.isStreaming) return;
            this.stopCurrentAudio();
            const conv = this.conversations.find(item => item.id === conversationId);
            if (!conv) return;
            this.activeConversationId = conv.id;
            this.sessionId = conv.sessionId;
            this.messages = conv.messages;
            this.selectedImage = null;
            this.selectedImageFile = null;
            this.userInput = '';
            this.$nextTick(() => this.scrollToBottom());
        },
        newChat() {
            if (this.isStreaming) this.stopStreaming();
            this.stopCurrentAudio();
            const conv = createConversation();
            this.conversations.unshift(conv);
            this.activeConversationId = conv.id;
            this.sessionId = conv.sessionId;
            this.messages = conv.messages;
            this.userInput = '';
            this.streamText = '';
            this.selectedImage = null;
            this.selectedImageFile = null;
            this.persistConversations();
            this.$nextTick(() => this.autoResize());
        },
        deleteConversation(conversationId) {
            if (this.isStreaming) return;
            const index = this.conversations.findIndex(conv => conv.id === conversationId);
            if (index === -1) return;
            this.conversations.splice(index, 1);
            if (this.conversations.length === 0) {
                this.conversations.push(createConversation());
            }
            if (this.activeConversationId === conversationId) {
                this.selectConversation(this.conversations[0].id);
            }
            this.persistConversations();
        },
        formatConversationTime(timestamp) {
            if (!timestamp) return '';
            const date = new Date(timestamp);
            const now = new Date();
            if (date.toDateString() === now.toDateString()) {
                return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
            }
            return date.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' });
        },

        // ── Chat ─────────────────────────────────────────────────

        async sendMessage() {
            const text = this.userInput.trim();
            if (!text && !this.selectedImage) return;
            if (this.isStreaming) return;
            this.stopCurrentAudio();

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
            this.persistActiveConversation();
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
                const requestBody = {
                    message: text || '',
                    session_id: this.sessionId,
                };
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
                this.persistActiveConversation();
                this.$nextTick(() => this.scrollToBottom());
            }
        },

        stopStreaming() {
            if (this.abortController) {
                this.abortController.abort();
            }
            this.isStreaming = false;
        },

        // ── Voice input / output ─────────────────────────────────

        async toggleRecording() {
            if (this.isRecording) {
                this.stopRecording();
                return;
            }
            await this.startRecording();
        },

        async startRecording() {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || typeof MediaRecorder === 'undefined') {
                this.voiceStatus = '当前浏览器不支持录音';
                return;
            }

            try {
                this.recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                this.audioChunks = [];
                const options = this.getMediaRecorderOptions();
                this.mediaRecorder = options ? new MediaRecorder(this.recordingStream, options) : new MediaRecorder(this.recordingStream);

                this.mediaRecorder.ondataavailable = (event) => {
                    if (event.data && event.data.size > 0) {
                        this.audioChunks.push(event.data);
                    }
                };
                this.mediaRecorder.onstop = () => this.handleRecordingStop();

                this.mediaRecorder.start();
                this.isRecording = true;
                this.voiceStatus = '正在录音，点击停止';
            } catch (err) {
                this.cleanupRecording();
                this.voiceStatus = '无法访问麦克风: ' + err.message;
            }
        },

        getMediaRecorderOptions() {
            const candidates = [
                'audio/webm;codecs=opus',
                'audio/webm',
                'audio/ogg;codecs=opus',
                'audio/mp4',
            ];
            const mimeType = candidates.find(type => MediaRecorder.isTypeSupported(type));
            return mimeType ? { mimeType } : null;
        },

        stopRecording() {
            if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
                this.mediaRecorder.stop();
            }
            this.isRecording = false;
            this.voiceStatus = '正在识别语音...';
        },

        cleanupRecording() {
            if (this.recordingStream) {
                this.recordingStream.getTracks().forEach(track => track.stop());
            }
            this.mediaRecorder = null;
            this.recordingStream = null;
            this.audioChunks = [];
            this.isRecording = false;
        },

        async handleRecordingStop() {
            const mimeType = this.mediaRecorder ? this.mediaRecorder.mimeType : 'audio/webm';
            const audioBlob = new Blob(this.audioChunks, { type: mimeType });
            this.cleanupRecording();

            if (!audioBlob.size) {
                this.voiceStatus = '没有录到音频';
                return;
            }

            this.isTranscribing = true;
            try {
                const audioBase64 = await this.blobToBase64(audioBlob);
                const response = await fetch('/unity/stt', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ audio_base64: audioBase64 }),
                });
                const result = await response.json();
                if (!response.ok) {
                    throw new Error(result.error || result.detail || '识别失败');
                }

                const text = (result.text || '').trim();
                if (text) {
                    this.userInput = this.userInput ? `${this.userInput} ${text}` : text;
                    this.voiceStatus = '';
                    this.$nextTick(() => this.autoResize());
                } else {
                    this.voiceStatus = '未识别到语音内容';
                }
            } catch (err) {
                this.voiceStatus = '语音识别失败: ' + err.message;
            } finally {
                this.isTranscribing = false;
            }
        },

        async toggleSpeech(msg, index) {
            if (this.playingMessageIndex === index) {
                this.stopCurrentAudio();
                return;
            }
            if (this.ttsLoadingIndex !== null || !msg.text) return;

            this.stopCurrentAudio();
            this.ttsLoadingIndex = index;
            try {
                const response = await fetch('/tts', {
                    method: 'POST',
                    headers: this.authHeaders({ 'Content-Type': 'application/json' }),
                    body: JSON.stringify({ text: msg.text }),
                });
                const result = await response.json();
                if (!response.ok) {
                    throw new Error(result.detail || '语音合成失败');
                }

                const audio = new Audio(`data:audio/wav;base64,${result.audio_base64}`);
                this.currentAudio = audio;
                this.playingMessageIndex = index;
                audio.onended = () => {
                    if (this.currentAudio === audio) this.stopCurrentAudio();
                };
                audio.onerror = () => {
                    if (this.currentAudio === audio) this.stopCurrentAudio();
                };
                await audio.play();
            } catch (err) {
                this.messages.push({ role: 'bot', text: '【错误】语音播放失败: ' + err.message });
            } finally {
                this.ttsLoadingIndex = null;
            }
        },

        clearChat() {
            this.newChat();
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
            // 将新选择的文件追加到已有列表，按文件名去重。
            const newFiles = Array.from(e.target.files);
            const skipped = [];
            for (const f of newFiles) {
                // 前端大小限制：单文件 ≤ 100MB。
                if (f.size > 100 * 1024 * 1024) {
                    skipped.push(`${f.name} (超过100MB)`);
                    continue;
                }
                if (this.selectedDocs.some(d => d.name === f.name)) {
                    skipped.push(f.name);
                    continue;
                }
                this.selectedDocs.push(f);
            }
            if (skipped.length > 0) {
                this.uploadStatus = {
                    type: 'error',
                    text: '已跳过重复或过大的文件: ' + skipped.join(', '),
                };
            } else {
                this.uploadStatus = null;
            }
            this.uploadResults = [];
            // 清空 input，允许再次选择同一文件。
            e.target.value = '';
        },

        removeSelectedDoc(index) {
            this.selectedDocs.splice(index, 1);
            this.uploadResults = [];
            this.uploadStatus = null;
        },

        async uploadDocument() {
            if (this.selectedDocs.length === 0) return;
            this.isUploading = true;
            this.uploadStatus = null;
            this.uploadResults = [];

            const formData = new FormData();
            for (const doc of this.selectedDocs) {
                formData.append('files', doc);
            }
            if (this.useLLMNaming) {
                formData.append('use_llm_naming', 'true');
            }
            formData.append('use_vlm_description', this.useVLMDescription ? 'true' : 'false');

            try {
                const response = await fetch('/documents/upload', {
                    method: 'POST',
                    headers: this.authHeaders({}),
                    body: formData,
                });
                const result = await response.json();
                if (response.ok) {
                    this.uploadResults = result.results || [];
                    const failedCount = this.uploadResults.filter(r => r.status === 'error').length;
                    if (failedCount === 0) {
                        this.uploadStatus = { type: 'success', text: result.summary };
                        this.selectedDocs = [];
                    } else if (failedCount === this.uploadResults.length) {
                        this.uploadStatus = { type: 'error', text: result.summary };
                    } else {
                        this.uploadStatus = { type: 'error', text: result.summary };
                    }
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

        async importUrlDocument() {
            const url = this.urlInput.trim();
            if (!url || this.isImportingUrl) return;
            this.isImportingUrl = true;
            this.uploadStatus = null;
            this.uploadResults = [];

            try {
                const response = await fetch('/documents/import-url', {
                    method: 'POST',
                    headers: this.authHeaders({ 'Content-Type': 'application/json' }),
                    body: JSON.stringify({
                        url,
                        use_llm_naming: this.useLLMNaming,
                        use_vlm_description: this.useVLMDescription,
                    }),
                });
                const result = await response.json();
                if (!response.ok) {
                    throw new Error(result.detail || 'URL 导入失败');
                }
                this.uploadResults = [{
                    filename: result.filename,
                    status: result.status,
                    message: result.message,
                }];
                this.uploadStatus = { type: 'success', text: result.message };
                this.urlInput = '';
                this.loadDocuments();
            } catch (err) {
                this.uploadStatus = { type: 'error', text: err.message };
            } finally {
                this.isImportingUrl = false;
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
                const result = await response.json().catch(() => ({}));
                if (response.ok) {
                    this.uploadStatus = { type: 'success', text: `已删除 ${filename}` };
                    this.loadDocuments();
                    return;
                }
                const message = result.detail || `删除 ${filename} 失败`;
                this.uploadStatus = { type: 'warning', text: message };
                alert(`警告：${message}`);
            } catch (err) {
                const message = `网络错误，删除请求未完成：${err.message}`;
                this.uploadStatus = { type: 'warning', text: message };
                alert(`警告：${message}`);
            }
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
