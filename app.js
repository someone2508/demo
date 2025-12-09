// ============================================
// MODERN AI CHAT INTERFACE - JavaScript
// Feature-rich implementation with ES6+ features
// ============================================

'use strict';

// ============================================
// CONFIGURATION & CONSTANTS
// ============================================

const CONFIG = {
    OPENAI_API_URL: 'https://api.openai.com/v1/chat/completions',
    DEFAULT_MODEL: 'gpt-3.5-turbo',
    DEFAULT_TEMPERATURE: 0.7,
    DEFAULT_MAX_TOKENS: 2000,
    MAX_CHAR_LIMIT: 4000,
    STORAGE_KEYS: {
        CONVERSATIONS: 'ai_chat_conversations',
        SETTINGS: 'ai_chat_settings',
        CURRENT_CONVERSATION: 'ai_chat_current',
    },
    ANIMATION_DURATION: 300,
    DEBOUNCE_DELAY: 300,
};

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Utility class with helper methods
 */
class Utils {
    /**
     * Debounce function to limit execution rate
     */
    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Generate unique ID using timestamp and random number
     */
    static generateId() {
        return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Format date to readable string
     */
    static formatDate(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;

        // Less than 1 minute
        if (diff < 60000) return 'Just now';
        
        // Less than 1 hour
        if (diff < 3600000) {
            const minutes = Math.floor(diff / 60000);
            return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
        }

        // Less than 1 day
        if (diff < 86400000) {
            const hours = Math.floor(diff / 3600000);
            return `${hours} hour${hours > 1 ? 's' : ''} ago`;
        }

        // Less than 1 week
        if (diff < 604800000) {
            const days = Math.floor(diff / 86400000);
            return `${days} day${days > 1 ? 's' : ''} ago`;
        }

        return date.toLocaleDateString();
    }

    /**
     * Escape HTML to prevent XSS
     */
    static escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Copy text to clipboard
     */
    static async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (err) {
            console.error('Failed to copy:', err);
            return false;
        }
    }

    /**
     * Download text as file
     */
    static downloadFile(content, filename, mimeType = 'text/plain') {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    /**
     * Parse markdown and highlight code
     */
    static parseMarkdown(text) {
        if (typeof marked !== 'undefined') {
            // Configure marked
            marked.setOptions({
                highlight: function(code, lang) {
                    if (lang && hljs.getLanguage(lang)) {
                        try {
                            return hljs.highlight(code, { language: lang }).value;
                        } catch (err) {
                            console.error('Highlight error:', err);
                        }
                    }
                    return hljs.highlightAuto(code).value;
                },
                breaks: true,
                gfm: true,
            });
            return marked.parse(text);
        }
        return Utils.escapeHtml(text).replace(/\n/g, '<br>');
    }

    /**
     * Estimate token count (rough approximation)
     */
    static estimateTokens(text) {
        // Rough estimate: ~4 characters per token
        return Math.ceil(text.length / 4);
    }
}

// ============================================
// LOCAL STORAGE MANAGER
// ============================================

/**
 * Manages localStorage operations with error handling
 */
class StorageManager {
    /**
     * Save data to localStorage
     */
    static save(key, data) {
        try {
            const serialized = JSON.stringify(data);
            localStorage.setItem(key, serialized);
            return true;
        } catch (err) {
            console.error('Storage save error:', err);
            ToastManager.show('Failed to save data', 'error');
            return false;
        }
    }

    /**
     * Load data from localStorage
     */
    static load(key, defaultValue = null) {
        try {
            const serialized = localStorage.getItem(key);
            return serialized ? JSON.parse(serialized) : defaultValue;
        } catch (err) {
            console.error('Storage load error:', err);
            return defaultValue;
        }
    }

    /**
     * Remove data from localStorage
     */
    static remove(key) {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (err) {
            console.error('Storage remove error:', err);
            return false;
        }
    }

    /**
     * Clear all data from localStorage
     */
    static clear() {
        try {
            localStorage.clear();
            return true;
        } catch (err) {
            console.error('Storage clear error:', err);
            return false;
        }
    }
}

// ============================================
// TOAST NOTIFICATION MANAGER
// ============================================

/**
 * Manages toast notifications
 */
class ToastManager {
    static container = null;

    static init() {
        this.container = document.getElementById('toastContainer');
    }

    static show(message, type = 'info', duration = 3000) {
        if (!this.container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icons = {
            success: '✓',
            error: '✕',
            warning: '⚠',
            info: 'ℹ'
        };

        toast.innerHTML = `
            <span class="toast-icon">${icons[type] || icons.info}</span>
            <span class="toast-message">${Utils.escapeHtml(message)}</span>
            <button class="toast-close">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            </button>
        `;

        this.container.appendChild(toast);

        // Close button handler
        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => this.remove(toast));

        // Auto remove after duration
        if (duration > 0) {
            setTimeout(() => this.remove(toast), duration);
        }

        return toast;
    }

    static remove(toast) {
        if (!toast || !toast.parentElement) return;
        toast.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            if (toast.parentElement) {
                toast.parentElement.removeChild(toast);
            }
        }, 300);
    }

    static success(message, duration) {
        return this.show(message, 'success', duration);
    }

    static error(message, duration) {
        return this.show(message, 'error', duration);
    }

    static warning(message, duration) {
        return this.show(message, 'warning', duration);
    }

    static info(message, duration) {
        return this.show(message, 'info', duration);
    }
}

// ============================================
// SETTINGS MANAGER
// ============================================

/**
 * Manages application settings
 */
class SettingsManager {
    static defaults = {
        apiKey: '',
        model: CONFIG.DEFAULT_MODEL,
        temperature: CONFIG.DEFAULT_TEMPERATURE,
        maxTokens: CONFIG.DEFAULT_MAX_TOKENS,
        streaming: true,
        sound: true,
        theme: 'dark',
    };

    static settings = { ...this.defaults };

    static init() {
        this.load();
        this.applyTheme();
    }

    static load() {
        const saved = StorageManager.load(CONFIG.STORAGE_KEYS.SETTINGS);
        if (saved) {
            this.settings = { ...this.defaults, ...saved };
        }
    }

    static save() {
        return StorageManager.save(CONFIG.STORAGE_KEYS.SETTINGS, this.settings);
    }

    static get(key) {
        return this.settings[key];
    }

    static set(key, value) {
        this.settings[key] = value;
        this.save();
    }

    static reset() {
        this.settings = { ...this.defaults };
        this.save();
    }

    static applyTheme() {
        const theme = this.settings.theme;
        if (theme === 'auto') {
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            document.documentElement.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
        } else {
            document.documentElement.setAttribute('data-theme', theme);
        }
    }
}

// ============================================
// MESSAGE CLASS
// ============================================

/**
 * Represents a single message
 */
class Message {
    constructor(role, content, timestamp = Date.now()) {
        this.id = Utils.generateId();
        this.role = role; // 'user' or 'assistant'
        this.content = content;
        this.timestamp = timestamp;
    }

    /**
     * Convert message to API format
     */
    toAPI() {
        return {
            role: this.role,
            content: this.content,
        };
    }

    /**
     * Create message from plain object
     */
    static fromObject(obj) {
        return new Message(obj.role, obj.content, obj.timestamp);
    }
}

// ============================================
// CONVERSATION CLASS
// ============================================

/**
 * Represents a conversation thread
 */
class Conversation {
    constructor(title = 'New Conversation') {
        this.id = Utils.generateId();
        this.title = title;
        this.messages = [];
        this.createdAt = Date.now();
        this.updatedAt = Date.now();
    }

    /**
     * Add message to conversation
     */
    addMessage(role, content) {
        const message = new Message(role, content);
        this.messages.push(message);
        this.updatedAt = Date.now();
        
        // Auto-generate title from first user message
        if (this.messages.length === 1 && role === 'user') {
            this.title = this.generateTitle(content);
        }
        
        return message;
    }

    /**
     * Generate conversation title from first message
     */
    generateTitle(content) {
        const maxLength = 50;
        const cleaned = content.replace(/\s+/g, ' ').trim();
        return cleaned.length > maxLength 
            ? cleaned.substring(0, maxLength) + '...' 
            : cleaned;
    }

    /**
     * Get messages in API format
     */
    getMessagesForAPI() {
        return this.messages.map(msg => msg.toAPI());
    }

    /**
     * Get last message preview
     */
    getPreview() {
        if (this.messages.length === 0) return 'No messages yet';
        const lastMessage = this.messages[this.messages.length - 1];
        const preview = lastMessage.content.substring(0, 60);
        return preview.length < lastMessage.content.length ? preview + '...' : preview;
    }

    /**
     * Export conversation as text
     */
    exportAsText() {
        let text = `Conversation: ${this.title}\n`;
        text += `Created: ${new Date(this.createdAt).toLocaleString()}\n`;
        text += `${'='.repeat(60)}\n\n`;

        this.messages.forEach(msg => {
            text += `${msg.role.toUpperCase()}: ${msg.content}\n\n`;
        });

        return text;
    }

    /**
     * Export conversation as JSON
     */
    exportAsJSON() {
        return JSON.stringify(this, null, 2);
    }

    /**
     * Create conversation from plain object
     */
    static fromObject(obj) {
        const conv = new Conversation(obj.title);
        conv.id = obj.id;
        conv.createdAt = obj.createdAt;
        conv.updatedAt = obj.updatedAt;
        conv.messages = obj.messages.map(msg => Message.fromObject(msg));
        return conv;
    }
}

// ============================================
// OPENAI API CLIENT
// ============================================

/**
 * Handles OpenAI API communication
 */
class OpenAIClient {
    /**
     * Send message to OpenAI API
     */
    static async sendMessage(messages, options = {}) {
        const apiKey = SettingsManager.get('apiKey');
        
        if (!apiKey) {
            throw new Error('API key not configured. Please add your OpenAI API key in settings.');
        }

        const {
            model = SettingsManager.get('model'),
            temperature = SettingsManager.get('temperature'),
            maxTokens = SettingsManager.get('maxTokens'),
            stream = SettingsManager.get('streaming'),
        } = options;

        const requestBody = {
            model,
            messages,
            temperature,
            max_tokens: maxTokens,
            stream,
        };

        const response = await fetch(CONFIG.OPENAI_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`,
            },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.error?.message || `API request failed with status ${response.status}`);
        }

        return response;
    }

    /**
     * Handle streaming response
     */
    static async *streamResponse(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        try {
            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed || trimmed === 'data: [DONE]') continue;
                    
                    if (trimmed.startsWith('data: ')) {
                        try {
                            const json = JSON.parse(trimmed.slice(6));
                            const content = json.choices[0]?.delta?.content;
                            if (content) {
                                yield content;
                            }
                        } catch (err) {
                            console.error('Parse error:', err);
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
    }

    /**
     * Get non-streaming response
     */
    static async getNonStreamingResponse(response) {
        const data = await response.json();
        return data.choices[0]?.message?.content || '';
    }
}

// ============================================
// CONVERSATION MANAGER
// ============================================

/**
 * Manages all conversations
 */
class ConversationManager {
    static conversations = [];
    static currentConversation = null;

    static init() {
        this.load();
        this.render();
    }

    static load() {
        const saved = StorageManager.load(CONFIG.STORAGE_KEYS.CONVERSATIONS, []);
        this.conversations = saved.map(obj => Conversation.fromObject(obj));
        
        const currentId = StorageManager.load(CONFIG.STORAGE_KEYS.CURRENT_CONVERSATION);
        if (currentId) {
            this.currentConversation = this.conversations.find(c => c.id === currentId);
        }
    }

    static save() {
        StorageManager.save(CONFIG.STORAGE_KEYS.CONVERSATIONS, this.conversations);
        if (this.currentConversation) {
            StorageManager.save(CONFIG.STORAGE_KEYS.CURRENT_CONVERSATION, this.currentConversation.id);
        }
    }

    static createConversation(title) {
        const conversation = new Conversation(title);
        this.conversations.unshift(conversation);
        this.currentConversation = conversation;
        this.save();
        this.render();
        UI.renderMessages();
        return conversation;
    }

    static deleteConversation(id) {
        const index = this.conversations.findIndex(c => c.id === id);
        if (index !== -1) {
            this.conversations.splice(index, 1);
            
            if (this.currentConversation?.id === id) {
                this.currentConversation = this.conversations[0] || null;
            }
            
            this.save();
            this.render();
            UI.renderMessages();
        }
    }

    static selectConversation(id) {
        const conversation = this.conversations.find(c => c.id === id);
        if (conversation) {
            this.currentConversation = conversation;
            this.save();
            UI.renderMessages();
            this.render();
        }
    }

    static clearAll() {
        if (confirm('Are you sure you want to delete all conversations? This cannot be undone.')) {
            this.conversations = [];
            this.currentConversation = null;
            this.save();
            this.render();
            UI.renderMessages();
            ToastManager.success('All conversations deleted');
        }
    }

    static render() {
        const container = document.getElementById('conversationsList');
        
        if (this.conversations.length === 0) {
            container.innerHTML = `
                <div style="padding: 2rem 1rem; text-align: center; color: var(--text-secondary);">
                    <p>No conversations yet</p>
                    <p style="font-size: 0.875rem; margin-top: 0.5rem;">Start a new conversation to begin</p>
                </div>
            `;
            return;
        }

        container.innerHTML = this.conversations.map(conv => `
            <div class="conversation-item ${conv.id === this.currentConversation?.id ? 'active' : ''}" 
                 data-id="${conv.id}">
                <div class="conversation-item-title">${Utils.escapeHtml(conv.title)}</div>
                <div class="conversation-item-preview">${Utils.escapeHtml(conv.getPreview())}</div>
                <div class="conversation-item-meta">
                    <span>${Utils.formatDate(conv.updatedAt)}</span>
                    <span>${conv.messages.length} messages</span>
                </div>
            </div>
        `).join('');

        // Add click handlers
        container.querySelectorAll('.conversation-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const id = item.getAttribute('data-id');
                this.selectConversation(id);
            });
        });
    }
}

// ============================================
// UI CONTROLLER
// ============================================

/**
 * Controls UI updates and interactions
 */
class UI {
    static elements = {};
    static isStreaming = false;

    static init() {
        this.cacheElements();
        this.attachEventListeners();
        this.updateStatus('Ready');
    }

    static cacheElements() {
        this.elements = {
            // Sidebar
            sidebar: document.getElementById('sidebar'),
            toggleSidebar: document.getElementById('toggleSidebar'),
            newChatBtn: document.getElementById('newChatBtn'),
            conversationsList: document.getElementById('conversationsList'),
            clearAllBtn: document.getElementById('clearAllBtn'),
            
            // Chat
            chatTitle: document.getElementById('chatTitle'),
            chatSubtitle: document.getElementById('chatSubtitle'),
            messagesContainer: document.getElementById('messagesContainer'),
            welcomeScreen: document.getElementById('welcomeScreen'),
            deleteConversationBtn: document.getElementById('deleteConversationBtn'),
            exportBtn: document.getElementById('exportBtn'),
            
            // Input
            messageInput: document.getElementById('messageInput'),
            sendBtn: document.getElementById('sendBtn'),
            charCount: document.getElementById('charCount'),
            statusIndicator: document.getElementById('statusIndicator'),
            
            // Settings Modal
            settingsBtn: document.getElementById('settingsBtn'),
            settingsModal: document.getElementById('settingsModal'),
            closeSettingsBtn: document.getElementById('closeSettingsBtn'),
            saveSettingsBtn: document.getElementById('saveSettingsBtn'),
            resetSettingsBtn: document.getElementById('resetSettingsBtn'),
            apiKeyInput: document.getElementById('apiKeyInput'),
            modelSelect: document.getElementById('modelSelect'),
            temperatureSlider: document.getElementById('temperatureSlider'),
            temperatureValue: document.getElementById('temperatureValue'),
            maxTokensInput: document.getElementById('maxTokensInput'),
            streamingCheckbox: document.getElementById('streamingCheckbox'),
            soundCheckbox: document.getElementById('soundCheckbox'),
            themeSelect: document.getElementById('themeSelect'),
        };
    }

    static attachEventListeners() {
        // Sidebar
        this.elements.toggleSidebar.addEventListener('click', () => this.toggleSidebar());
        this.elements.newChatBtn.addEventListener('click', () => this.handleNewChat());
        this.elements.clearAllBtn.addEventListener('click', () => ConversationManager.clearAll());
        
        // Chat
        this.elements.deleteConversationBtn.addEventListener('click', () => this.handleDeleteConversation());
        this.elements.exportBtn.addEventListener('click', () => this.handleExport());
        
        // Input
        this.elements.messageInput.addEventListener('input', () => this.handleInputChange());
        this.elements.messageInput.addEventListener('keydown', (e) => this.handleInputKeydown(e));
        this.elements.sendBtn.addEventListener('click', () => this.handleSendMessage());
        
        // Settings
        this.elements.settingsBtn.addEventListener('click', () => this.openSettings());
        this.elements.closeSettingsBtn.addEventListener('click', () => this.closeSettings());
        this.elements.saveSettingsBtn.addEventListener('click', () => this.saveSettings());
        this.elements.resetSettingsBtn.addEventListener('click', () => this.resetSettings());
        this.elements.temperatureSlider.addEventListener('input', (e) => {
            this.elements.temperatureValue.textContent = e.target.value;
        });
        
        // Modal backdrop click
        this.elements.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.elements.settingsModal) {
                this.closeSettings();
            }
        });

        // Example prompts
        document.querySelectorAll('.example-prompt').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const prompt = e.target.getAttribute('data-prompt');
                this.elements.messageInput.value = prompt;
                this.handleInputChange();
                this.elements.messageInput.focus();
            });
        });

        // Auto-resize textarea
        this.elements.messageInput.addEventListener('input', () => {
            this.autoResizeTextarea();
        });
    }

    static toggleSidebar() {
        this.elements.sidebar.classList.toggle('hidden');
    }

    static handleNewChat() {
        ConversationManager.createConversation('New Conversation');
        ToastManager.success('New conversation created');
    }

    static handleDeleteConversation() {
        if (!ConversationManager.currentConversation) {
            ToastManager.warning('No conversation to delete');
            return;
        }

        if (confirm('Are you sure you want to delete this conversation?')) {
            ConversationManager.deleteConversation(ConversationManager.currentConversation.id);
            ToastManager.success('Conversation deleted');
        }
    }

    static handleExport() {
        if (!ConversationManager.currentConversation) {
            ToastManager.warning('No conversation to export');
            return;
        }

        const conv = ConversationManager.currentConversation;
        const filename = `conversation_${conv.id}_${Date.now()}.txt`;
        Utils.downloadFile(conv.exportAsText(), filename);
        ToastManager.success('Conversation exported');
    }

    static handleInputChange() {
        const value = this.elements.messageInput.value;
        const length = value.length;
        
        this.elements.charCount.textContent = `${length} / ${CONFIG.MAX_CHAR_LIMIT}`;
        this.elements.sendBtn.disabled = length === 0 || length > CONFIG.MAX_CHAR_LIMIT || this.isStreaming;
        
        if (length > CONFIG.MAX_CHAR_LIMIT) {
            this.elements.charCount.classList.add('error-text');
        } else {
            this.elements.charCount.classList.remove('error-text');
        }
    }

    static handleInputKeydown(e) {
        // Send on Enter (without Shift)
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!this.elements.sendBtn.disabled) {
                this.handleSendMessage();
            }
        }
    }

    static autoResizeTextarea() {
        const textarea = this.elements.messageInput;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }

    static async handleSendMessage() {
        const content = this.elements.messageInput.value.trim();
        
        if (!content) return;

        // Create conversation if none exists
        if (!ConversationManager.currentConversation) {
            ConversationManager.createConversation('New Conversation');
        }

        const conversation = ConversationManager.currentConversation;

        // Add user message
        conversation.addMessage('user', content);
        this.elements.messageInput.value = '';
        this.handleInputChange();
        this.autoResizeTextarea();
        
        // Save and render
        ConversationManager.save();
        this.renderMessages();
        ConversationManager.render();

        // Show typing indicator
        this.showTypingIndicator();
        this.updateStatus('Thinking...');
        this.isStreaming = true;

        try {
            const messages = conversation.getMessagesForAPI();
            const response = await OpenAIClient.sendMessage(messages);

            if (SettingsManager.get('streaming')) {
                await this.handleStreamingResponse(response, conversation);
            } else {
                await this.handleNonStreamingResponse(response, conversation);
            }

            ToastManager.success('Response received');
        } catch (err) {
            console.error('API Error:', err);
            ToastManager.error(err.message || 'Failed to get response');
            this.updateStatus('Error', 'error');
        } finally {
            this.hideTypingIndicator();
            this.isStreaming = false;
            this.handleInputChange();
            this.updateStatus('Ready');
        }
    }

    static async handleStreamingResponse(response, conversation) {
        let fullContent = '';
        const messageId = Utils.generateId();
        
        // Add empty assistant message
        const message = conversation.addMessage('assistant', '');
        ConversationManager.save();
        this.renderMessages();

        // Stream the response
        for await (const chunk of OpenAIClient.streamResponse(response)) {
            fullContent += chunk;
            message.content = fullContent;
            this.updateStreamingMessage(messageId, fullContent);
        }

        // Final save
        ConversationManager.save();
        ConversationManager.render();
        this.renderMessages();
    }

    static async handleNonStreamingResponse(response, conversation) {
        const content = await OpenAIClient.getNonStreamingResponse(response);
        conversation.addMessage('assistant', content);
        ConversationManager.save();
        ConversationManager.render();
        this.renderMessages();
    }

    static showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'message assistant';
        indicator.id = 'typing-indicator';
        indicator.innerHTML = `
            <div class="message-avatar">AI</div>
            <div class="message-content">
                <div class="message-body">
                    <div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            </div>
        `;
        this.elements.messagesContainer.appendChild(indicator);
        this.scrollToBottom();
    }

    static hideTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    static updateStreamingMessage(messageId, content) {
        const lastMessage = this.elements.messagesContainer.lastElementChild;
        if (lastMessage && lastMessage.classList.contains('assistant')) {
            const body = lastMessage.querySelector('.message-body');
            if (body) {
                body.innerHTML = Utils.parseMarkdown(content);
            }
        }
        this.scrollToBottom();
    }

    static renderMessages() {
        const conversation = ConversationManager.currentConversation;

        // Update header
        if (conversation) {
            this.elements.chatTitle.textContent = conversation.title;
            this.elements.chatSubtitle.textContent = `${conversation.messages.length} messages`;
            this.elements.welcomeScreen.style.display = 'none';
        } else {
            this.elements.chatTitle.textContent = 'New Conversation';
            this.elements.chatSubtitle.textContent = 'Start a conversation';
            this.elements.welcomeScreen.style.display = 'flex';
        }

        // Clear messages
        this.elements.messagesContainer.innerHTML = '';

        // Show welcome or messages
        if (!conversation || conversation.messages.length === 0) {
            this.elements.messagesContainer.appendChild(this.elements.welcomeScreen);
            return;
        }

        // Render messages
        conversation.messages.forEach((message, index) => {
            const messageEl = this.createMessageElement(message, index);
            this.elements.messagesContainer.appendChild(messageEl);
        });

        this.scrollToBottom();
    }

    static createMessageElement(message, index) {
        const div = document.createElement('div');
        div.className = `message ${message.role}`;
        
        const avatar = message.role === 'user' ? 'You' : 'AI';
        const roleName = message.role === 'user' ? 'You' : 'Assistant';

        div.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-header">
                    <span class="message-role">${roleName}</span>
                    <span class="message-time">${Utils.formatDate(message.timestamp)}</span>
                </div>
                <div class="message-body">
                    ${message.role === 'user' 
                        ? Utils.escapeHtml(message.content).replace(/\n/g, '<br>')
                        : Utils.parseMarkdown(message.content)
                    }
                </div>
                <div class="message-actions">
                    <button class="message-action-btn" data-action="copy" data-index="${index}">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                        Copy
                    </button>
                </div>
            </div>
        `;

        // Add action handlers
        const copyBtn = div.querySelector('[data-action="copy"]');
        copyBtn.addEventListener('click', async () => {
            const success = await Utils.copyToClipboard(message.content);
            if (success) {
                ToastManager.success('Copied to clipboard');
            } else {
                ToastManager.error('Failed to copy');
            }
        });

        return div;
    }

    static scrollToBottom() {
        this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight;
    }

    static updateStatus(text, type = 'success') {
        this.elements.statusIndicator.innerHTML = `
            <span class="status-dot ${type}"></span>
            ${text}
        `;
    }

    static openSettings() {
        // Load current settings
        this.elements.apiKeyInput.value = SettingsManager.get('apiKey');
        this.elements.modelSelect.value = SettingsManager.get('model');
        this.elements.temperatureSlider.value = SettingsManager.get('temperature');
        this.elements.temperatureValue.textContent = SettingsManager.get('temperature');
        this.elements.maxTokensInput.value = SettingsManager.get('maxTokens');
        this.elements.streamingCheckbox.checked = SettingsManager.get('streaming');
        this.elements.soundCheckbox.checked = SettingsManager.get('sound');
        this.elements.themeSelect.value = SettingsManager.get('theme');

        this.elements.settingsModal.classList.add('active');
    }

    static closeSettings() {
        this.elements.settingsModal.classList.remove('active');
    }

    static saveSettings() {
        SettingsManager.set('apiKey', this.elements.apiKeyInput.value.trim());
        SettingsManager.set('model', this.elements.modelSelect.value);
        SettingsManager.set('temperature', parseFloat(this.elements.temperatureSlider.value));
        SettingsManager.set('maxTokens', parseInt(this.elements.maxTokensInput.value));
        SettingsManager.set('streaming', this.elements.streamingCheckbox.checked);
        SettingsManager.set('sound', this.elements.soundCheckbox.checked);
        SettingsManager.set('theme', this.elements.themeSelect.value);

        SettingsManager.applyTheme();
        this.closeSettings();
        ToastManager.success('Settings saved');
    }

    static resetSettings() {
        if (confirm('Reset all settings to default?')) {
            SettingsManager.reset();
            this.openSettings(); // Reload form
            ToastManager.success('Settings reset to default');
        }
    }
}

// ============================================
// APPLICATION INITIALIZATION
// ============================================

/**
 * Main application class
 */
class App {
    static async init() {
        console.log('🚀 Initializing AI Chat Interface...');

        try {
            // Initialize managers
            ToastManager.init();
            SettingsManager.init();
            ConversationManager.init();
            UI.init();

            // Show welcome message
            ToastManager.info('Welcome to AI Chat Interface!');

            console.log('✅ Application initialized successfully');
        } catch (err) {
            console.error('❌ Initialization error:', err);
            ToastManager.error('Failed to initialize application');
        }
    }
}

// ============================================
// START APPLICATION
// ============================================

// Wait for DOM to be ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => App.init());
} else {
    App.init();
}

// Handle page visibility for notifications
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
        console.log('👋 Welcome back!');
    }
});

// Prevent accidental page unload
window.addEventListener('beforeunload', (e) => {
    if (UI.isStreaming) {
        e.preventDefault();
        e.returnValue = 'A message is being processed. Are you sure you want to leave?';
        return e.returnValue;
    }
});

// Export for debugging (optional)
window.DebugAPI = {
    conversations: ConversationManager.conversations,
    settings: SettingsManager.settings,
    currentConversation: () => ConversationManager.currentConversation,
    utils: Utils,
};

console.log('💡 Debug API available at window.DebugAPI');
