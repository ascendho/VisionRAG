// Core State
let uploadedDocs = [];
let isUploading = false;
let isChatting = false;
let selectedDocIds = null; // null = 全部文档；array of IDs = 当前选中的范围
let conversationHistory = []; // Multi-turn memory
const STREAM_SCROLL_THRESHOLD = 96;
const assistantMessageCache = new Map();
const messageMetaCache = new Map();
const retryableRequests = new Map();
const activeChat = {
  controller: null,
  msgId: null,
  answerText: '',
  renderFrame: null,
  autoScroll: true,
  requestSnapshot: null,
  loadId: null,
  wasStopped: false,
  stopRequested: false,
};
const SEND_BUTTON_HTML = `<svg class="w-5 h-5 mx-auto" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg>`;
const STOP_BUTTON_HTML = `<svg class="w-4 h-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><rect x="7" y="7" width="10" height="10" rx="2"></rect></svg><span class="text-[13px] font-semibold leading-none">停止</span>`;
const STOPPING_BUTTON_HTML = `<div class="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div><span class="text-[13px] font-semibold leading-none">停止中</span>`;

// Elements
const bottomFileInput = document.getElementById('bottomFileInput');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const inputContainer = document.getElementById('inputContainer');

const welcomeScreen = document.getElementById('welcomeScreen');
const chatHistory = document.getElementById('chatHistory');
const messagesContainer = document.getElementById('messagesContainer');
const appBody = document.getElementById('appBody');
const appMain = document.getElementById('appMain');

function cloneMessages(messages = []) {
  return messages.map((message) => ({ ...message }));
}

function getSelectedDocIdsSnapshot() {
  return Array.isArray(selectedDocIds) ? [...selectedDocIds] : null;
}

function cloneRequestSnapshot(snapshot) {
  return {
    query: snapshot.query,
    top_k: snapshot.top_k,
    selectedDocIds: Array.isArray(snapshot.selectedDocIds) ? [...snapshot.selectedDocIds] : null,
    chatHistory: cloneMessages(snapshot.chatHistory || []),
    targetMsgId: snapshot.targetMsgId || null,
  };
}

function buildRequestSnapshot(queryText, requestSnapshot = null) {
  if (requestSnapshot) return cloneRequestSnapshot(requestSnapshot);
  return {
    query: queryText,
    top_k: 5,
    selectedDocIds: getSelectedDocIdsSnapshot(),
    chatHistory: cloneMessages(conversationHistory),
    targetMsgId: null,
  };
}

function getMessageMeta(msgId) {
  return messageMetaCache.get(msgId) || { evidences: [], allCandidates: [], confidence: null, retrievalTiming: null };
}

function setMessageMeta(msgId, meta = {}) {
  messageMetaCache.set(msgId, {
    evidences: Array.isArray(meta.evidences) ? meta.evidences : [],
    allCandidates: Array.isArray(meta.allCandidates) ? meta.allCandidates : [],
    confidence: meta.confidence || null,
    retrievalTiming: meta.retrievalTiming || null,
  });
}

function buildConfidenceBadgeMarkup(confidence) {
  if (!confidence || !confidence.sample_size) return '';

  return `<span class="message-meta-badge message-meta-badge--neutral" title="已采用证据页 top-${confidence.sample_size} 的归一化 MaxSim 平均分">证据质量 ${Number(confidence.score || 0).toFixed(2)}</span>`;
}

function buildRetrievalTimingMarkup(retrievalTiming) {
  if (!retrievalTiming || typeof retrievalTiming.total_retrieval_ms !== 'number') return '';
  return `<span class="message-meta-badge message-meta-badge--neutral" title="Query Embedding ${retrievalTiming.query_embedding_ms} ms · Qdrant ${retrievalTiming.qdrant_query_ms} ms">检索 ${Math.round(retrievalTiming.total_retrieval_ms)} ms</span>`;
}

function buildMessageMetaMarkup(msgId, meta = {}) {
  const badges = [
    buildConfidenceBadgeMarkup(meta.confidence),
    buildRetrievalTimingMarkup(meta.retrievalTiming),
  ].filter(Boolean);

  if (!badges.length) return '';
  return `<div id="stream-meta-${msgId}" class="flex flex-wrap items-center gap-2 mb-3">${badges.join('')}</div>`;
}

function toggleEvidenceSection(msgId, forceExpanded = null) {
  const panel = document.getElementById(`evidence-body-${msgId}`);
  const button = document.getElementById(`evidence-toggle-${msgId}`);
  if (!panel || !button) return false;

  const shouldExpand = forceExpanded === null
    ? panel.style.display === 'none'
    : forceExpanded;
  const count = Number(button.dataset.count || 0);
  const collapsedMeta = button.dataset.collapsedMeta || `默认收起 · 共 ${count} 条依据`;
  const label = button.querySelector('[data-evidence-label]');
  const meta = button.querySelector('[data-evidence-meta]');
  const icon = button.querySelector('svg');

  panel.style.display = shouldExpand ? 'block' : 'none';
  button.setAttribute('aria-expanded', shouldExpand ? 'true' : 'false');
  if (label) label.textContent = shouldExpand ? '收起依据' : '查看依据';
  if (meta) meta.textContent = shouldExpand ? '已展开引用来源' : collapsedMeta;
  if (icon) icon.style.transform = shouldExpand ? 'rotate(180deg)' : 'rotate(0deg)';
  return shouldExpand;
}

function ensureEvidenceSectionVisible(msgId) {
  toggleEvidenceSection(msgId, true);
}

function decorateCitationHtml(msgId, html) {
  const evidenceIds = new Set(
    getMessageMeta(msgId).evidences
      .map((evidence) => evidence.evidence_id)
      .filter(Boolean)
  );

  return html.replace(/\[(E\d+)\]/g, (match, evidenceId) => {
    if (!evidenceIds.has(evidenceId)) {
      return `<span class="citation-token citation-token--missing" title="未找到 ${evidenceId} 对应的证据卡片">${match}</span>`;
    }

    return `<button type="button" class="citation-token" onclick="focusEvidenceCard('${msgId}', '${evidenceId}')" title="定位到 ${evidenceId} 对应的证据卡片">${match}</button>`;
  });
}

function renderMarkdownWithCitations(msgId, markdownText) {
  return decorateCitationHtml(msgId, marked.parse(markdownText));
}

function focusEvidenceCard(msgId, evidenceId) {
  ensureEvidenceSectionVisible(msgId);
  const card = document.getElementById(`evidence-${msgId}-${evidenceId}`);
  if (!card) return;

  requestAnimationFrame(() => {
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
    card.classList.remove('evidence-card-highlight');
    void card.offsetWidth;
    card.classList.add('evidence-card-highlight');
    setTimeout(() => card.classList.remove('evidence-card-highlight'), 1600);
  });
}

function applySuggestedQuestion(question) {
  if (!queryInput || !question) return;
  queryInput.value = question;
  queryInput.focus();
  if (isChatting) return;
  handleChat({ queryText: question });
}

function buildSuggestionChipMarkup(question) {
  const escapedQuestion = escapeHtml(question);
  return `<button class="suggestion-chip" onclick="applySuggestedQuestion(this.dataset.question)" data-question="${escapedQuestion}">${escapedQuestion}</button>`;
}

function buildSuggestedQuestionBody(state, questions = []) {
  if (state === 'loading') {
    return `
      <div class="suggestion-skeleton" aria-hidden="true">
        <div class="suggestion-skeleton-bar suggestion-skeleton-bar--long"></div>
        <div class="suggestion-skeleton-bar suggestion-skeleton-bar--mid"></div>
        <div class="suggestion-skeleton-bar suggestion-skeleton-bar--short"></div>
      </div>`;
  }

  if (state === 'error' || questions.length === 0) {
    return `<p class="document-ready-note">常见问题暂时还没生成出来，你也可以直接输入自己的问题开始提问。</p>`;
  }

  return `<div class="suggestion-grid">${questions.map(buildSuggestionChipMarkup).join('')}</div>`;
}

function buildDocumentReadyCardMarkup(documentInfo, options = {}) {
  const {
    state = 'loading',
    questions = [],
  } = options;
  const title = escapeHtml(documentInfo.document_name || '当前文档');
  const pageCount = Number(documentInfo.page_count || 0);
  const pageCopy = pageCount > 0 ? `共 ${pageCount} 页，` : '';
  const statusCopy = state === 'loading'
    ? '正在整理中...'
    : questions.length > 0
      ? `${questions.length} 个推荐问题`
      : '可直接输入问题';

  return `
      <div class="w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1">
        <svg class="w-6 h-6 outline-none" fill="#4285f4" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2.628c-.896 5.867-5.505 10.476-11.372 11.372 5.867.896 10.476 5.505 11.372 11.372.896-5.867 5.505-10.476 11.372-11.372-5.867-.896-10.476-5.505-11.372-11.372z"/>
        </svg>
      </div>
      <div class="flex flex-col items-start w-full">
        <section class="document-ready-card">
          <div class="document-ready-badge">文档已就绪</div>
          <div class="document-ready-header">
            <h3>${title} 已成功上传并建立索引</h3>
            <p>${pageCopy}可以直接点击下面的常见问题开始提问，也可以继续自由输入。</p>
          </div>
          <div class="document-ready-divider"></div>
          <div class="document-ready-section-head">
            <span class="document-ready-section-title">常见问题</span>
            <span class="document-ready-section-meta">${statusCopy}</span>
          </div>
          ${buildSuggestedQuestionBody(state, questions)}
        </section>
      </div>`;
}

function upsertDocumentReadyCard(documentInfo, options = {}) {
  const cardId = options.cardId || 'msg-' + Date.now();
  const existingEl = document.getElementById(cardId);
  const nextMarkup = buildDocumentReadyCardMarkup(documentInfo, options);

  welcomeScreen.classList.add('hidden');
  chatHistory.classList.remove('hidden');

  if (existingEl) {
    existingEl.className = 'flex gap-4 flex-row group';
    existingEl.innerHTML = nextMarkup;
  } else {
    chatHistory.innerHTML += `
      <div id="${cardId}" class="flex gap-4 flex-row group">
        ${nextMarkup}
      </div>`;
  }

  scrollToBottom();
  return cardId;
}

async function fetchSuggestedQuestions(documentInfo, cardId = null) {
  const targetCardId = upsertDocumentReadyCard(documentInfo, { cardId, state: 'loading' });

  try {
    const res = await fetch(`/api/rag/files/${encodeURIComponent(documentInfo.document_id)}/suggestions`);
    if (!res.ok) {
      upsertDocumentReadyCard(documentInfo, { cardId: targetCardId, state: 'error' });
      return;
    }
    const data = await res.json();
    if (Array.isArray(data.questions) && data.questions.length > 0) {
      upsertDocumentReadyCard(
        { ...documentInfo, document_name: data.document_name || documentInfo.document_name },
        { cardId: targetCardId, state: 'ready', questions: data.questions }
      );
      return;
    }

    upsertDocumentReadyCard(documentInfo, { cardId: targetCardId, state: 'error' });
  } catch (err) {
    console.error(err);
    upsertDocumentReadyCard(documentInfo, { cardId: targetCardId, state: 'error' });
  }
}

function getStreamingThinkingMarkup(msgId) {
  return `<div id="stream-thinking-${msgId}" class="flex items-center gap-1.5 py-1">
    <div class="w-1.5 h-1.5 bg-[#4285f4] opacity-70 rounded-full animate-bounce" style="animation-delay:-0.3s"></div>
    <div class="w-1.5 h-1.5 bg-[#4285f4] opacity-70 rounded-full animate-bounce" style="animation-delay:-0.15s"></div>
    <div class="w-1.5 h-1.5 bg-[#4285f4] opacity-70 rounded-full animate-bounce"></div>
    <span class="ml-1 text-[13px] text-[#80868b] dark:text-slate-400">正在生成回答...</span>
  </div>`;
}

function buildStreamingShellBody(msgId, evidences = [], allCandidates = [], meta = {}) {
  const metaHTML = buildMessageMetaMarkup(msgId, meta);
  const evHTML = _buildEvidenceCards(msgId, evidences, allCandidates);
  return `
      <div class="w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1">
        <svg class="w-6 h-6 outline-none" fill="#4285f4" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2.628c-.896 5.867-5.505 10.476-11.372 11.372 5.867.896 10.476 5.505 11.372 11.372.896-5.867 5.505-10.476 11.372-11.372-5.867-.896-10.476-5.505-11.372-11.372z"/>
        </svg>
      </div>
      <div class="flex flex-col gap-1 items-start w-full">
        ${metaHTML}
        <div id="stream-answer-${msgId}" class="text-[16px] leading-[1.8] text-[#1f1f1f] dark:text-slate-200 w-full markdown-body">
          ${getStreamingThinkingMarkup(msgId)}
        </div>
        <div id="stream-evidence-${msgId}">${evHTML}</div>
        <div id="stream-actions-${msgId}" class="flex gap-2 ml-2 opacity-0 group-hover:opacity-100 transition-opacity items-center mt-1"></div>
      </div>`;
}

function prepareRetryMessageShell(msgId) {
  const existingEl = document.getElementById(msgId);
  if (!existingEl) return false;

  existingEl.className = 'flex gap-4 flex-row group';
  setMessageMeta(msgId, {});
  existingEl.innerHTML = buildStreamingShellBody(msgId, [], [], {});
  assistantMessageCache.set(msgId, '');
  retryableRequests.delete(msgId);
  scrollToBottom();
  return true;
}

function isMessagesContainerNearBottom() {
  if (!messagesContainer) return true;
  const distanceFromBottom = messagesContainer.scrollHeight - messagesContainer.scrollTop - messagesContainer.clientHeight;
  return distanceFromBottom <= STREAM_SCROLL_THRESHOLD;
}

function setSendButtonMode(mode) {
  if (!sendBtn) return;

  sendBtn.disabled = mode === 'stopping';

  if (mode === 'stop') {
    sendBtn.className = 'h-10 px-4 text-white rounded-full shrink-0 flex items-center justify-center gap-2 bg-gradient-to-r from-[#4285f4] to-[#6a8dff] hover:from-[#2f77ec] hover:to-[#5b7eff] shadow-[0_10px_24px_rgba(66,133,244,0.28)] transition-all border border-white/20';
    sendBtn.innerHTML = STOP_BUTTON_HTML;
    sendBtn.title = '停止生成';
    sendBtn.setAttribute('aria-label', '停止生成');
    return;
  }

  if (mode === 'stopping') {
    sendBtn.className = 'h-10 px-4 text-white rounded-full shrink-0 flex items-center justify-center gap-2 bg-gradient-to-r from-[#7ba7f8] to-[#9ab8ff] transition-all border border-white/20 cursor-wait opacity-90 shadow-[0_10px_24px_rgba(66,133,244,0.18)]';
    sendBtn.innerHTML = STOPPING_BUTTON_HTML;
    sendBtn.title = '正在停止';
    sendBtn.setAttribute('aria-label', '正在停止');
    return;
  }

  sendBtn.className = 'p-2.5 text-white rounded-full shrink-0 flex items-center justify-center bg-gradient-to-r from-blue-500 to-blue-600 disabled:opacity-50 transition-all';
  sendBtn.innerHTML = SEND_BUTTON_HTML;
  sendBtn.title = '发送';
  sendBtn.setAttribute('aria-label', '发送');
}

function resetActiveChatState() {
  if (activeChat.renderFrame) {
    cancelAnimationFrame(activeChat.renderFrame);
  }

  activeChat.controller = null;
  activeChat.msgId = null;
  activeChat.answerText = '';
  activeChat.renderFrame = null;
  activeChat.autoScroll = true;
  activeChat.requestSnapshot = null;
  activeChat.loadId = null;
  activeChat.wasStopped = false;
  activeChat.stopRequested = false;
  setSendButtonMode('send');
}

function renderStreamingAnswer(msgId, answerText, options = {}) {
  const { showCursor = false, fallbackHtml = '' } = options;
  const answerEl = document.getElementById('stream-answer-' + msgId);
  if (!answerEl) return;

  const contentHtml = answerText
    ? renderMarkdownWithCitations(msgId, answerText)
    : fallbackHtml;
  const cursorHtml = showCursor
    ? `<span id="cursor-${msgId}" class="inline-block w-[2px] h-[1.1em] bg-current align-middle ml-0.5 animate-pulse opacity-70"></span>`
    : '';

  answerEl.innerHTML = contentHtml + cursorHtml;
}

function scheduleStreamRender() {
  if (!activeChat.msgId || activeChat.renderFrame) return;

  activeChat.renderFrame = requestAnimationFrame(() => {
    activeChat.renderFrame = null;
    renderStreamingAnswer(activeChat.msgId, activeChat.answerText, { showCursor: true });
    scrollToBottom();
  });
}

function flushStreamRender(options = {}) {
  if (!activeChat.msgId) return;

  if (activeChat.renderFrame) {
    cancelAnimationFrame(activeChat.renderFrame);
    activeChat.renderFrame = null;
  }

  renderStreamingAnswer(activeChat.msgId, activeChat.answerText, options);
}

async function copyAssistantMessage(msgId, button) {
  const text = assistantMessageCache.get(msgId) || '';
  if (!text) return;

  try {
    await navigator.clipboard.writeText(text);
  } catch (err) {
    console.error(err);
    return;
  }

  const icon = button?.querySelector('svg');
  if (!icon) return;
  const previousIcon = icon.innerHTML;
  icon.innerHTML = '<polyline points="20 6 9 17 4 12"></polyline>';
  setTimeout(() => {
    icon.innerHTML = previousIcon;
  }, 1500);
}

function renderAssistantActions(msgId, options = {}) {
  const {
    showCopy = true,
    retrySnapshot = null,
    stopped = false,
    statusLabel = stopped ? '已停止' : '',
  } = options;
  const actionsEl = document.getElementById('stream-actions-' + msgId);
  if (!actionsEl) return;

  if (retrySnapshot) {
    retryableRequests.set(msgId, cloneRequestSnapshot(retrySnapshot));
  } else {
    retryableRequests.delete(msgId);
  }

  const actions = [];
  if (statusLabel) {
    actions.push(`<span class="text-[11px] font-medium text-[#80868b] dark:text-slate-400 mr-1">${statusLabel}</span>`);
  }
  if (retrySnapshot) {
    actions.push(`<button onclick="retryAssistantMessage('${msgId}')" class="p-1.5 text-gray-500 dark:text-slate-400 hover:text-gray-700 dark:hover:text-slate-200 hover:bg-gray-100 dark:hover:bg-slate-800 rounded-full transition-colors flex items-center gap-1.5 text-xs font-medium" title="重新生成"><svg class="w-[15px] h-[15px]" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7"></path><path d="M3 3v6h6"></path></svg><span>重试</span></button>`);
  }
  if (showCopy && (assistantMessageCache.get(msgId) || '').trim()) {
    actions.push(`<button onclick="copyAssistantMessage('${msgId}', this)" class="p-1.5 text-gray-400 dark:text-slate-500 hover:text-gray-600 dark:hover:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-800 rounded-full transition-colors flex items-center gap-1.5 text-xs font-medium" title="复制结果"><svg class="w-[15px] h-[15px]" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg><span>复制</span></button>`);
  }

  actionsEl.innerHTML = actions.join('');
  actionsEl.style.opacity = statusLabel || retrySnapshot ? '1' : '';
}

function finalizeRetryableErrorState(snapshot, message) {
  if (!activeChat.msgId) {
    addAssistantMessage(message);
    return;
  }

  const retrySnapshot = cloneRequestSnapshot(snapshot || activeChat.requestSnapshot);
  retrySnapshot.targetMsgId = activeChat.msgId;
  const nextText = activeChat.answerText.trim()
    ? `${activeChat.answerText}\n\n${message}`
    : message;

  activeChat.answerText = nextText;
  flushStreamRender();
  renderStreamingAnswer(activeChat.msgId, nextText);
  assistantMessageCache.set(activeChat.msgId, nextText);
  renderAssistantActions(activeChat.msgId, {
    retrySnapshot,
    statusLabel: '可重试',
    showCopy: Boolean(nextText.trim()),
  });
  scrollToBottom();
}

function createStoppedMessageShell(snapshot) {
  const msgId = 'msg-' + Date.now();
  const retrySnapshot = cloneRequestSnapshot(snapshot);
  retrySnapshot.targetMsgId = msgId;
  chatHistory.innerHTML += `
    <div id="${msgId}" class="flex gap-4 flex-row group">
      <div class="w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1">
        <svg class="w-6 h-6 outline-none" fill="#4285f4" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2.628c-.896 5.867-5.505 10.476-11.372 11.372 5.867.896 10.476 5.505 11.372 11.372.896-5.867 5.505-10.476 11.372-11.372-5.867-.896-10.476-5.505-11.372-11.372z"/>
        </svg>
      </div>
      <div class="flex flex-col gap-1 items-start w-full">
        <div id="stream-answer-${msgId}" class="text-[16px] leading-[1.8] text-[#1f1f1f] dark:text-slate-200 w-full markdown-body"></div>
        <div id="stream-actions-${msgId}" class="flex gap-2 ml-2 opacity-0 group-hover:opacity-100 transition-opacity items-center mt-1"></div>
      </div>
    </div>`;

  assistantMessageCache.set(msgId, '');
  renderStreamingAnswer(msgId, '', {
    fallbackHtml: '<p class="mb-0 text-[14px] text-[#80868b] dark:text-slate-400">已停止生成，保留当前内容；可点击重试从头重新回答当前问题。</p>',
  });
  renderAssistantActions(msgId, { retrySnapshot, stopped: true, showCopy: false });
  scrollToBottom();
  return msgId;
}

function finalizeCompletedStream(queryText) {
  if (!activeChat.msgId) return;

  flushStreamRender();
  assistantMessageCache.set(activeChat.msgId, activeChat.answerText);
  renderAssistantActions(activeChat.msgId, { showCopy: true });
  scrollToBottom();

  appendConversationTurn(queryText, activeChat.answerText);
}

function appendConversationTurn(queryText, answerText) {
  conversationHistory.push({ role: 'user', content: queryText });
  conversationHistory.push({ role: 'assistant', content: answerText });
  if (conversationHistory.length > 20) {
    conversationHistory = conversationHistory.slice(conversationHistory.length - 20);
  }
}

function finalizeGuardedState(snapshot, payload = {}) {
  const queryText = snapshot?.query || activeChat.requestSnapshot?.query || '';
  const message = payload.message || '这个问题不属于当前文档问答范围，请换一个更贴近文档内容的问题。';

  if (!activeChat.msgId) {
    addAssistantMessage(message);
    appendConversationTurn(queryText, message);
    return;
  }

  activeChat.answerText = message;
  flushStreamRender();
  assistantMessageCache.set(activeChat.msgId, message);
  renderAssistantActions(activeChat.msgId, { showCopy: true });
  scrollToBottom();
  appendConversationTurn(queryText, message);
}

function finalizeStoppedStream(snapshot) {
  const retrySnapshot = cloneRequestSnapshot(snapshot);
  const fallbackHtml = '<p class="mb-0 text-[14px] text-[#80868b] dark:text-slate-400">已停止生成，保留当前内容；可点击重试从头重新回答当前问题。</p>';

  if (!activeChat.msgId) {
    createStoppedMessageShell(retrySnapshot);
    return;
  }

  retrySnapshot.targetMsgId = activeChat.msgId;
  flushStreamRender({ fallbackHtml });
  assistantMessageCache.set(activeChat.msgId, activeChat.answerText);
  renderAssistantActions(activeChat.msgId, {
    retrySnapshot,
    stopped: true,
    showCopy: Boolean(activeChat.answerText.trim()),
  });
  scrollToBottom();
}

function stopActiveChat() {
  if (!isChatting || !activeChat.controller || activeChat.stopRequested) return;
  activeChat.stopRequested = true;
  activeChat.wasStopped = true;
  setSendButtonMode('stopping');
  activeChat.controller.abort();
}

function retryAssistantMessage(msgId) {
  const snapshot = retryableRequests.get(msgId);
  if (!snapshot || isChatting) return;
  handleChat({
    fromHistory: true,
    queryText: snapshot.query,
    reuseVisibleUserMessage: true,
    requestSnapshot: snapshot,
  });
}

setSendButtonMode('send');

// Theme Management
function initTheme() {
  try {
    if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      // Force default to light as per requirements if not explicitly set, 
      // but let's actually just default to light unconditionally if not set.
    }
    
    if (localStorage.theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.theme = 'light';
    }
  } catch (e) {
    // Safari strict mode localStorage security error fallback
    document.documentElement.classList.remove('dark');
  }
}

function toggleTheme() {
  try {
    if (document.documentElement.classList.contains('dark')) {
      document.documentElement.classList.remove('dark');
      localStorage.theme = 'light';
    } else {
      document.documentElement.classList.add('dark');
      localStorage.theme = 'dark';
    }
  } catch (e) {
    document.documentElement.classList.toggle('dark');
  }
}

initTheme();

// Lightbox functions
function openImageModal(base64Src) {
  const modal = document.getElementById('imageModal');
  const modalImg = document.getElementById('modalImage');

  modalImg.src = base64Src;

  modal.classList.remove('hidden');
  modal.classList.add('flex');
  setTimeout(() => modal.classList.remove('opacity-0'), 10);
}

function closeImageModal() {
  const modal = document.getElementById('imageModal');
  modal.classList.add('opacity-0');
  setTimeout(() => {
    modal.classList.add('hidden');
    modal.classList.remove('flex');
  }, 300);
}

function getFileIcon(filename) {
  const ext = (filename || '').split('.').pop().toLowerCase();
  if (ext === 'pdf') {
    return `<div class="w-12 h-12 rounded-2xl bg-red-50 dark:bg-red-900/20 flex flex-col items-center justify-center shrink-0 border border-red-100 dark:border-red-900/50 text-red-500">
      <svg class="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
        <polyline points="14 2 14 8 20 8"></polyline>
        <path d="M9 15.5v-5h1.5a1.5 1.5 0 0 1 0 3H9"></path>
        <path d="M13 15.5v-5h1a2 2 0 0 1 0 4h-1"></path>
        <path d="M17 15.5v-5h2"></path>
        <path d="M17 13h1.5"></path>
      </svg>
    </div>`;
  } else if (['png', 'jpg', 'jpeg', 'webp'].includes(ext)) {
    return `<div class="w-12 h-12 rounded-2xl bg-blue-50 dark:bg-blue-900/20 flex flex-col items-center justify-center shrink-0 border border-blue-100 dark:border-blue-900/50 text-blue-500">
      <svg class="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
        <circle cx="8.5" cy="8.5" r="1.5"></circle>
        <polyline points="21 15 16 10 5 21"></polyline>
      </svg>
    </div>`;
  } else {
    return `<div class="w-12 h-12 rounded-2xl bg-gray-100 dark:bg-slate-700 flex flex-col items-center justify-center shrink-0 border border-gray-200 dark:border-slate-600 text-gray-400">
      <svg class="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
        <polyline points="13 2 13 9 20 9"></polyline>
      </svg>
    </div>`;
  }
}

function renderScopeBar() {
  const bar = document.getElementById('scopeBar');
  if (!bar) return;
  if (uploadedDocs.length === 0) {
    bar.classList.add('hidden');
    return;
  }
  bar.classList.remove('hidden');
  const allActive = selectedDocIds === null;
  // Only show "全部" chip when there are 2+ docs
  const chips = [
    ...(uploadedDocs.length >= 2 ? [`<button onclick="selectScope(null)" class="px-3 py-1 rounded-full text-[12px] font-medium border transition-colors ${
      allActive
        ? 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 border-blue-300 dark:border-blue-600'
        : 'bg-white dark:bg-slate-800 text-[#444746] dark:text-slate-400 border-gray-200 dark:border-slate-600 hover:border-blue-300'
    }">全部</button>`] : []),
    ...uploadedDocs.map(doc => {
      const isActive = Array.isArray(selectedDocIds) && selectedDocIds.includes(doc.document_id);
      const label = doc.document_name.length > 18 ? doc.document_name.slice(0, 16) + '…' : doc.document_name;
      const chipBase = isActive
        ? 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 border-blue-300 dark:border-blue-600'
        : 'bg-white dark:bg-slate-800 text-[#444746] dark:text-slate-400 border-gray-200 dark:border-slate-600 hover:border-blue-300';
      return `<span id="chip-${doc.document_id}" class="inline-flex items-center gap-1 pl-3 pr-1 py-1 rounded-full text-[12px] font-medium border transition-colors ${chipBase}">
        <span onclick="selectScope('${doc.document_id}')" title="${doc.document_name}" class="cursor-pointer">${label}</span>
        <button id="chip-del-${doc.document_id}" onclick="deleteScopeChip('${doc.document_id}', event)" title="删除文档" class="w-4 h-4 flex items-center justify-center rounded-full hover:bg-red-100 dark:hover:bg-red-900/40 hover:text-red-500 dark:hover:text-red-400 transition-colors text-current opacity-60 hover:opacity-100">
          <svg class="w-2.5 h-2.5" viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="2" y1="2" x2="8" y2="8"/><line x1="8" y1="2" x2="2" y2="8"/></svg>
        </button>
      </span>`;
    })
  ];
  bar.innerHTML = chips.join('');
}

function selectScope(docId) {
  if (docId === null) {
    selectedDocIds = null;
  } else {
    if (!Array.isArray(selectedDocIds)) {
      // coming from "全部" state — select just this one
      selectedDocIds = [docId];
    } else if (selectedDocIds.includes(docId)) {
      // toggle off
      const next = selectedDocIds.filter(id => id !== docId);
      selectedDocIds = next.length === 0 ? null : next;
    } else {
      // toggle on
      selectedDocIds = [...selectedDocIds, docId];
    }
    // if all docs are selected, treat as "全部"
    if (Array.isArray(selectedDocIds) && selectedDocIds.length === uploadedDocs.length) {
      selectedDocIds = null;
    }
  }
  renderScopeBar();
}

async function deleteScopeChip(docId, e) {
  e.stopPropagation();
  const btn = document.getElementById('chip-del-' + docId);
  if (btn) btn.innerHTML = `<div class="w-3 h-3 border border-current border-t-transparent rounded-full animate-spin"></div>`;
  try {
    const res = await fetch('/api/rag/files/' + docId, { method: 'DELETE' });
    if (res.ok) {
      uploadedDocs = uploadedDocs.filter(d => d.document_id !== docId);
      if (Array.isArray(selectedDocIds)) {
        const next = selectedDocIds.filter(id => id !== docId);
        selectedDocIds = next.length === 0 ? null : next;
      }
      updateInputState(uploadedDocs.length);
      renderScopeBar();
    } else {
      if (btn) btn.innerHTML = `<svg class="w-2.5 h-2.5" viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="2" y1="2" x2="8" y2="8"/><line x1="8" y1="2" x2="2" y2="8"/></svg>`;
      alert('删除失败');
    }
  } catch (err) {
    if (btn) btn.innerHTML = `<svg class="w-2.5 h-2.5" viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="2" y1="2" x2="8" y2="8"/><line x1="8" y1="2" x2="2" y2="8"/></svg>`;
    alert('网络错误');
  }
}

async function openFilesModal() {
  const modal = document.getElementById('filesModal');
  const content = document.getElementById('filesModalContent');
  const list = document.getElementById('filesList');
  
  modal.classList.remove('hidden');
  modal.classList.add('flex');
  
  list.innerHTML = `<div class="flex justify-center py-6">
    <div class="w-6 h-6 border-2 border-[#1a73e8] border-t-transparent rounded-full animate-spin"></div>
  </div>`;
  
  // Delay classes for transition
  setTimeout(() => {
    modal.classList.remove('opacity-0');
    content.classList.remove('scale-95', 'opacity-0');
  }, 10);
  
  try {
    const res = await fetch('/api/rag/files');
    const data = await res.json();
    uploadedDocs = data.files || [];
    updateInputState(uploadedDocs.length);
    renderScopeBar();

    if (data.files && data.files.length > 0) {
      list.innerHTML = `<div class="space-y-3">
        ${data.files.map(f => `
          <div class="flex items-center justify-between p-4 rounded-3xl bg-[#f8f9fa] dark:bg-slate-800 transition-colors">
            <div class="flex items-center gap-4 min-w-0 flex-1">
              ${getFileIcon(f.document_name)}
              <div class="min-w-0 flex-1">
                <h4 class="text-[16px] font-semibold text-[#1f1f1f] dark:text-slate-200 truncate" title="${f.document_name}">${f.document_name}</h4>
                <p class="text-[13px] text-[#80868b] dark:text-slate-400 mt-0.5">${f.page_count} pages • Indexed</p>
              </div>
            </div>
            <div class="flex items-center gap-1 shrink-0 ml-4">
              <a href="/api/rag/files/${f.document_id}/download" target="_blank" title="打开文件" class="p-2 hover:bg-blue-50 dark:hover:bg-blue-900/30 rounded-full text-[#80868b] hover:text-blue-500 transition-colors">
                <svg class="w-5 h-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
              </a>
              <button id="del-btn-${f.document_id}" onclick="deleteDocument('${f.document_id}')" title="删除文件" class="p-2 hover:bg-red-50 dark:hover:bg-red-900/30 rounded-full text-[#80868b] hover:text-[#d93025] dark:hover:text-red-400 transition-colors">
                <svg class="w-5 h-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"></path><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path></svg>
              </button>
            </div>
          </div>
        `).join('')}
      </div>`;
    } else {
      list.innerHTML = `<div class="text-center py-8 text-[#80868b] dark:text-slate-400">
        <svg class="w-12 h-12 mx-auto text-gray-300 dark:text-slate-600 mb-3" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path><polyline points="13 2 13 9 20 9"></polyline></svg>
        <p class="text-sm font-medium text-[#444746] dark:text-slate-300">还没上传文件</p>
        <p class="text-xs mt-1">请上传文档开始使用</p>
      </div>`;
    }
  } catch (err) {
    list.innerHTML = `<div class="text-center py-6 text-red-500 text-sm">加载失败: 网络错误</div>`;
  }
}

async function deleteDocument(docId) {
  const btn = document.getElementById('del-btn-' + docId);
  if (btn) btn.innerHTML = `<div class="w-4 h-4 border-2 border-red-500 border-t-transparent rounded-full animate-spin"></div>`;
  try {
    const res = await fetch('/api/rag/files/' + docId, { method: 'DELETE' });
    if (res.ok) {
      openFilesModal(); // re-fetch and render
    } else {
      alert('删除失败');
    }
  } catch(err) {
    alert('网络错误');
  }
}

function closeFilesModal() {
  const modal = document.getElementById('filesModal');
  const content = document.getElementById('filesModalContent');
  modal.classList.add('opacity-0');
  content.classList.add('scale-95', 'opacity-0');
  setTimeout(() => {
    modal.classList.add('hidden');
    modal.classList.remove('flex');
  }, 300);
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closeImageModal();
    closeFilesModal();
  }
});

// Handle Uploads
function showUploadingAnimation(text) {
  welcomeScreen.classList.add("hidden");
  chatHistory.classList.remove("hidden");
  inputContainer.classList.remove("-translate-y-[25vh]", "md:-translate-y-[30vh]");
  
  const id = 'load-' + Date.now();
  chatHistory.innerHTML += `
    <div id="${id}" class="flex gap-4 flex-row">
      <div class="w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1">
        <svg class="w-6 h-6 outline-none" fill="url(#sparkle-loading-grad1)" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="sparkle-loading-grad1" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#4285f4" />
              <stop offset="50%" stop-color="#9b72cb" />
              <stop offset="100%" stop-color="#d96570" />
            </linearGradient>
          </defs>
          <path d="M12 2.628c-.896 5.867-5.505 10.476-11.372 11.372 5.867.896 10.476 5.505 11.372 11.372.896-5.867 5.505-10.476 11.372-11.372-5.867-.896-10.476-5.505-11.372-11.372z"/>
        </svg>
      </div>
      <div class="flex flex-col justify-center gap-1.5 mt-1">
        <div class="px-5 flex items-center gap-1.5 h-[32px] relative">
          <div class="w-2 h-2 bg-[#4285f4] opacity-90 rounded-full animate-bounce" style="animation-delay: -0.3s"></div>
          <div class="w-2 h-2 bg-[#4285f4] opacity-90 rounded-full animate-bounce" style="animation-delay: -0.15s"></div>
          <div class="w-2 h-2 bg-[#4285f4] opacity-90 rounded-full animate-bounce"></div>
          <span class="ml-3 text-[14px] font-medium text-[#444746] dark:text-slate-300">${text || '正在提取与建立索引...'}</span>
        </div>
        <div class="mx-5 h-1 w-48 bg-gray-200 dark:bg-slate-700 rounded-full overflow-hidden">
          <div id="upload-progress-fill-${id}" class="h-full bg-blue-500 rounded-full transition-all duration-200" style="width: 0%"></div>
        </div>
      </div>
    </div>
  `;
  requestAnimationFrame(() => scrollToBottom());
  return id;
}

function updateUploadProgress(loadId, ratio) {
  const fill = document.getElementById('upload-progress-fill-' + loadId);
  if (fill) fill.style.width = Math.min(100, Math.round(ratio * 100)) + '%';
}

async function handleUpload(file) {
  if (!file || isUploading) return;
  isUploading = true;
  
  const loadId = showUploadingAnimation("正在解析并建立特征索引...");
  
  const formData = new FormData();
  formData.append('file', file);

  try {
    const data = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/rag/upload');
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) updateUploadProgress(loadId, e.loaded / e.total);
      });
      xhr.addEventListener('load', () => {
        try { resolve({ status: xhr.status, body: JSON.parse(xhr.responseText) }); }
        catch { resolve({ status: xhr.status, body: { detail: xhr.responseText } }); }
      });
      xhr.addEventListener('error', () => reject(new Error('网络错误')));
      xhr.send(formData);
    });

    // Snap progress to 100% before removing the loader
    updateUploadProgress(loadId, 1);

    removeLoading(loadId);
    if (data.status === 200) {
      uploadedDocs.push(data.body);
      if(typeof updateInputState === 'function') updateInputState(uploadedDocs.length);
      renderScopeBar();
      // 若"已加载文档"弹窗此时处于打开状态，自动刷新列表无需用户手动关闭重开
      const _fm = document.getElementById('filesModal');
      if (_fm && !_fm.classList.contains('hidden')) openFilesModal();
      fetchSuggestedQuestions(data.body);
    } else {
      addAssistantMessage(`❌ 上传失败: ${data.body.detail}`);
    }
  } catch (err) {
    removeLoading(loadId);
    addAssistantMessage(`❌ 网络错误，上传失败。`);
  } finally {
    isUploading = false;
    bottomFileInput.value = "";
  }
}

if (bottomFileInput) {
  bottomFileInput.addEventListener('change', (e) => handleUpload(e.target.files[0]));
}

// Handle Chat
async function handleChat(options = {}) {
  const {
    fromHistory = false,
    queryText = queryInput.value.trim(),
    reuseVisibleUserMessage = false,
    requestSnapshot = null,
  } = options;
  const text = queryText.trim();
  if (!text || isChatting) return;
  
  if (!fromHistory) {
    history.pushState({view: 'chat'}, '', '#chat');
  }

  welcomeScreen.classList.add("hidden");
  chatHistory.classList.remove("hidden");
  inputContainer.classList.remove("-translate-y-[25vh]", "md:-translate-y-[30vh]");

  if (!reuseVisibleUserMessage) {
    addUserMessage(text);
  }
  if (!queryText || queryText === queryInput.value.trim()) {
    queryInput.value = "";
  }
  
  if (uploadedDocs.length === 0) {
    addAssistantMessage("🤖 提示：请先点击下方上传一份文档～");
    return;
  }
  
  isChatting = true;
  activeChat.controller = new AbortController();
  activeChat.msgId = requestSnapshot?.targetMsgId || null;
  activeChat.answerText = '';
  activeChat.autoScroll = true;
  activeChat.requestSnapshot = buildRequestSnapshot(text, requestSnapshot);
  if (activeChat.msgId && !prepareRetryMessageShell(activeChat.msgId)) {
    activeChat.msgId = null;
    activeChat.requestSnapshot.targetMsgId = null;
  }
  activeChat.loadId = activeChat.msgId ? null : showLoading();
  activeChat.wasStopped = false;
  setSendButtonMode('stop');

  try {
    const res = await fetch('/api/rag/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: activeChat.controller.signal,
      body: JSON.stringify({
        query: activeChat.requestSnapshot.query,
        top_k: activeChat.requestSnapshot.top_k,
        chat_history: activeChat.requestSnapshot.chatHistory,
        ...(activeChat.requestSnapshot.selectedDocIds ? { document_ids: activeChat.requestSnapshot.selectedDocIds } : {})
      })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: res.statusText }));
      removeLoading(activeChat.loadId);
      activeChat.loadId = null;
      finalizeRetryableErrorState(activeChat.requestSnapshot, `❌ Error: ${errData.detail}`);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let completed = false;
    let streamFinished = false;

    while (!streamFinished) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // retain incomplete last line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();

        if (raw === '[DONE]') {
          completed = true;
          finalizeCompletedStream(text);
          streamFinished = true;
          break;
        }

        let event;
        try { event = JSON.parse(raw); } catch { continue; }

        if (event.type === 'guarded') {
          removeLoading(activeChat.loadId);
          activeChat.loadId = null;
          finalizeGuardedState(activeChat.requestSnapshot, event.data);
          return;
        }

        if (event.type === 'error') {
          removeLoading(activeChat.loadId);
          activeChat.loadId = null;
          finalizeRetryableErrorState(activeChat.requestSnapshot, event.data);
          return;
        }

        if (event.type === 'evidences') {
          removeLoading(activeChat.loadId);
          activeChat.loadId = null;
          if (!activeChat.msgId) {
            activeChat.msgId = 'msg-' + Date.now();
            activeChat.requestSnapshot.targetMsgId = activeChat.msgId;
          }
          if (event.data.retrieval_timing) {
            console.info('[Retrieval timing]', event.data.retrieval_timing);
          }
          _insertStreamingShell(activeChat.msgId, event.data.evidences, event.data.all_candidates, {
            confidence: event.data.confidence,
            retrievalTiming: event.data.retrieval_timing,
          });
        }

        if (event.type === 'token' && activeChat.msgId) {
          activeChat.answerText += event.data;
          scheduleStreamRender();
        }
      }
    }

    if (!completed && activeChat.msgId && !activeChat.wasStopped) {
      finalizeCompletedStream(text);
    }
  } catch (err) {
    removeLoading(activeChat.loadId);
    activeChat.loadId = null;

    if (err?.name === 'AbortError' || activeChat.wasStopped) {
      finalizeStoppedStream(activeChat.requestSnapshot);
      return;
    }

    finalizeRetryableErrorState(activeChat.requestSnapshot, '❌ 网络错误，请重试。');
  } finally {
    isChatting = false;
    resetActiveChatState();
  }
}

if (sendBtn) {
  sendBtn.addEventListener('click', () => {
    if (isChatting) {
      stopActiveChat();
      return;
    }
    handleChat();
  });
}
if (queryInput) {
  queryInput.addEventListener('keydown', (e) => {
    if (e.isComposing || e.keyCode === 229) return;
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleChat();
    }
  });
}

if (messagesContainer) {
  messagesContainer.addEventListener('scroll', () => {
    if (!isChatting) return;
    activeChat.autoScroll = isMessagesContainerNearBottom();
  });
}

function resetUI(fromHistory = false) {
  if (fromHistory !== true && history.state && history.state.view === 'chat') {
    history.pushState({view: 'home'}, '', window.location.pathname);
  }
  welcomeScreen.classList.remove("hidden");
  chatHistory.classList.add("hidden");
  inputContainer.classList.add("-translate-y-[25vh]", "md:-translate-y-[30vh]");
  
  queryInput.value = "";
  isChatting = false;
  isUploading = false;
  conversationHistory = [];
}

window.addEventListener('popstate', (e) => {
  if (e.state && e.state.view === 'chat') {
    welcomeScreen.classList.add("hidden");
    chatHistory.classList.remove("hidden");
    inputContainer.classList.remove("-translate-y-[25vh]", "md:-translate-y-[30vh]");
  } else {
    resetUI(true);
  }
});

// UI Helpers
function scrollToBottom() {
  if (!messagesContainer) return;
  if (isChatting && !activeChat.autoScroll) return;
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function addUserMessage(text) {
  const escapedText = text.replace(/"/g, '&quot;').replace(/'/g, '&apos;').replace(/\n/g, '\\n');
  chatHistory.innerHTML += `
    <div class="flex gap-4 flex-row-reverse group">
      <div class="w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1 bg-gradient-to-tr from-purple-500 to-fuchsia-500 shadow-md">
        <span class="text-[11px] font-bold text-white tracking-wider">ME</span>
      </div>
      <div class="flex flex-col gap-1 items-end max-w-[80%]">
        <div class="px-5 py-3.5 text-[16px] leading-[1.75] bg-[#f0f4f9] dark:bg-slate-800 text-[#1f1f1f] dark:text-slate-200 rounded-[24px] rounded-tr-sm">
          ${text}
        </div>
        <!-- Action Bar: Visible only on hover -->
        <div class="flex gap-2 mr-2 opacity-0 group-hover:opacity-100 transition-opacity items-center">
          <button onclick="navigator.clipboard.writeText('${escapedText}')" class="p-1.5 text-gray-400 dark:text-slate-500 hover:text-gray-600 dark:hover:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-800 rounded-full transition-colors" title="复制">
            <svg class="w-[15px] h-[15px]" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
          </button>
          <button onclick="const i = document.getElementById('queryInput'); i.value = '${escapedText}'; i.focus();" class="p-1.5 text-gray-400 dark:text-slate-500 hover:text-gray-600 dark:hover:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-800 rounded-full transition-colors" title="重新编辑">
            <svg class="w-[15px] h-[15px]" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"></path><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path></svg>
          </button>
        </div>
      </div>
    </div>
  `;
  scrollToBottom();
}

function buildEvidenceCropOverlayMarkup(cropRegion) {
  if (!cropRegion) return '';

  const leftPct = Math.max(0, Math.min(100, Number(cropRegion.left_pct)));
  const topPct = Math.max(0, Math.min(100, Number(cropRegion.top_pct)));
  const widthPct = Math.max(0, Math.min(100 - leftPct, Number(cropRegion.width_pct)));
  const heightPct = Math.max(0, Math.min(100 - topPct, Number(cropRegion.height_pct)));
  if (!Number.isFinite(leftPct) || !Number.isFinite(topPct) || !Number.isFinite(widthPct) || !Number.isFinite(heightPct)) {
    return '';
  }
  if (widthPct <= 0 || heightPct <= 0) return '';

  const alpha = Math.max(0.18, Math.min(0.38, Number(cropRegion.confidence || 0.24)));
  const title = cropRegion.query_text
    ? `高亮命中区域 · ${escapeHtml(cropRegion.query_text)}`
    : '高亮命中区域';

  return `<div class="evidence-crop-overlay" style="left:${leftPct}%;top:${topPct}%;width:${widthPct}%;height:${heightPct}%;--crop-alpha:${alpha.toFixed(2)};" title="${title}"></div>`;
}

function buildMatchedSubQueryMarkup(matchedSubQueries = []) {
  const firstMatchedQuery = (matchedSubQueries || []).find((item) => String(item || '').trim());
  if (!firstMatchedQuery) return '';
  const escapedQuery = escapeHtml(String(firstMatchedQuery).trim());
  return `<p class="evidence-subquery" title="${escapedQuery}">命中：${escapedQuery}</p>`;
}

function _buildEvidenceCards(msgId, evidences, allCandidates) {
  const formalEvidences = Array.isArray(evidences) ? evidences : [];
  const allCandidatePages = Array.isArray(allCandidates) ? allCandidates : [];
  const unused = allCandidatePages.filter(c => !c.is_used);
  if (formalEvidences.length === 0 && unused.length === 0) return '';

  const cards = formalEvidences.map(ev => {
    const imgSrc = ev.image_base64.startsWith('data:') ? ev.image_base64 : `data:image/jpeg;base64,${ev.image_base64}`;
    const cropOverlay = buildEvidenceCropOverlayMarkup(ev.crop_region);
    const matchedSubQuery = buildMatchedSubQueryMarkup(ev.matched_sub_queries);
    const evidenceBadge = ev.evidence_id
      ? `<span class="evidence-chip">${ev.evidence_id}</span>`
      : '';
    const evidenceAttrs = ev.evidence_id
      ? `id="evidence-${msgId}-${ev.evidence_id}" data-evidence-id="${ev.evidence_id}"`
      : '';
    return `
      <div ${evidenceAttrs} class="evidence-card w-full bg-white dark:bg-slate-800 border border-[#e2e8f0] dark:border-slate-700 rounded-2xl overflow-hidden shadow-sm cursor-pointer hover:shadow-md transition-all group" onclick="openImageModal('${imgSrc}')">
        <div class="relative bg-[#f0f4f9] dark:bg-slate-900" style="aspect-ratio: 4/3;">
          <img src="${imgSrc}" class="w-full h-full object-fill group-hover:scale-[1.015] transition-transform duration-300">
          ${cropOverlay}
          <div class="absolute inset-0 bg-black/0 group-hover:bg-black/10 dark:group-hover:bg-white/10 transition-colors flex items-center justify-center">
            <svg class="w-6 h-6 text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow-md" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.35-4.35"></path><line x1="11" y1="8" x2="11" y2="14"></line><line x1="8" y1="11" x2="14" y2="11"></line></svg>
          </div>
          <div class="absolute top-1.5 right-1.5">${evidenceBadge}</div>
        </div>
        <div class="p-2.5">
          <h4 class="text-[13px] font-medium text-[#1f1f1f] dark:text-slate-200 line-clamp-1">${ev.document_name}</h4>
          <p class="text-[11px] text-[#80868b] dark:text-slate-400 mt-0.5">Page ${ev.page_number} · ${ev.score.toFixed(2)}</p>
          ${matchedSubQuery}
        </div>
      </div>`;
  }).join('');

  let disclosureHtml = '';
  if (unused.length > 0) {
    const toggleId = 'all-cands-' + Date.now();
    const unusedCards = unused.map(ev => {
      const imgSrc = ev.image_base64.startsWith('data:') ? ev.image_base64 : `data:image/jpeg;base64,${ev.image_base64}`;
      const unusedReason = String(ev.unused_reason || '').trim();
      const escapedReason = unusedReason ? escapeHtml(unusedReason) : '未采用';
      return `
        <div class="w-full bg-white dark:bg-slate-800 border border-[#e2e8f0] dark:border-slate-700 rounded-2xl overflow-hidden shadow-sm cursor-pointer hover:shadow-md transition-all group opacity-50 hover:opacity-80" onclick="openImageModal('${imgSrc}')">
          <div class="relative bg-[#f0f4f9] dark:bg-slate-900" style="aspect-ratio: 4/3;">
            <img src="${imgSrc}" class="w-full h-full object-fill group-hover:scale-[1.015] transition-transform duration-300">
            <div class="absolute top-1.5 left-1.5">
              <span title="${escapedReason}" class="text-[10px] bg-[#e8eaed] dark:bg-slate-700 text-[#80868b] dark:text-slate-400 px-1.5 py-0.5 rounded-full font-medium">未采用</span>
            </div>
          </div>
          <div class="p-2.5">
            <h4 class="text-[13px] font-medium text-[#1f1f1f] dark:text-slate-200 line-clamp-1">${ev.document_name}</h4>
            <p class="text-[11px] text-[#80868b] dark:text-slate-400 mt-0.5">Page ${ev.page_number} · ${ev.score.toFixed(2)}</p>
            ${unusedReason ? `<p class="text-[10px] text-[#b3261e] dark:text-red-400 mt-0.5">${escapedReason}</p>` : ''}
          </div>
        </div>`;
    }).join('');
    disclosureHtml = `
      <div class="mt-4 w-full">
        <button onclick="(function(btn){var grid=document.getElementById('${toggleId}');var isHidden=grid.style.display==='none';grid.style.display=isHidden?'grid':'none';btn.querySelector('svg').style.transform=isHidden?'rotate(180deg)':''})(this)" class="flex items-center gap-1.5 text-[11px] font-semibold text-[#80868b] dark:text-slate-400 hover:text-[#4285f4] dark:hover:text-blue-400 transition-colors mb-2 pl-1 uppercase tracking-wider">
          <svg xmlns="http://www.w3.org/2000/svg" class="w-3 h-3 transition-transform duration-200" style="transform:none" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
          查看全部候选页 (${unused.length} 未采用)
        </button>
        <div id="${toggleId}" style="display:none" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">${unusedCards}</div>
      </div>`;
  }

  const primarySectionHtml = formalEvidences.length > 0
    ? `
        <p class="text-[11px] font-semibold text-[#80868b] dark:text-slate-400 mb-3 uppercase tracking-wider pl-1">参考源 (Source Evidence)</p>
        <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">${cards}</div>`
    : `
        <p class="text-[11px] font-semibold text-[#80868b] dark:text-slate-400 mb-1 uppercase tracking-wider pl-1">参考源 (Source Evidence)</p>
        <p class="text-[12px] text-[#80868b] dark:text-slate-400 pl-1">当前没有达到采用阈值的正式依据，下面仅保留未采用候选供调试查看。</p>`;

  const disclosureMeta = formalEvidences.length > 0
    ? `默认收起 · 共 ${formalEvidences.length} 条依据${unused.length > 0 ? `，${unused.length} 条未采用候选` : ''}`
    : `默认收起 · 当前无正式依据${unused.length > 0 ? `，${unused.length} 条未采用候选` : ''}`;

  return `
    <div class="mt-4 w-full evidence-disclosure">
      <button id="evidence-toggle-${msgId}" data-count="${formalEvidences.length}" data-collapsed-meta="${escapeHtml(disclosureMeta)}" type="button" onclick="toggleEvidenceSection('${msgId}')" aria-expanded="false" class="evidence-disclosure-toggle">
        <span class="evidence-disclosure-copy">
          <span class="evidence-disclosure-title" data-evidence-label>查看依据</span>
          <span class="evidence-disclosure-meta" data-evidence-meta>${disclosureMeta}</span>
        </span>
        <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4 transition-transform duration-200" style="transform:rotate(0deg)" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
      </button>
      <div id="evidence-body-${msgId}" style="display:none" class="evidence-disclosure-body">
        ${primarySectionHtml}
        ${disclosureHtml}
      </div>
    </div>`;
}

function _insertStreamingShell(msgId, evidences, allCandidates) {
  const meta = arguments[3] || {};
  setMessageMeta(msgId, { evidences, allCandidates, ...meta });
  const existingEl = document.getElementById(msgId);
  if (existingEl) {
    existingEl.className = 'flex gap-4 flex-row group';
    existingEl.innerHTML = buildStreamingShellBody(msgId, evidences, allCandidates, meta);
  } else {
    chatHistory.innerHTML += `
    <div id="${msgId}" class="flex gap-4 flex-row group">
      ${buildStreamingShellBody(msgId, evidences, allCandidates, meta)}
    </div>`;
  }
  assistantMessageCache.set(msgId, '');
  scrollToBottom();
}

function addAssistantMessage(markdownText, evidences = []) {
  welcomeScreen.classList.add("hidden");
  chatHistory.classList.remove("hidden");

  const msgId = 'msg-' + Date.now();
  setMessageMeta(msgId, { evidences });
  const evHTML = _buildEvidenceCards(msgId, evidences, []);

  const htmlContent = renderMarkdownWithCitations(msgId, markdownText);
  assistantMessageCache.set(msgId, markdownText);
  
  chatHistory.innerHTML += `
    <div class="flex gap-4 flex-row group">
      <div class="w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1">
        <svg class="w-6 h-6 outline-none" fill="#4285f4" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2.628c-.896 5.867-5.505 10.476-11.372 11.372 5.867.896 10.476 5.505 11.372 11.372.896-5.867 5.505-10.476 11.372-11.372-5.867-.896-10.476-5.505-11.372-11.372z"/>
        </svg>
      </div>
      <div class="flex flex-col gap-1 items-start w-full">
        <div class="text-[16px] leading-[1.8] text-[#1f1f1f] dark:text-slate-200 w-full markdown-body">
          ${htmlContent}
        </div>
        ${evHTML}
        <!-- Action Bar: Visible only on hover -->
        <div id="stream-actions-${msgId}" class="flex gap-2 ml-2 opacity-0 group-hover:opacity-100 transition-opacity items-center mt-1">
        </div>
      </div>
    </div>
  `;
  renderAssistantActions(msgId, { showCopy: true });
  scrollToBottom();
}

function showLoading() {
  const id = 'load-' + Date.now();
  chatHistory.innerHTML += `
    <div id="${id}" class="flex gap-4">
      <div class="w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1">
        <svg class="w-6 h-6 outline-none" fill="url(#sparkle-loading-grad2)" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="sparkle-loading-grad2" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#4285f4" />
              <stop offset="50%" stop-color="#9b72cb" />
              <stop offset="100%" stop-color="#d96570" />
            </linearGradient>
          </defs>
          <path d="M12 2.628c-.896 5.867-5.505 10.476-11.372 11.372 5.867.896 10.476 5.505 11.372 11.372.896-5.867 5.505-10.476 11.372-11.372-5.867-.896-10.476-5.505-11.372-11.372z"/>
        </svg>
      </div>
      <div class="px-5 flex items-center justify-center gap-1.5 h-[32px] mt-1 relative">
        <div class="w-2 h-2 bg-[#4285f4] opacity-90 rounded-full animate-bounce" style="animation-delay: -0.3s"></div>
        <div class="w-2 h-2 bg-[#4285f4] opacity-90 rounded-full animate-bounce" style="animation-delay: -0.15s"></div>
        <div class="w-2 h-2 bg-[#4285f4] opacity-90 rounded-full animate-bounce"></div>
        <span class="ml-3 text-[14px] font-medium text-[#444746] dark:text-slate-300">思考中...</span>
      </div>
    </div>
  `;
  scrollToBottom();
  return id;
}

function removeLoading(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function updateInputState(count) {
  const txt = document.getElementById('uploadBtnText');
  const input = document.getElementById('queryInput');
  if (!txt || !input) return;
  if (count === 0) {
    txt.classList.remove('w-0', 'opacity-0', 'ml-0');
    txt.classList.add('w-[60px]', 'opacity-100', 'ml-1.5');
    input.placeholder = "💡 请先点击左侧上传 PDF、图片或文本文档...";
  } else {
    txt.classList.remove('w-[60px]', 'opacity-100', 'ml-1.5');
    txt.classList.add('w-0', 'opacity-0', 'ml-0');
    input.placeholder = "检索已上传的文档...";
  }
}

async function initApp() {
  try {
    const res = await fetch('/api/rag/files');
    const data = await res.json();
    uploadedDocs = data.files || [];
  } catch (err) {
    console.error(err);
  }
  updateInputState(uploadedDocs.length);
  renderScopeBar();
}

initApp();
