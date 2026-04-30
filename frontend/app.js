// Core State
let uploadedDocs = [];
let isUploading = false;
let isChatting = false;
let selectedDocIds = null; // null = 全部文档；array of IDs = 当前选中的范围
let conversationHistory = []; // Multi-turn memory

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
      addAssistantMessage(`✅ **${data.body.document_name}** 已成功上传并建立索引。现在，您可以就此文档向我提问了！`);
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
async function handleChat(fromHistory = false) {
  const text = queryInput.value.trim();
  if (!text || isChatting) return;
  
  if (!fromHistory) {
    history.pushState({view: 'chat'}, '', '#chat');
  }

  welcomeScreen.classList.add("hidden");
  chatHistory.classList.remove("hidden");
  inputContainer.classList.remove("-translate-y-[25vh]", "md:-translate-y-[30vh]");

  addUserMessage(text);
  queryInput.value = "";
  
  if (uploadedDocs.length === 0) {
    addAssistantMessage("🤖 提示：请先点击下方上传一份文档～");
    return;
  }
  
  isChatting = true;
  const loadId = showLoading();

  try {
    const res = await fetch('/api/rag/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: text,
        top_k: 5,
        chat_history: conversationHistory,
        ...(selectedDocIds ? { document_ids: selectedDocIds } : {})
      })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: res.statusText }));
      removeLoading(loadId);
      addAssistantMessage(`❌ Error: ${errData.detail}`);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let answerText = '';
    let msgId = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // retain incomplete last line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();

        if (raw === '[DONE]') {
          if (msgId) {
            // Remove typing cursor
            const cursor = document.getElementById('cursor-' + msgId);
            if (cursor) cursor.remove();
            // Add copy button
            const actionsEl = document.getElementById('stream-actions-' + msgId);
            if (actionsEl) {
              const escaped = answerText.replace(/"/g, '&quot;').replace(/'/g, '&apos;').replace(/\n/g, '\\n');
              actionsEl.innerHTML = `<button onclick="navigator.clipboard.writeText('${escaped}'); const i=this.querySelector('svg'); const old=i.innerHTML; i.innerHTML='<polyline points=\\'20 6 9 17 4 12\\'></polyline>'; setTimeout(()=>i.innerHTML=old,1500);" class="p-1.5 text-gray-400 dark:text-slate-500 hover:text-gray-600 dark:hover:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-800 rounded-full transition-colors flex items-center gap-1.5 text-xs font-medium" title="复制结果"><svg class="w-[15px] h-[15px]" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg><span>复制</span></button>`;
              actionsEl.classList.remove('opacity-0');
              actionsEl.classList.add('opacity-0'); // stays hidden until hover via group
            }
            // Save multi-turn history (cap at 20 messages = 10 turns)
            conversationHistory.push({ role: 'user', content: text });
            conversationHistory.push({ role: 'assistant', content: answerText });
            if (conversationHistory.length > 20) {
              conversationHistory = conversationHistory.slice(conversationHistory.length - 20);
            }
          }
          break;
        }

        let event;
        try { event = JSON.parse(raw); } catch { continue; }

        if (event.type === 'error') {
          removeLoading(loadId);
          addAssistantMessage(event.data);
          return;
        }

        if (event.type === 'evidences') {
          removeLoading(loadId);
          msgId = 'msg-' + Date.now();
          _insertStreamingShell(msgId, event.data.evidences, event.data.all_candidates);
        }

        if (event.type === 'token' && msgId) {
          answerText += event.data;
          const answerEl = document.getElementById('stream-answer-' + msgId);
          if (answerEl) {
            answerEl.innerHTML = marked.parse(answerText) +
              `<span id="cursor-${msgId}" class="inline-block w-[2px] h-[1.1em] bg-current align-middle ml-0.5 animate-pulse opacity-70"></span>`;
          }
          scrollToBottom();
        }
      }
    }
  } catch (err) {
    removeLoading(loadId);
    addAssistantMessage(`❌ 网络错误，请重试。`);
  } finally {
    isChatting = false;
  }
}

if (sendBtn) {
  sendBtn.addEventListener('click', handleChat);
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
  messagesContainer.scrollTo({ top: messagesContainer.scrollHeight, behavior: 'smooth' });
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

function _buildEvidenceCards(evidences, allCandidates) {
  if (!evidences || evidences.length === 0) return '';
  const cards = evidences.map(ev => {
    const imgSrc = ev.image_base64.startsWith('data:') ? ev.image_base64 : `data:image/jpeg;base64,${ev.image_base64}`;
    return `
      <div class="w-full bg-white dark:bg-slate-800 border border-[#e2e8f0] dark:border-slate-700 rounded-2xl overflow-hidden shadow-sm cursor-pointer hover:shadow-md transition-all group" onclick="openImageModal('${imgSrc}')">
        <div class="relative bg-[#f0f4f9] dark:bg-slate-900" style="aspect-ratio: 4/3;">
          <img src="${imgSrc}" class="w-full h-full object-fill group-hover:scale-[1.015] transition-transform duration-300">
          <div class="absolute inset-0 bg-black/0 group-hover:bg-black/10 dark:group-hover:bg-white/10 transition-colors flex items-center justify-center">
            <svg class="w-6 h-6 text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow-md" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.35-4.35"></path><line x1="11" y1="8" x2="11" y2="14"></line><line x1="8" y1="11" x2="14" y2="11"></line></svg>
          </div>
        </div>
        <div class="p-2.5">
          <h4 class="text-[13px] font-medium text-[#1f1f1f] dark:text-slate-200 line-clamp-1">${ev.document_name}</h4>
          <p class="text-[11px] text-[#80868b] dark:text-slate-400 mt-0.5">Page ${ev.page_number} · ${ev.score.toFixed(2)}</p>
        </div>
      </div>`;
  }).join('');

  // 未被采用的候选页（得分不足 min_score 或排名靠后）
  const unused = (allCandidates || []).filter(c => !c.is_used);
  let disclosureHtml = '';
  if (unused.length > 0) {
    const toggleId = 'all-cands-' + Date.now();
    const unusedCards = unused.map(ev => {
      const imgSrc = ev.image_base64.startsWith('data:') ? ev.image_base64 : `data:image/jpeg;base64,${ev.image_base64}`;
      return `
        <div class="w-full bg-white dark:bg-slate-800 border border-[#e2e8f0] dark:border-slate-700 rounded-2xl overflow-hidden shadow-sm cursor-pointer hover:shadow-md transition-all group opacity-50 hover:opacity-80" onclick="openImageModal('${imgSrc}')">
          <div class="relative bg-[#f0f4f9] dark:bg-slate-900" style="aspect-ratio: 4/3;">
            <img src="${imgSrc}" class="w-full h-full object-fill group-hover:scale-[1.015] transition-transform duration-300">
            <div class="absolute top-1.5 left-1.5">
              <span class="text-[10px] bg-[#e8eaed] dark:bg-slate-700 text-[#80868b] dark:text-slate-400 px-1.5 py-0.5 rounded-full font-medium">未采用</span>
            </div>
          </div>
          <div class="p-2.5">
            <h4 class="text-[13px] font-medium text-[#1f1f1f] dark:text-slate-200 line-clamp-1">${ev.document_name}</h4>
            <p class="text-[11px] text-[#80868b] dark:text-slate-400 mt-0.5">Page ${ev.page_number} · ${ev.score.toFixed(2)}</p>
          </div>
        </div>`;
    }).join('');
    disclosureHtml = `
      <div class="mt-2 w-full">
        <button onclick="(function(btn){var grid=document.getElementById('${toggleId}');var isHidden=grid.style.display==='none';grid.style.display=isHidden?'grid':'none';btn.querySelector('svg').style.transform=isHidden?'rotate(180deg)':''})(this)" class="flex items-center gap-1.5 text-[11px] font-semibold text-[#80868b] dark:text-slate-400 hover:text-[#4285f4] dark:hover:text-blue-400 transition-colors mb-2 pl-1 uppercase tracking-wider">
          <svg xmlns="http://www.w3.org/2000/svg" class="w-3 h-3 transition-transform duration-200" style="transform:none" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
          查看全部候选页 (${unused.length} 未采用)
        </button>
        <div id="${toggleId}" style="display:none" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">${unusedCards}</div>
      </div>`;
  }

  return `
    <div class="mt-3 w-full">
      <p class="text-[11px] font-semibold text-[#80868b] dark:text-slate-400 mb-2 uppercase tracking-wider pl-1">参考源 (Source Evidence)</p>
      <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">${cards}</div>
      ${disclosureHtml}
    </div>`;
}

function _insertStreamingShell(msgId, evidences, allCandidates) {
  const evHTML = _buildEvidenceCards(evidences, allCandidates);
  chatHistory.innerHTML += `
    <div id="${msgId}" class="flex gap-4 flex-row group">
      <div class="w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1">
        <svg class="w-6 h-6 outline-none" fill="#4285f4" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2.628c-.896 5.867-5.505 10.476-11.372 11.372 5.867.896 10.476 5.505 11.372 11.372.896-5.867 5.505-10.476 11.372-11.372-5.867-.896-10.476-5.505-11.372-11.372z"/>
        </svg>
      </div>
      <div class="flex flex-col gap-1 items-start w-full">
        <div id="stream-answer-${msgId}" class="text-[16px] leading-[1.8] text-[#1f1f1f] dark:text-slate-200 w-full markdown-body">
          <div id="stream-thinking-${msgId}" class="flex items-center gap-1.5 py-1">
            <div class="w-1.5 h-1.5 bg-[#4285f4] opacity-70 rounded-full animate-bounce" style="animation-delay:-0.3s"></div>
            <div class="w-1.5 h-1.5 bg-[#4285f4] opacity-70 rounded-full animate-bounce" style="animation-delay:-0.15s"></div>
            <div class="w-1.5 h-1.5 bg-[#4285f4] opacity-70 rounded-full animate-bounce"></div>
            <span class="ml-1 text-[13px] text-[#80868b] dark:text-slate-400">正在生成回答...</span>
          </div>
        </div>
        ${evHTML}
        <div id="stream-actions-${msgId}" class="flex gap-2 ml-2 opacity-0 group-hover:opacity-100 transition-opacity items-center mt-1"></div>
      </div>
    </div>`;
  scrollToBottom();
}

function addAssistantMessage(markdownText, evidences = []) {
  welcomeScreen.classList.add("hidden");
  chatHistory.classList.remove("hidden");

  const evHTML = _buildEvidenceCards(evidences);

  const htmlContent = marked.parse(markdownText);
  const escapedMd = markdownText.replace(/"/g, '&quot;').replace(/'/g, '&apos;').replace(/\n/g, '\\n');
  
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
        <div class="flex gap-2 ml-2 opacity-0 group-hover:opacity-100 transition-opacity items-center mt-1">
          <button onclick="navigator.clipboard.writeText('${escapedMd}'); const i=this.querySelector('svg'); const old=i.innerHTML; i.innerHTML='<polyline points=\\'20 6 9 17 4 12\\'></polyline>'; setTimeout(()=>i.innerHTML=old, 1500);" class="p-1.5 text-gray-400 dark:text-slate-500 hover:text-gray-600 dark:hover:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-800 rounded-full transition-colors flex items-center gap-1.5 text-xs text-gray-500 dark:text-slate-400 font-medium" title="复制结果">
            <svg class="w-[15px] h-[15px]" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            <span>复制</span>
          </button>
        </div>
      </div>
    </div>
  `;
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
