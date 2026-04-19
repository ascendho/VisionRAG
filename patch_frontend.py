import re

with open('frontend/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove aside
content = re.sub(
    r'(?s)<!-- Sidebar -->.*?</aside>', 
    '', 
    content
)

# 2. Remove avatar
content = re.sub(
    r'<div class="w-8 h-8 rounded-full bg-emerald-700 font-bold text-white flex items-center justify-center text-sm">\s*M\s*</div>',
    '',
    content
)

# 3. Replace Welcome Screen
welcome_old = r'(?s)<div id="welcomeScreen".*?</div>\s*</div>\s*</div>'
welcome_new = '''<div id="welcomeScreen" class="h-full flex flex-col justify-center max-w-3xl mx-auto w-full mt-[-8vh]">
        <div class="mb-12">
          <h1 class="text-[40px] md:text-[56px] font-semibold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-500 via-purple-500 to-rose-400 mb-2 leading-tight">
            你好
          </h1>
          <p class="text-3xl md:text-4xl text-[#c4c7c5] font-medium tracking-tight">
            上传 PDF 开启多模态视觉检索
          </p>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mt-8 max-w-3xl">
          <div class="p-4 rounded-xl bg-[#f0f4f9] text-[#444746] h-[100px] flex flex-col justify-between hover:bg-[#e8eef6] transition-colors cursor-default">
            <p class="text-sm">在图表和复杂图片中精准搜索信息</p>
            <i data-lucide="scan-eye" class="w-5 h-5 self-end text-zinc-400"></i>
          </div>
          <div class="p-4 rounded-xl bg-[#f0f4f9] text-[#444746] h-[100px] flex flex-col justify-between hover:bg-[#e8eef6] transition-colors cursor-default">
            <p class="text-sm">支持多文件全局智能检索，无缝对话</p>
            <i data-lucide="sparkles" class="w-5 h-5 self-end text-zinc-400"></i>
          </div>
        </div>
      </div>'''
content = re.sub(welcome_old, welcome_new, content)

# Add hide-scrollbar utility class
css_old = r'/\* Hide scrollbar for clean look \*/\s*::-webkit-scrollbar \{'
css_new = r'''.hide-scrollbar::-webkit-scrollbar { display: none; }
    .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
    /* Hide scrollbar for clean look */
    ::-webkit-scrollbar {'''
content = re.sub(css_old, css_new, content)

# Add Modal
modal_html = '''  <!-- Lightbox Modal -->
  <div id="imageModal" class="fixed inset-0 z-50 bg-black/80 hidden items-center justify-center p-4 md:p-8 backdrop-blur-sm transition-opacity opacity-0 cursor-pointer" onclick="closeImageModal()">
    <div class="relative max-w-full max-h-full">
      <img id="modalImage" src="" class="max-w-full max-h-[90vh] object-contain rounded-lg shadow-2xl" onclick="event.stopPropagation()">
      <button class="absolute -top-4 -right-4 p-2 bg-white/10 hover:bg-white/20 rounded-full text-white backdrop-blur-md transition-colors" onclick="closeImageModal()">
        <i data-lucide="x" class="w-5 h-5"></i>
      </button>
    </div>
  </div>
  <script>'''
content = content.replace('  <script>', modal_html)

# Rewrite JS variables
js_vars_old = r'(?s)// Elements.*?const messagesContainer = document.getElementById\(\'messagesContainer\'\);'
js_vars_new = '''// Elements
    const bottomFileInput = document.getElementById('bottomFileInput');
    const queryInput = document.getElementById('queryInput');
    const sendBtn = document.getElementById('sendBtn');
    
    const welcomeScreen = document.getElementById('welcomeScreen');
    const chatHistory = document.getElementById('chatHistory');
    const messagesContainer = document.getElementById('messagesContainer');
    
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
    
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeImageModal();
    });'''
content = re.sub(js_vars_old, js_vars_new, content)

# Rewrite Upload
upload_old = r'(?s)// Handle Uploads.*?function renderDocs[^\}]+\}'
upload_new = '''// Handle Uploads
    function showUploadingAnimation(text) {
      welcomeScreen.classList.add("hidden");
      chatHistory.classList.remove("hidden");
      
      const id = 'load-' + Date.now();
      chatHistory.innerHTML += `
        <div id="${id}" class="flex gap-4 flex-row">
          <div class="w-8 h-8 rounded-full bg-[#1a73e8]/10 flex items-center justify-center shrink-0 mt-1 relative overflow-hidden">
            <div class="absolute inset-0 bg-gradient-to-tr from-blue-600 to-purple-500 animate-[spin_2s_linear_infinite]"></div>
            <div class="absolute inset-[2px] bg-white rounded-full flex items-center justify-center">
               <i data-lucide="sparkles" class="w-3.5 h-3.5 text-[#1a73e8]"></i>
            </div>
          </div>
          <div class="flex flex-col gap-3 items-start">
            <div class="px-5 py-3.5 text-[15px] leading-relaxed text-[#444746] rounded-3xl bg-[#f0f4f9]/50 animate-pulse">
              ${text}
            </div>
          </div>
        </div>
      `;
      lucide.createIcons();
      scrollToBottom();
      return id;
    }

    async function handleUpload(file) {
      if (!file || isUploading) return;
      isUploading = true;
      
      const loadId = showUploadingAnimation("正在解析并建立特征索引...");
      
      const formData = new FormData();
      formData.append('file', file);
      
      try {
        const res = await fetch('/api/rag/upload', { method: 'POST', body: formData });
        const data = await res.json();
        
        removeLoading(loadId);
        if (res.ok) {
          uploadedDocs.push(data);
          addAssistantMessage(`✅ **${data.document_name}** 上传并索引成功。您可以直接提问了。`);
        } else {
          addAssistantMessage(`❌ 上传失败: ${data.detail}`);
        }
      } catch (err) {
        removeLoading(loadId);
        addAssistantMessage(`❌ 网络错误，上传失败。`);
      } finally {
        isUploading = false;
        bottomFileInput.value = "";
      }
    }

    bottomFileInput.addEventListener('change', (e) => handleUpload(e.target.files[0]));'''
content = re.sub(upload_old, upload_new, content)

# Rewrite addAssistantMessage image cards
ev_old = r'(?s)let cards = evidences\.map\(ev => `.*?`\)\.join\(\'\'\);'
ev_new = '''let cards = evidences.map(ev => {
          // Backend already attaches data URI prefix if configured right, but ensure we don't duplicate
          const imgSrc = ev.image_base64.startsWith('data:') ? ev.image_base64 : `data:image/jpeg;base64,${ev.image_base64}`;
          return `
          <div class="snap-start shrink-0 w-[140px] md:w-[160px] bg-white border border-[#e2e8f0] rounded-2xl overflow-hidden shadow-sm cursor-pointer hover:shadow-md transition-all group" onclick="openImageModal('${imgSrc}')">
            <div class="relative">
              <img src="${imgSrc}" class="w-full h-[100px] object-cover bg-[#f0f4f9] group-hover:scale-105 transition-transform duration-300">
              <div class="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-center justify-center">
                <i data-lucide="zoom-in" class="w-6 h-6 text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow-md"></i>
              </div>
            </div>
            <div class="p-2.5">
              <h4 class="text-[13px] font-medium text-[#1f1f1f] line-clamp-1">${ev.document_name}</h4>
              <p class="text-[11px] text-[#80868b] mt-0.5">Page ${ev.page_number} • Score: ${ev.score.toFixed(2)}</p>
            </div>
          </div>
        `}).join('');'''
content = re.sub(ev_old, ev_new, content)

# Change evHTML wrapper styles
evhtml_old = r'(?s)evHTML = `\s*<div class="mt-2 w-full">\s*<p class="text-xs font-semibold text-\[\#80868b\] mb-3 uppercase tracking-wider pl-1">Source Evidence</p>\s*<div class="flex gap-4 overflow-x-auto pb-4 snap-x pr-4">\s*\$\{cards\}\s*</div>\s*</div>\s*`;'
evhtml_new = '''evHTML = `
          <div class="mt-3 w-full">
            <p class="text-[11px] font-semibold text-[#80868b] mb-2 uppercase tracking-wider pl-1">参考源 (Source Evidence)</p>
            <div class="flex gap-3 overflow-x-auto pb-4 snap-x pr-4 hide-scrollbar">
              ${cards}
            </div>
          </div>
        `;'''
content = re.sub(evhtml_old, evhtml_new, content)

# Remove old event listener for fileInput
content = re.sub(r"fileInput\.addEventListener.*?\n", "", content)

# Clean up docList and let uploadedDocs stay since it tracks uploaded stuff but docList is gone
content = re.sub(r'const docList = document\.getElementById\(\'docList\'\);\s*', '', content)

# Fix prompt placeholder
content = content.replace('Ask about your documents...', '检索已上传的文档...')

with open('frontend/index.html', 'w', encoding='utf-8') as f:
    f.write(content)
