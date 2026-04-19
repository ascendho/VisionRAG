import re
import os

html_path = 'frontend/index.html'

with open(html_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove lucide script tag
content = re.sub(r'<script\s+src="https://unpkg\.com/lucide@latest"></script>\n?', '', content)

# 2. Remove all lucide.createIcons() calls
content = re.sub(r'\s*lucide\.createIcons\(\);?\s*', '\n', content)

# 3. Fix syntax error
error_block = r"""    bottomFileInput\.addEventListener\('change', \(e\) => handleUpload\(e\.target\.files\[0\]\)\);</span>\s*</div>\s*`\)\.join\(''\);\s*lucide\.createIcons\(\);\s*\}"""
content = re.sub(error_block, r"    bottomFileInput.addEventListener('change', (e) => handleUpload(e.target.files[0]));", content, flags=re.DOTALL)
# Also clean up any trailing stray createIcons/join calls if regex missed it for some reason by just hard replacing the main one.
content = content.replace("bottomFileInput.addEventListener('change', (e) => handleUpload(e.target.files[0]));</span>\n        </div>\n      `).join('');\n      lucide.createIcons();\n    }", "bottomFileInput.addEventListener('change', (e) => handleUpload(e.target.files[0]));")

# 4. Add welcome card CSS
css_to_add = """
    .welcome-card {
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      border-radius: 1rem;
      background: linear-gradient(145deg, #f8fafc 0%, #f0f4f9 100%);
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      min-height: 110px;
      border: 1px solid rgba(26, 115, 232, 0.08);
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    .welcome-card:hover {
      background: linear-gradient(145deg, #ffffff 0%, #f5f9ff 100%);
      border-color: rgba(26, 115, 232, 0.2);
      box-shadow: 0 10px 20px -3px rgba(26, 115, 232, 0.15), 0 4px 6px -2px rgba(26, 115, 232, 0.08);
      transform: translateY(-2px);
    }
"""
content = content.replace('    .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }', '    .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }\n' + css_to_add)

# 5. Apply welcome-card class
content = content.replace('class="p-4 rounded-xl bg-[#f0f4f9] text-[#444746] h-[100px] flex flex-col justify-between hover:bg-[#e8eef6] transition-colors cursor-default"', 'class="welcome-card text-[#444746] cursor-pointer"')

# 6. Replace Icons with inline SVGs
svgs = {
    "sparkles": '<svg class="{0}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l2.5 7.5H22l-6 4.5 2.5 7.5L12 17l-6 4.5 2.5-7.5L2 9.5h7.5L12 2z"/></svg>',
    "plus": '<svg class="{0}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>',
    "send-horizontal": '<svg class="{0}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"></polyline><path d="M4 12h12"></path></svg>',
    "scan-eye": '<svg class="{0}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7v6h6"></path><path d="M21 17v-6h-6"></path><circle cx="12" cy="12" r="1"></circle><path d="M9 12a3 3 0 1 0 6 0 3 3 0 0 0-6 0"></path></svg>',
    "zoom-in": '<svg class="{0}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.35-4.35"></path><line x1="11" y1="8" x2="11" y2="14"></line><line x1="8" y1="11" x2="14" y2="11"></line></svg>',
    "x": '<svg class="{0}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>',
    "file-text": '<svg class="{0}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>'
}

def replace_icon(m):
    icon_name = m.group(1)
    classes = m.group(2) if m.group(2) else ""
    if icon_name in svgs:
        return svgs[icon_name].format(classes)
    return m.group(0)

content = re.sub(r'<i\s+data-lucide="([^"]+)"(?:\s+class="([^"]+)")?[^>]*></i>', replace_icon, content)

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Done fixing frontend/index.html")
