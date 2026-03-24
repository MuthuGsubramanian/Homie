"""Chat page generator for Homie desktop companion.

Produces a self-contained HTML chat interface. No external dependencies.
"""
from __future__ import annotations


def render_chat_page(session_token: str, api_port: int) -> str:
    """Render a self-contained chat HTML page."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Homie — Chat</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; display: flex; flex-direction: column; height: 100vh; }}

  .nav {{ display: flex; gap: 8px; padding: 12px 24px; background: #161b22; border-bottom: 1px solid #30363d; }}
  .nav a {{ color: #8b949e; text-decoration: none; font-size: 13px; padding: 6px 12px; border-radius: 6px; }}
  .nav a:hover {{ color: #c9d1d9; background: #21262d; }}
  .nav a.active {{ color: #f0f6fc; background: #21262d; }}

  .chat-container {{ flex: 1; overflow-y: auto; padding: 24px; max-width: 800px; margin: 0 auto; width: 100%; }}

  .message {{ margin-bottom: 16px; display: flex; gap: 12px; }}
  .message.user {{ justify-content: flex-end; }}
  .message .bubble {{ max-width: 75%; padding: 12px 16px; border-radius: 12px; font-size: 14px; line-height: 1.5; white-space: pre-wrap; }}
  .message.user .bubble {{ background: #1f6feb; color: #fff; border-bottom-right-radius: 4px; }}
  .message.assistant .bubble {{ background: #161b22; border: 1px solid #30363d; color: #c9d1d9; border-bottom-left-radius: 4px; }}
  .message .source {{ font-size: 11px; color: #484f58; margin-top: 4px; }}

  .welcome {{ text-align: center; padding: 60px 20px; color: #8b949e; }}
  .welcome h2 {{ color: #f0f6fc; font-size: 20px; margin-bottom: 8px; }}
  .welcome p {{ font-size: 14px; }}

  .input-area {{ padding: 16px 24px; background: #161b22; border-top: 1px solid #30363d; }}
  .input-row {{ display: flex; gap: 8px; max-width: 800px; margin: 0 auto; }}
  .input-row textarea {{ flex: 1; background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 10px 14px; color: #c9d1d9; font-size: 14px; font-family: inherit; resize: none; min-height: 44px; max-height: 120px; }}
  .input-row textarea:focus {{ outline: none; border-color: #1f6feb; }}
  .input-row button {{ background: #1f6feb; color: #fff; border: none; border-radius: 8px; padding: 10px 20px; font-size: 14px; cursor: pointer; font-weight: 500; }}
  .input-row button:hover {{ background: #388bfd; }}
  .input-row button:disabled {{ background: #21262d; color: #484f58; cursor: not-allowed; }}

  .typing {{ display: none; margin-bottom: 16px; }}
  .typing.visible {{ display: flex; }}
  .typing .bubble {{ background: #161b22; border: 1px solid #30363d; padding: 12px 16px; border-radius: 12px; color: #8b949e; font-size: 14px; }}

  .footer {{ text-align: center; padding: 8px; font-size: 11px; color: #484f58; }}
</style>
</head>
<body>

<div class="nav">
  <a href="/briefing">Briefing</a>
  <a href="/chat" class="active">Chat</a>
  <a href="/settings">Settings</a>
</div>

<div class="chat-container" id="chatContainer">
  <div class="welcome" id="welcome">
    <h2>Chat with Homie</h2>
    <p>Ask about your emails, draft replies, or get help with anything.<br>All processing happens locally on your machine.</p>
  </div>
</div>

<div class="input-area">
  <div class="input-row">
    <textarea id="msgInput" placeholder="Ask Homie anything..." rows="1" onkeydown="if(event.key==='Enter'&&!event.shiftKey){{event.preventDefault();sendMessage()}}"></textarea>
    <button id="sendBtn" onclick="sendMessage()">Send</button>
  </div>
</div>

<div class="footer">Homie AI &mdash; all data stays local</div>

<script>
const API = "http://127.0.0.1:{api_port}";
document.cookie = "homie_session={session_token}; path=/; SameSite=Strict";

const container = document.getElementById("chatContainer");
const input = document.getElementById("msgInput");
const sendBtn = document.getElementById("sendBtn");
const welcome = document.getElementById("welcome");

// Auto-resize textarea
input.addEventListener("input", function() {{
  this.style.height = "auto";
  this.style.height = Math.min(this.scrollHeight, 120) + "px";
}});

// Load history on page load
async function loadHistory() {{
  try {{
    const resp = await fetch(API + "/api/chat/history", {{credentials: "include"}});
    const data = await resp.json();
    if (data.messages && data.messages.length > 0) {{
      welcome.style.display = "none";
      data.messages.forEach(m => addMessage(m.role, m.content));
      scrollToBottom();
    }}
  }} catch(e) {{}}
}}

function addMessage(role, content, source) {{
  const div = document.createElement("div");
  div.className = "message " + role;
  let html = '<div class="bubble">' + escapeHtml(content) + '</div>';
  if (source && role === "assistant") {{
    html += '<div class="source">via ' + escapeHtml(source) + '</div>';
  }}
  div.innerHTML = html;
  container.appendChild(div);
}}

function escapeHtml(text) {{
  const d = document.createElement("div");
  d.textContent = text;
  return d.innerHTML;
}}

function scrollToBottom() {{
  container.scrollTop = container.scrollHeight;
}}

async function sendMessage() {{
  const text = input.value.trim();
  if (!text) return;

  welcome.style.display = "none";
  addMessage("user", text);
  input.value = "";
  input.style.height = "auto";
  sendBtn.disabled = true;

  // Show typing indicator
  const typing = document.createElement("div");
  typing.className = "message assistant";
  typing.innerHTML = '<div class="bubble" style="color:#8b949e">Thinking...</div>';
  container.appendChild(typing);
  scrollToBottom();

  try {{
    const resp = await fetch(API + "/api/chat", {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      credentials: "include",
      body: JSON.stringify({{message: text}}),
    }});
    const data = await resp.json();
    container.removeChild(typing);
    addMessage("assistant", data.response || data.error, data.source);
  }} catch(e) {{
    container.removeChild(typing);
    addMessage("assistant", "Connection error: " + e.message);
  }}

  sendBtn.disabled = false;
  input.focus();
  scrollToBottom();
}}

loadHistory();
input.focus();
</script>

</body>
</html>"""
