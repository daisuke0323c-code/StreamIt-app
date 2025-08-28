import os
import time
import json
import requests
import streamlit as st
from typing import Optional, List, Dict

# --- ヘルパ: api.key を同ディレクトリから読む ---
def _load_api_key_from_file() -> Optional[str]:
    base = os.path.dirname(__file__)
    key_path = os.path.join(base, "api.key")
    if os.path.exists(key_path):
        try:
            with open(key_path, "r", encoding="utf-8") as f:
                k = f.read().strip()
                return k or None
        except Exception:
            return None
    return None

# --- OpenAI Chat 呼び出し (HTTP) ---
def call_openai_chat(messages: List[Dict[str, str]], api_key: str, base_url: str, model: str, temperature: Optional[float], max_tokens: Optional[int]) -> Dict:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if max_tokens:
        payload["max_completion_tokens"] = int(max_tokens)
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
    except Exception as e:
        return {"error": f"request error: {e}"}
    try:
        return {"status_code": resp.status_code, "json": resp.json()}
    except Exception:
        return {"status_code": resp.status_code, "json": {"raw_text": resp.text}}

# --- セッション初期化 ---
st.set_page_config(page_title="Simple Chat (OpenAI)", layout="wide")
st.title("Simple Chat (OpenAI)")

st.session_state.setdefault("chat_history", [])  # list of {"role":..., "content":..., "ts":...}
st.session_state.setdefault("last_response", None)

# --- サイドバー: 設定 ---
with st.sidebar:
    st.header("設定")
    api_key_input = st.text_input("API Key（空欄で環境 / api.key）", type="password")
    base_url = st.text_input("Base URL", value=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))
    model = st.text_input("Model", value=os.environ.get("LLM_MODEL", "gpt-4"))
    use_temp = st.checkbox("温度を付与する", value=False)
    temperature = None
    if use_temp:
        temperature = st.slider("temperature", 0.0, 1.0, 0.3, 0.05)
    max_tokens = st.number_input("max_completion_tokens (0=無効)", min_value=0, value=0)
    include_history = st.checkbox("履歴を含める（送信に会話履歴を含める）", value=True)
    save_history = st.checkbox("履歴を保存する（セッション）", value=True)
    st.markdown("---")
    if st.button("履歴をクリア"):
        st.session_state["chat_history"] = []
        st.session_state["last_response"] = None
        st.success("履歴をクリアしました。")

# --- APIキー決定 ---
api_key = api_key_input or os.environ.get("OPENAI_API_KEY") or _load_api_key_from_file()

# --- メイン: チャット表示 ---
col_left, col_right = st.columns([3,1])

with col_left:
    st.subheader("会話")
    # 表示用の会話履歴（最新順でなく時系列）
    if st.session_state["chat_history"]:
        for m in st.session_state["chat_history"]:
            role = m.get("role", "user")
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m.get("ts", 0)))
            if role == "user":
                st.markdown(f"**User** ({ts}):")
                st.write(m.get("content", ""))
            else:
                st.markdown(f"**Assistant** ({ts}) :")
                st.code(m.get("content", ""), language="text")
    else:
        st.info("会話履歴は空です。以下にプロンプトを入力して送信してください。")

    # 入力欄
    user_input = st.text_area("メッセージを入力", height=120, key="chat_input")
    send = st.button("送信")

    if send:
        if not api_key:
            st.error("API Key が見つかりません。サイドバーに入力するか環境変数/ api.key を用意してください。")
        elif not user_input.strip():
            st.warning("メッセージを入力してください。")
        else:
            # 組み立て: 履歴を含める場合は session の履歴を messages に変換
            messages = []
            if include_history:
                for h in st.session_state["chat_history"]:
                    messages.append({"role": h["role"], "content": h["content"]})
            # 現行ユーザメッセージを追加
            messages.append({"role": "user", "content": user_input})

            # API 呼び出し
            with st.spinner("OpenAI に送信中..."):
                res = call_openai_chat(messages, api_key, base_url, model, temperature if use_temp else None, max_tokens if max_tokens>0 else None)
            st.session_state["last_response"] = res

            # 抽出: assistant のテキストがあれば取り出す
            assistant_text = None
            if isinstance(res, dict) and res.get("status_code") == 200:
                try:
                    assistant_text = res["json"]["choices"][0]["message"]["content"]
                except Exception:
                    try:
                        assistant_text = res["json"]["choices"][0].get("text")
                    except Exception:
                        assistant_text = None

            # 履歴保存オプション
            if save_history:
                st.session_state["chat_history"].append({"role":"user","content":user_input,"ts":time.time()})
                if assistant_text:
                    st.session_state["chat_history"].append({"role":"assistant","content":assistant_text,"ts":time.time()})
            else:
                # 一時的に表示する（履歴に保存しない）: append to a transient list for immediate rendering
                temp_h = list(st.session_state["chat_history"])
                temp_h.append({"role":"user","content":user_input,"ts":time.time()})
                if assistant_text:
                    temp_h.append({"role":"assistant","content":assistant_text,"ts":time.time()})
                # 上書き表示 by setting a temporary session var then rendering
                st.session_state.setdefault("_temp_display", temp_h)

            # 表示応答
            if assistant_text:
                st.success("応答を受信しました。")
            else:
                st.warning("応答が抽出できませんでした。生JSONを参照してください。")

with col_right:
    st.subheader("レスポンス(JSON)")
    if st.session_state.get("last_response"):
        st.json(st.session_state["last_response"])
    else:
        st.info("送信後にここに生JSONが表示されます。")

    st.markdown("---")
    st.subheader("履歴管理")
    st.write("会話履歴はセッション内に保存されます。保存しない設定にすると送信後に履歴へ残りません。")
    if st.button("履歴をエクスポート (JSON)"):
        st.download_button("Download history", data=json.dumps(st.session_state["chat_history"], ensure_ascii=False, indent=2), file_name="chat_history.json", mime="application/json")

# トランジェント表示（保存しない場合の即時表示）
if st.session_state.get("_temp_display"):
    st.markdown("---")
    st.subheader("送信時の一時会話表示（保存しないモード）")
    for m in st.session_state["_temp_display"]:
        if m["role"] == "user":
            st.markdown(f"**User**:")
            st.write(m["content"])
        else:
            st.markdown(f"**Assistant**:")
            st.code(m["content"], language="text")
    # 一度表示したら消す
    st.session_state.pop("_temp_display", None)

# フッタ
st.markdown("---")
st.caption("注意: モデルやパラメータは環境に応じて調整してください。エラー時は Model を変更するか温度をオフにして再試行してください。")
