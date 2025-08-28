# -*- coding: utf-8 -*-

import json
import os
import re
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st
from gitapi.api import GitApiClient
from openai import OpenAI

# =========================
# Util / JSON helper
# =========================

def safe_rerun():
    try:
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
    except Exception:
        pass

def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が未設定です。サイドバーで入力してください。")
    return OpenAI(api_key=api_key)

def json_pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)

def extract_codeblock_json(text: str) -> Optional[Any]:
    if "```" in text:
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            lang_or_body = parts[i].strip()
            body = ""
            if "\n" in lang_or_body:
                first, rest = lang_or_body.split("\n", 1)
                lang = first.strip().lower()
                if lang in ("json", "application/json"):
                    body = rest
            else:
                body = lang_or_body
            if body.strip():
                try:
                    return json.loads(body)
                except Exception:
                    pass
    return None

def extract_first_json(text: str) -> Optional[Any]:
    start_idx = None
    stack = []
    for i, ch in enumerate(text):
        if ch in "{[":
            if not stack:
                start_idx = i
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            open_ch = stack.pop()
            if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                stack = []
                start_idx = None
                continue
            if not stack and start_idx is not None:
                candidate = text[start_idx:i+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    start_idx = None
                    continue
    return None

def force_parse_json(text: str) -> Optional[Any]:
    j = extract_codeblock_json(text)
    if j is not None:
        return j
    if "```" in text:
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            body = parts[i]
            try:
                return json.loads(body)
            except Exception:
                continue
    j = extract_first_json(text)
    if j is not None:
        return j
    m = re.search(r"JSON\s*:\s*(\{.*|\[.*)", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None

def call_llm(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int, seed: Optional[int]) -> str:
    client = get_openai_client()
    kwargs = {"model": model, "temperature": temperature, "max_tokens": max_tokens}
    if seed is not None:
        kwargs["seed"] = seed
    resp = client.chat.completions.create(messages=messages, **kwargs)
    return resp.choices[0].message.content


# =========================
# Data structures
# =========================

@dataclass
class Agent:
    id: str
    name: str = "Agent"
    user_prompt: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 1200
    seed: Optional[int] = None
    enabled: bool = True
    history: List[Dict[str, str]] = field(default_factory=list)
    last_raw: str = ""
    last_json: Any = None

def new_agent(name: str = "Agent") -> Agent:
    return Agent(id=str(uuid.uuid4())[:8], name=name)

def ensure_state():
    if "OPENAI_API_KEY" not in st.session_state:
        # Try to load API key from api.key file located next to this script or in the current working directory.
        key_val = ""
        try:
            script_dir = os.path.dirname(__file__)
        except Exception:
            script_dir = os.getcwd()
        candidates = [os.path.join(script_dir, "api.key"), os.path.join(os.getcwd(), "api.key"), "api.key"]
        for p in candidates:
            try:
                if p and os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as fh:
                        key_val = fh.read().strip()
                    if key_val:
                        break
            except Exception:
                # ignore read errors and try next
                key_val = key_val or ""
        st.session_state.OPENAI_API_KEY = key_val or ""
    if "grid" not in st.session_state:
        st.session_state.grid = [[new_agent("Agent 1")]]
    if "loop_index" not in st.session_state:
        st.session_state.loop_index = 0
    if "current_row" not in st.session_state:
        st.session_state.current_row = 0
    if "current_loop_steps" not in st.session_state:
        st.session_state.current_loop_steps = []
    if "history_loops" not in st.session_state:
        st.session_state.history_loops = []
    if "global_kv" not in st.session_state:
        st.session_state.global_kv = {}
    if "defaults" not in st.session_state:
        st.session_state.defaults = {
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 1200,
            "seed": None,
            "history_depth": 5,
            "columns_per_row": 4,
        }
    if "ui_target_agent_id" not in st.session_state:
        st.session_state.ui_target_agent_id = None
    if "loaded_sample" not in st.session_state:
        st.session_state.loaded_sample = False
    if "view" not in st.session_state:
        st.session_state.view = "main"  # "main" | "detail"
        # initialize doc-related KV if missing
        init_doc_kv_if_missing()


# ---------- response builder (for easy path access) ----------

def snapshot_current_grid_meta() -> List[List[Dict[str, Any]]]:
    return [[{"id": a.id, "name": a.name} for a in row] for row in st.session_state.grid]


def init_doc_kv_if_missing():
    """
    Initialize document-related keys in st.session_state.global_kv if they are missing.
    Called once on startup.
    """
    kv = st.session_state.get("global_kv")
    if kv is None:
        st.session_state.global_kv = {}
        kv = st.session_state.global_kv
    kv.setdefault("doc_title", "富岳運用 Runbook: ジョブACLと障害対応")
    kv.setdefault("doc_outline", ["# タイトル", "## 概要", "## 手順", "## 検証", "## 参考"])
    kv.setdefault("style_rules", [
        "見出しは『## 』から始める（H1はタイトルのみ）",
        "禁止語: 「等」「など」",
        "コマンドはコードブロックで表記し、行頭に$は付けない",
        "手順は番号付きリスト（1. 2. 3. ...）",
        "環境依存値は <VAR_*> で表記"
    ])
    kv.setdefault("lint_rules", [
        "冗長な表現を簡潔にする",
        "半角英数を使用",
        "時刻は JST 表記（YYYY-MM-DD HH:MM JST）",
        "固有名詞は正式名称で統一（pjsub, pjstat など）"
    ])
    kv.setdefault("target_doc", "# タイトル\n\n(初期ドラフト未作成)\n")
    kv.setdefault("sources_slack", [])
    kv.setdefault("sources_confluence", [])
    kv.setdefault("sources_pdfs", [])
    kv.setdefault("repo_info", {"root": "/repo/docs", "path": "runbooks/job-acl.md"})


def _repo_workspace_root() -> str:
    """Return the filesystem path to the repository workspace root.
    Heuristics: REPO_ROOT env var -> global_kv.repo_info.root (if absolute) -> two levels up from this file.
    """
    # 1) env
    env_root = os.environ.get("REPO_ROOT")
    if env_root:
        return os.path.abspath(env_root)
    # 2) global_kv
    try:
        rv = st.session_state.global_kv.get("repo_info", {}).get("root")
        if rv:
            # if absolute path, use as-is; if relative, make relative to workspace
            if os.path.isabs(rv):
                return os.path.abspath(rv)
            # relative: treat relative to workspace two levels up
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            return os.path.abspath(os.path.join(base, rv.lstrip('/\\')))
    except Exception:
        pass
    # 3) default: two levels up from this file (repo root)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def detect_local_git_branch() -> Optional[str]:
    """Try to detect current branch name from .git/HEAD in the repo root.

    Returns branch name like 'main' or None if not detectable.
    """
    try:
        root = _repo_workspace_root()
        head_path = os.path.join(root, '.git', 'HEAD')
        if os.path.exists(head_path):
            with open(head_path, 'r', encoding='utf-8') as fh:
                txt = fh.read().strip()
            if txt.startswith('ref:'):
                # format: ref: refs/heads/main
                ref = txt.split(None, 1)[1].strip()
                parts = ref.split('/')
                if parts:
                    return parts[-1]
            # detached HEAD or other format
            return None
    except Exception:
        return None
    return None


def repo_fs_path(path_in_repo: str) -> str:
    """Return absolute filesystem path for a repo-relative path (path_in_repo).
    """
    base = _repo_workspace_root()
    return os.path.abspath(os.path.join(base, path_in_repo.lstrip('/\\')))


def read_repo_file(path_in_repo: str) -> tuple[bool, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Return (ok, content_or_error, used_path, source_display, target_url, git_message).

    target_url: the raw URL or API URL used to fetch (if remote) or absolute local path
    git_message: when local, attempt to detect HEAD sha and last commit message
    """
    p = repo_fs_path(path_in_repo)
    try:
        # If remote repo is configured, prefer remote fetch (file_path first)
        cfg = load_github_config()
        token = cfg.get("token")
        owner = cfg.get("owner")
        repo = cfg.get("repo")
        base_branch = cfg.get("base_branch") or 'main'
        if owner and repo:
            try:
                client = GitApiClient(token=(token or ""), owner=owner, repo=repo)
                tried_paths = []
                for candidate in (cfg.get("file_path"), path_in_repo):
                    if not candidate:
                        continue
                    cand = candidate.lstrip("/\\")
                    if cand in tried_paths:
                        continue
                    tried_paths.append(cand)
                    try:
                        remote = client.get_file(cand, ref=base_branch)
                    except Exception as e:
                        remote = None
                        last_err = str(e)
                    else:
                        last_err = None
                    if remote is not None:
                        src = f"{owner}/{repo}@{base_branch}"
                        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{base_branch}/{cand}"
                        return True, remote, cand, src, raw_url, None
                # remote not found; fall back to local if exists
            except Exception:
                # proceed to local fallback
                pass

        # local fallback
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as fh:
                content = fh.read()
            branch = detect_local_git_branch()
            src = f"local ({branch})" if branch else 'local'
            # attempt to get local reflog message
            git_msg = None
            try:
                root = _repo_workspace_root()
                if branch:
                    logs_path = os.path.join(root, '.git', 'logs', 'refs', 'heads', branch)
                    if os.path.exists(logs_path):
                        with open(logs_path, 'r', encoding='utf-8', errors='ignore') as lf:
                            lines = [ln.strip() for ln in lf.readlines() if ln.strip()]
                        if lines:
                            last = lines[-1]
                            if '\t' in last:
                                git_msg = last.split('\t', 1)[1]
                            else:
                                parts = last.split()
                                if parts:
                                    git_msg = parts[-1]
            except Exception:
                git_msg = None
            local_abs = p
            return True, content, path_in_repo, src, local_abs, git_msg

        return False, f"not found: {p}", None, None, None, None
    except Exception as e:
        return False, str(e), None, None, None, None


def write_repo_file(path_in_repo: str, content: str) -> tuple[bool, Optional[str]]:
    p = repo_fs_path(path_in_repo)
    try:
        # If GitHub config with token is available, attempt remote push (branch + file + PR)
        try:
            cfg = load_github_config()
            token = cfg.get("token")
            owner = cfg.get("owner")
            repo = cfg.get("repo")
            # read-branch used as the base for creating the write branch if needed
            base_for_create = cfg.get("base_branch_read") or cfg.get("base_branch") or "main"
            write_branch = cfg.get("base_branch_write") or cfg.get("base_branch") or None
            file_path = cfg.get("file_path") or path_in_repo
            if token and owner and repo:
                client = GitApiClient(token=token, owner=owner, repo=repo)
                # target branch to update/create
                branch_name = write_branch if write_branch else f"streamlit/write_{uuid.uuid4().hex[:8]}"
                try:
                    # attempt to create the branch from base_for_create; if it exists this may error which we ignore
                    client.create_branch(branch_name, base_for_create)
                except Exception:
                    # branch already exists or creation failed; continue and try to put file
                    pass
                commit_msg = f"Update {file_path} via StreamIt"
                try:
                    client.put_file(path=file_path, content=content, branch=branch_name, message=commit_msg)
                    return True, f"remote updated branch {branch_name}"
                except Exception as e:
                    # fall back to local write if remote push fails
                    fallback_err = str(e)
        except Exception:
            fallback_err = None

        # local write fallback
        d = os.path.dirname(p)
        os.makedirs(d, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as fh:
            fh.write(content)
        info = p
        if 'fallback_err' in locals() and fallback_err:
            info = f"local write succeeded, remote push failed: {fallback_err}"
        return True, info
    except Exception as e:
        return False, str(e)


def set_kv_json(key: str, text: str):
    try:
        val = json.loads(text)
        st.session_state.global_kv[key] = val
        return True, None
    except Exception as e:
        return False, str(e)


def set_sample_doc_kv():
    kv = st.session_state.global_kv
    kv["doc_title"] = "富岳運用 Runbook: ジョブACLと障害対応"
    kv["doc_outline"] = ["# 富岳運用 Runbook: ジョブACLと障害対応", "## 概要", "## 手順", "## 検証", "## 参考"]
    kv["style_rules"] = [
        "見出しは『## 』から始める（H1はタイトルのみ）",
        "禁止語: 「等」「など」",
        "コマンドはコードブロックで表記し、行頭に$は付けない",
        "手順は番号付きリスト（1. 2. 3. ...）",
        "環境依存値は <VAR_*> で表記"
    ]
    kv["lint_rules"] = [
        "冗長な表現を簡潔にする",
        "半角英数を使用",
        "時刻は JST 表記（YYYY-MM-DD HH:MM JST）",
        "固有名詞は正式名称で統一（pjsub, pjstat など)"
    ]
    kv["sources_slack"] = [
        "[2024-08-03 10:12] ops-a: ACL更新は /etc/pj_acl を編集して pj_flush すれば反映？",
        "[2024-08-03 10:13] ops-b: 反映は pjctl reload-acl コマンド推奨。手順Runbook古いかも。",
        "[2024-08-20 22:31] oncall: ジョブ投入失敗(ACL)の暫定回避手順を追記したい。"
    ]
    kv["sources_confluence"] = [
        {"title": "旧Runbook: ACL更新手順", "url": "https://conf/ops/acl", "body": "ACLの更新は /etc/pj_acl を編集後、サービス再起動（古い）"},
        {"title": "障害対応ログ（2024-08-20）", "url": "https://conf/ops/incidents/20240820", "body": "ACL不整合でジョブ投入失敗。対処: pjctl reload-acl 実行。検証: pjstat で権限確認。"}
    ]
    kv["sources_pdfs"] = [
        "FugakuOpsGuide_v3.2.pdf p.45 ACL運用: reload-aclコマンドで反映。旧手順は非推奨。"
    ]
    kv["target_doc"] = (
        "# 富岳運用 Runbook: ジョブACLと障害対応\n\n"
        "## 概要\n"
        "ジョブACLの更新手順と、ACLに起因するジョブ投入失敗時の対処をまとめる。\n\n"
        "## 手順\n"
        "1. ACL定義を更新する（場所は要確認）。\n"
        "2. サービスを再起動して反映する。\n\n"
        "## 検証\n"
        "- pjstat で確認等。\n\n"
        "## 参考\n"
        "- 旧Runbook: https://conf/ops/acl\n"
    )


def load_sample_doc_workflow():
    # 5役: 収集（3並列）→ 整理/リント → 選別/更新
    def p_collector(source_key: str, label: str) -> str:
        return (
            "あなたは収集(Collector)エージェントです。Supervisorの指示に従い、指定ソースから短く独立したMarkdownチャンクを作成してください。\n"
            f"対象ソース: Context.global_kv.{source_key} (配列またはテキスト)。\n"
            "要件:\n"
            "- 出力は必ず JSON を返すこと。トップレベルに message.type='collect' を含めること。\n"
            "- message.chunks は配列。各チャンクは {source,label,title,path,ts,tags,content_md} を持つこと。\n"
            "  - content_md は Markdown 片（短く、1-5 行程度の節ごと）。\n"
            "  - tags は '手順','検証','参考','注意' などの分類を最大3つまで含めること。\n"
            "- 重複は許容するが、同一内容は軽くノート（duplicates: true）を付けると親切。\n"
            f"- チャンクは label='{label}' をメタ情報として付与してください。\n"
            "出力例:\n"
            "{\n"
            "  \"message\":{\"type\":\"collect\", \"chunks\": [ {\"source\":\"slack\", \"label\":\"Slackログ\", \"title\":\"ジョブACLの失敗ログ\", \"content_md\":\"...\"} ] }\n"
            "}\n"
        )

    p_normalizer = (
        "あなたは正規化/統合(Normalizer)エージェントです。Collectors の出力を受け取り、重複排除、表記統一、そしてContext.global_kv.doc_outline の適切なセクションに割り当てるための『パッチ提案』を生成してください。\n"
        "入力: 直前の collectors の message.chunks (response の該当箇所)。\n"
        "要件:\n"
        "- 出力は JSON で message.type='proposals' を含めること。\n"
        "- message.patches は配列で、要素は {op, anchor, section, before, after, rationale, risk, score}。\n"
        "  - op は add|replace|remove。anchor は見出し文字列で、存在しない場合は新規作成案を出すこと。\n"
        "  - before/after は差分が明確になるように markdown 断片を入れること。\n"
        "- 各パッチに対して短い rationale とリスク記述を付け、score を 0-1 で見積もること。\n"
        "- 最大5件まで、かつ具体的に適用可能であること。\n"
    )

    p_linter = (
        "あなたはReader/Linter 役割です。Normalizer のパッチを受け、各パッチについて『表記・スタイル・重複・明瞭さ』の観点で採点し、必要なら修正パッチを追加で提案してください。\n"
        "- 入力: Normalizer の message.patches。Context.global_kv.style_rules と lint_rules を参照。\n"
        "- 出力: message.type='lints'、message.patches=[{op,anchor,section,before,after,rationale,rule,score}]。\n"
        "- また各 incoming patch に対して score を 0-1 で付与し、reader_score フィールドで総合評価を返すこと。\n"
        "- 最大5件まで。\n"
    )

    p_judge = (
        "あなたはJudge/Selector 役です。Normalizer と Linter の提案を比較し、採用するパッチのみを選別してください。\n"
        "- 入力: Normalizer と Linter の message.patches とそれぞれの score。\n"
        "- 基準: score、rationale、risk を総合して採否を決定。\n"
        "- 出力: message.type='selection', message.accepted=[{from_col, index, reason, final_score}]。\n"
        "- from_col: 0=Normalizer,1=Linter。final_score は 0-1 の正規化スコア。\n"
        "- 最大採択数: 5 件。\n"
    )

    p_editor = (
        "あなたはエディタ/コミッターです。\n"
        "- Judge の採択結果は response[0][-1][0].message.accepted にあります。\n"
        "- Normalizer と Linter の原パッチは response[0][-2][0/1].message.patches にあります。\n"
        "- Context.global_kv.target_doc に対して、採択パッチを順次適用して新しいMarkdown全文を生成し、kv_patch.target_doc に格納してください。\n"
        "- アンカー（anchor）が見つからなければ、Context.global_kv.doc_outlineの順序に従いセクションを新設して差し込んで構いません。\n"
        "- 出力:\n"
        "  message.type='commit'\n"
        "  message.commit_summary='...'（要約）\n"
        "  message.diff='...'（短い差分プレビュー）\n"
        "- kv_patch.target_doc に最終Markdown全文を必ず入れてください。\n"
    )

    st.session_state.grid = [
        [new_agent("Collector Slack"), new_agent("Collector Confluence"), new_agent("Collector PDF")],
        [new_agent("Normalizer"), new_agent("Linter")],
        [new_agent("Judge"), new_agent("Editor")]
    ]

    # プロンプト設定
    st.session_state.grid[0][0].user_prompt = p_collector("sources_slack", "Slackログ")
    st.session_state.grid[0][1].user_prompt = p_collector("sources_confluence", "Confluence/古いRunbook/障害ログ")
    st.session_state.grid[0][2].user_prompt = p_collector("sources_pdfs", "PDF資料")

    st.session_state.grid[1][0].user_prompt = p_normalizer
    st.session_state.grid[1][1].user_prompt = p_linter

    st.session_state.grid[2][0].user_prompt = p_judge
    st.session_state.grid[2][1].user_prompt = p_editor

    # モデル/温度はデフォルトから継承
    set_sample_doc_kv()
    st.session_state.current_row = 0
    st.session_state.loaded_sample = True
    safe_rerun()


# --- GitHub config loader (centralized) ---
def load_github_config() -> Dict[str, Optional[str]]:
    """Load GitHub related config with precedence: env -> secrets.toml -> session global_kv -> defaults.

    Returns a dict with keys: token, owner, repo, base_branch, file_path
    """
    cfg: Dict[str, Optional[str]] = {
        "token": None,
        "owner": None,
        "repo": None,
        "base_branch": None,
        "base_branch_read": None,
        "base_branch_write": None,
        "file_path": None,
    }

    # 1) try secrets.toml first (user requested: token stored in secrets.toml)
    toml_loader = None
    try:
        import tomllib as toml_loader  # type: ignore
    except Exception:
        try:
            import toml as toml_loader  # type: ignore
        except Exception:
            toml_loader = None

    if toml_loader is not None:
        s_path = os.path.join(os.path.dirname(__file__), "secrets.toml")
        try:
            if os.path.exists(s_path):
                with open(s_path, "r", encoding="utf-8") as fh:
                    text = fh.read()
                try:
                    data = toml_loader.loads(text)
                except Exception:
                    try:
                        import toml as _toml
                        data = _toml.loads(text)
                    except Exception:
                        data = {}
                cfg["token"] = data.get("GITHUB_TOKEN") or data.get("LLM_API_KEY") or cfg["token"]
                cfg["owner"] = data.get("GITHUB_OWNER") or cfg["owner"]
                cfg["repo"] = data.get("GITHUB_REPO") or cfg["repo"]
                cfg["base_branch"] = data.get("BASE_BRANCH") or cfg["base_branch"]
                cfg["base_branch_read"] = data.get("BASE_BRANCH_READ") or data.get("BASE_BRANCH") or cfg.get("base_branch_read")
                cfg["base_branch_write"] = data.get("BASE_BRANCH_WRITE") or data.get("BASE_BRANCH") or cfg.get("base_branch_write")
                cfg["file_path"] = data.get("FILE_PATH") or cfg["file_path"]
        except Exception:
            pass

    # 2) overlay environment variables (use when secrets.toml not present)
    cfg["token"] = cfg["token"] or os.environ.get("GITHUB_TOKEN")
    cfg["owner"] = cfg["owner"] or os.environ.get("GITHUB_OWNER")
    cfg["repo"] = cfg["repo"] or os.environ.get("GITHUB_REPO")
    cfg["base_branch"] = cfg["base_branch"] or os.environ.get("BASE_BRANCH")
    cfg["base_branch_read"] = cfg.get("base_branch_read") or os.environ.get("BASE_BRANCH_READ")
    cfg["base_branch_write"] = cfg.get("base_branch_write") or os.environ.get("BASE_BRANCH_WRITE")
    cfg["file_path"] = cfg["file_path"] or os.environ.get("FILE_PATH")

    # 3) fallback to session_state.global_kv
    gkv = st.session_state.get("global_kv", {}) or {}
    cfg["token"] = cfg["token"] or gkv.get("GITHUB_TOKEN") or gkv.get("github_token")
    cfg["owner"] = cfg["owner"] or gkv.get("GITHUB_OWNER") or gkv.get("github_owner")
    cfg["repo"] = cfg["repo"] or gkv.get("GITHUB_REPO") or gkv.get("github_repo")
    cfg["base_branch"] = cfg["base_branch"] or gkv.get("BASE_BRANCH") or gkv.get("base_branch")
    cfg["base_branch_read"] = cfg.get("base_branch_read") or gkv.get("BASE_BRANCH_READ") or gkv.get("base_branch_read")
    cfg["base_branch_write"] = cfg.get("base_branch_write") or gkv.get("BASE_BRANCH_WRITE") or gkv.get("base_branch_write")
    # final default
    cfg["base_branch_read"] = cfg.get("base_branch_read") or cfg.get("base_branch") or "main"
    cfg["base_branch_write"] = cfg.get("base_branch_write") or cfg.get("base_branch") or "main"
    cfg["file_path"] = cfg["file_path"] or gkv.get("FILE_PATH") or gkv.get("file_path") or "docs/runbooks/job-acl.md"

    # final hardcoded fallbacks
    cfg["owner"] = cfg["owner"] or "daisuke0323c-code"
    cfg["repo"] = cfg["repo"] or "runbook-sandbox"
    return cfg

def build_response_row_from_agents(row_idx_abs: int) -> List[Dict[str, Any]]:
    cells = []
    for c, agent in enumerate(st.session_state.grid[row_idx_abs]):
        cells.append({
            "message": agent.last_json,
            "raw": agent.last_raw or None,
            "agent_id": agent.id,
            "name": agent.name,
            "row": row_idx_abs,
            "col": c
        })
    return cells

def build_indexed_response_current(row_idx: int) -> Dict[str, List[Dict[str, Any]]]:
    mapping: Dict[str, List[Dict[str, Any]]] = {}
    for abs_row in range(0, row_idx):
        rel = abs_row - row_idx
        mapping[str(rel)] = build_response_row_from_agents(abs_row)
    return mapping

def build_indexed_response_prev_loop() -> Dict[str, List[Dict[str, Any]]]:
    if not st.session_state.history_loops:
        return {}
    last = st.session_state.history_loops[-1]
    steps = last.get("steps", [])
    snap = last.get("grid_snapshot", [])
    mapping: Dict[str, List[Dict[str, Any]]] = {}
    for i in range(len(steps)):
        abs_idx = len(steps) - 1 - i
        rel = -1 - i
        step_map = steps[abs_idx]
        meta_row = snap[abs_idx] if abs_idx < len(snap) else []
        row_cells = []
        for c, meta in enumerate(meta_row):
            j = step_map.get(meta["id"])
            row_cells.append({
                "message": j,
                "raw": None,
                "agent_id": meta["id"],
                "name": meta.get("name"),
                "row": abs_idx,
                "col": c
            })
        mapping[str(rel)] = row_cells
    return mapping

def build_context_for_row(row_idx: int) -> Dict[str, Any]:
    resp_cur = build_indexed_response_current(row_idx)
    resp_prev = build_indexed_response_prev_loop()
    response = {"0": resp_cur, "-1": resp_prev}
    help_text = (
        "前結果アクセス例:\n"
        "- 現ループ 直前行 col=0: response[0][-1][0].message\n"
        "- 現ループ 2つ上 col=1: response[0][-2][1].message\n"
        "- 直前ループ 最終行 col=0: response[-1][-1][0].message\n"
        "- 短縮: resp[-1][0].message は response[0][-1][0].message と同じ"
    )
    return {
        "loop_index": st.session_state.loop_index,
        "current_row": row_idx,
        "response": response,
        "resp": resp_cur,
        "global_kv": deepcopy(st.session_state.global_kv),
        "help": help_text,
    }


# =========================
# LLM run with unified output schema
# =========================

SCHEMA_ENFORCER = (
    "以下の制約を厳守して応答してください:\n"
    "- 出力は有効なJSONのみ（前置き・説明・コードブロック禁止）\n"
    "- 統一スキーマ:\n"
    "{\n"
    "  \"ok\": true,\n"
    "  \"message\": { ... },\n"
    "  \"kv_patch\": { ... },\n"
    "  \"meta\": { ... }\n"
    "}\n"
)

def normalize_unified(parsed: Any) -> Dict[str, Any]:
    if isinstance(parsed, dict):
        if "message" in parsed or "ok" in parsed or "kv_patch" in parsed or "meta" in parsed:
            msg = parsed.get("message")
            if msg is None:
                msg = {k: v for k, v in parsed.items() if k not in ("ok", "kv_patch", "meta")}
            return {
                "ok": bool(parsed.get("ok", True)),
                "message": msg,
                "kv_patch": parsed.get("kv_patch") or parsed.get("__kv_patch__", {}),
                "meta": parsed.get("meta", {})
            }
        else:
            return {"ok": True, "message": parsed, "kv_patch": {}, "meta": {}}
    elif parsed is None:
        return {"ok": False, "message": {"error": "JSON parse failed"}, "kv_patch": {}, "meta": {}}
    else:
        return {"ok": True, "message": {"value": parsed}, "kv_patch": {}, "meta": {}}

def run_agent(agent: Agent, row_idx: int) -> Agent:
    if not agent.enabled:
        return agent

    ctx = build_context_for_row(row_idx)
    user_prompt = f"""{SCHEMA_ENFORCER}

指示:
{agent.user_prompt}

[Context]
{json_pretty(ctx)}
"""
    messages = [{"role": "user", "content": user_prompt}]
    hist = agent.history[-40:] if len(agent.history) > 40 else agent.history
    messages = hist + messages

    model = agent.model or st.session_state.defaults["model"]
    temperature = agent.temperature if agent.temperature is not None else st.session_state.defaults["temperature"]
    max_tokens = int(agent.max_tokens or st.session_state.defaults["max_tokens"])
    seed = agent.seed if agent.seed is not None else st.session_state.defaults["seed"]

    try:
        raw = call_llm(messages, model=model, temperature=temperature, max_tokens=max_tokens, seed=seed)
    except Exception as e:
        raw = f'{{"ok":false,"message":{{"error":"LLM呼び出しエラー: {e}"}}, "kv_patch":{{}}, "meta":{{}}}}'

    parsed = force_parse_json(raw)
    unified = normalize_unified(parsed)

    agent.history.append({"role": "user", "content": user_prompt})
    agent.history.append({"role": "assistant", "content": raw})
    if len(agent.history) > 60:
        agent.history = agent.history[-60:]

    agent.last_raw = raw
    agent.last_json = unified
    return agent


# =========================
# Loop execution
# =========================

def step_execute(row_idx: int):
    if row_idx >= len(st.session_state.grid):
        return
    row = st.session_state.grid[row_idx]
    step_result: Dict[str, Any] = {}
    for col, agent in enumerate(row):
        if not agent.enabled:
            continue
        updated = run_agent(agent, row_idx)
        st.session_state.grid[row_idx][col] = updated
        step_result[updated.id] = updated.last_json
        if isinstance(updated.last_json, dict):
            patch = updated.last_json.get("kv_patch") or updated.last_json.get("__kv_patch__")
            if isinstance(patch, dict):
                st.session_state.global_kv.update(patch)

    if len(st.session_state.current_loop_steps) == row_idx:
        st.session_state.current_loop_steps.append(step_result)
    elif len(st.session_state.current_loop_steps) > row_idx:
        st.session_state.current_loop_steps[row_idx] = step_result
    else:
        while len(st.session_state.current_loop_steps) < row_idx:
            st.session_state.current_loop_steps.append({})
        st.session_state.current_loop_steps.append(step_result)

def complete_loop_if_needed(row_idx: int):
    if row_idx == len(st.session_state.grid) - 1:
        grid_snapshot = snapshot_current_grid_meta()
        st.session_state.history_loops.append({
            "loop_index": st.session_state.loop_index,
            "steps": deepcopy(st.session_state.current_loop_steps),
            "grid_snapshot": grid_snapshot
        })
        st.session_state.loop_index += 1
        st.session_state.current_loop_steps = []
        st.session_state.current_row = 0
    else:
        st.session_state.current_row = row_idx + 1

def run_one_step():
    if not st.session_state.grid or not st.session_state.grid[0]:
        st.warning("エージェントがありません。行/列を追加してください。")
        return
    row_idx = st.session_state.current_row
    step_execute(row_idx)
    complete_loop_if_needed(row_idx)

def run_to_end():
    if not st.session_state.grid or not st.session_state.grid[0]:
        st.warning("エージェントがありません。行/列を追加してください。")
        return
    while True:
        row_idx = st.session_state.current_row
        step_execute(row_idx)
        if row_idx == len(st.session_state.grid) - 1:
            complete_loop_if_needed(row_idx)
            break
        else:
            st.session_state.current_row = row_idx + 1

def run_n_loops(n: int):
    for _ in range(n):
        run_to_end()

def reset_state(keep_grid=True):
    if not keep_grid:
        st.session_state.grid = [[new_agent("Agent 1")]]
    for row in st.session_state.grid:
        for agent in row:
            agent.history = []
            agent.last_raw = ""
            agent.last_json = None
    st.session_state.current_row = 0
    st.session_state.current_loop_steps = []
    st.session_state.history_loops = []
    st.session_state.loop_index = 0
    st.session_state.global_kv = {}
    st.session_state.loaded_sample = False
    st.session_state.view = "main"
    st.session_state.ui_target_agent_id = None


# =========================
# UI: Main / Detail
# =========================

st.set_page_config(page_title="マルチエージェント・グリッド（白UI・テーブル）", layout="wide")

# --- CSS Styles ---
st.markdown("""
<style>
    /* Ensure Streamlit sidebar is visible despite other CSS overrides */
    .css-1d391kg, .css-1d391kg * {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    .stSidebar, .css-1d391kg, .css-1d391kg .css-1v3fvcr, .css-1d391kg > div,
    [data-testid="stSidebar"], aside[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        width: 300px !important;
        min-width: 220px !important;
        transform: none !important;
    }
    /* --- Base --- */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FAFAFA;
    }
    .stTextArea textarea {
        background-color: #1E1E1E;
        color: #FAFAFA;
        border: 1px solid #444;
    }
    /* --- Main Grid --- */
    .agent-cell {
        background-color: transparent;
        /* border removed per user request */
        border: none !important;
        border-radius: 8px;
    padding: 0.15rem 0.25rem 0.45rem 0.25rem;
    margin-top: 0 !important;
        margin-bottom: 0.75rem;
        min-height: auto; /* avoid large empty area above result */
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: stretch;
    }
    .agent-cell.has-agent {
        background-color: #1c1c24;
        /* border removed per user request */
        border: none !important;
        align-items: stretch;
        justify-content: flex-start;
        padding: 0.6rem;
    }
    .agent-cell.empty {
        background-color: transparent;
        /* border removed per user request */
        border: none !important;
        align-items: center;
        justify-content: center;
    }
    .add-tile .stButton button {
        background-color: transparent;
        color: #888;
        border: none;
        font-size: 1rem;
    height: 150px;
    width: 100% !important;
    }
    /* --- Agent Box --- */
    .agent-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .agent-name {
        font-weight: bold;
        font-size: 1.1rem;
    }
    .agent-status {
        font-family: monospace;
        font-size: 0.9rem;
        color: #a0a0a0;
        background-color: #333;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .result-preview {
        background-color: #262730;
        border: 1px dashed #444;
        border-radius: 4px;
    padding: 0.6rem;
    margin: 0.25rem 0 0.5rem 0;
        min-height: 80px;
        font-family: monospace;
        font-size: 0.85rem;
        color: #e0e0e0;
        white-space: pre-wrap;
        overflow-y: auto;
        max-height: 240px;
    }
    .result-preview.selected {
        background-color: #33343a !important;
        border-color: #6b7280 !important;
    }
    .result-preview.no-result {
        color: #888;
        font-style: italic;
    }
    .tile-area {
        width: 100%;
        height: 180px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 8px;
        box-sizing: border-box;
        height: auto;
        min-height: 140px;
    }
    .tile-area .agent-name {
        font-weight: bold;
        color: #f5f5f5;
        margin-bottom: 6px;
        text-align: left;
        width: 100%;
    }
    .detail-btn .stButton button {
        background-color: #4a5568;
        border: 1px solid #666;
        color: #fff;
        padding: 0.4rem 0.8rem;
        font-size: 0.9rem;
        border-radius: 4px;
    }
    .detail-btn .stButton button:hover {
        background-color: #5a6578;
    }
    .agent-ctrl {
        margin-top: 1rem;
    }
    .step-title {
        border-bottom: 2px solid #4a4a52;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .row-sep {
        margin-bottom: 2rem;
    }
    /* --- 修正: タイル内に Streamlit ウィジェットや code を収めるためのルール --- */
    /* make tile buttons compact (icon-size) */
    .agent-cell .stButton button,
    .agent-box .stButton button {
        width: auto !important;
        box-sizing: border-box !important;
        margin: 4px 6px !important;
        padding: 6px 8px !important;
        font-size: 0.9rem !important;
    }
    /* keep add-tile button full width */
    .add-tile .stButton button {
        width: 100% !important;
    }
    /* チェックボックスを左寄せしてボタンと並ばないようにする */
    .agent-cell .stCheckbox,
    .agent-cell .stCheckbox .stMarkdown {
        display: block !important;
        margin: 6px 0 !important;
    }
    /* Hide empty agent-cell elements inserted into Streamlit's markdown wrapper */
    .stMarkdown .agent-cell:empty {
        display: none !important;
    }
    .stMarkdown > div.agent-cell:empty {
        display: none !important;
    }
    /* aggressively hide empty agent-cell elements and collapse their space */
    .stElementContainer .agent-cell:empty,
    .stElementContainer.element-container .agent-cell:empty,
    .stMarkdown .agent-cell:empty {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
    }
    /* specifically hide the nested empty agent-cell inside Streamlit element containers
       matches the structure the user reported and prevents the thin pill from appearing */
    .stElementContainer.element-container .stMarkdown > div > div.agent-cell {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    /* hide any empty divs inside a tile (small pills injected by streamlit) */
    .agent-cell > div:empty {
        display: none !important;
    }
    .agent-cell > div:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    /* make controller widgets inline and compact */
    .agent-cell .stButton, .agent-cell .stCheckbox {
        display: inline-block !important;
        vertical-align: middle !important;
    }
    .agent-cell .stCheckbox .stMarkdown {
        display: inline-block !important;
        margin-left: 6px !important;
        margin-right: 8px !important;
        font-size: 0.9rem !important;
    }
    /* code / pre ブロックをタイル内でスクロール可能にする */
    .agent-cell pre,
    .agent-cell .stCodeBlock,
    .agent-cell .stMarkdown code {
        max-height: 180px !important;
        overflow: auto !important;
        white-space: pre-wrap !important;
        word-break: break-word !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    /* also remove top margin for markdown/code inside the tile to avoid a slim light band */
    .agent-cell .stMarkdown,
    .agent-cell .stCodeBlock,
    .agent-cell pre {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    /* タイル内の result-preview の余白を確保 */
    .agent-cell .result-preview,
    .agent-cell .result-preview.no-result {
        margin-bottom: 12px !important;
    }
    /* === カラム（data-testid="stColumn"）に枠をつける（破線） === */
    /* Target the Streamlit column wrapper which the user pasted in the DOM snippet */
    .stColumn[data-testid="stColumn"] {
        /* no outer dashed border per request */
        border: none !important;
        border-radius: 10px !important;
        padding: 6px !important;
        box-sizing: border-box !important;
        background-color: transparent !important;
        transition: background-color 0.12s ease, border-color 0.12s ease;
    }

    /* Dim only the add-button (element keys start with addcol_). Other buttons remain normal. */
    .stElementContainer[class*="st-key-addcol_"] .stButton button {
        opacity: 0.55 !important;
        color: #999 !important;
        background-color: transparent !important;
        border-color: transparent !important;
    }

    /* Brighten the column when it contains agent widgets */
    .stColumn[data-testid="stColumn"]:has(.row-widget) {
        background-color: #1c1c24 !important;
    }

    .stColumn[data-testid="stColumn"]:has(.row-widget) .stButton button {
        opacity: 1 !important;
        color: inherit !important;
        background-color: initial !important;
    }

    /* Fallback: if :has isn't supported, highlight columns that contain a non-empty result-preview */
    .stColumn[data-testid="stColumn"] .result-preview:not(.no-result) {
        background-color: #31353b !important;
        border-color: #9ca3af !important;
        color: #f3f4f6 !important;
    }

    .stColumn[data-testid="stColumn"] .result-preview {
        margin: 6px 0 !important;
    }
</style>
""", unsafe_allow_html=True)

ensure_state()

# --- 追記: UI で参照されるが未定義だった小さなヘルパを追加（NameError 回避） ---
def go_detail(agent_id: str):
    # 詳細画面へ遷移
    st.session_state.ui_target_agent_id = agent_id
    st.session_state.view = "detail"
    safe_rerun()

def go_main():
    # メイン画面へ戻る
    st.session_state.view = "main"
    st.session_state.ui_target_agent_id = None
    safe_rerun()

def find_agent_pos(agent_id: str):
    # grid から (row, col, agent) を探して返す、見つからなければ None
    for r, row in enumerate(st.session_state.grid):
        for c, agent in enumerate(row):
            if agent.id == agent_id:
                return (r, c, agent)
    return None

def load_sample_3step():
    # 単純な 3-step サンプルをセットする
    st.session_state.grid = [
        [new_agent("Worker 1"), new_agent("Worker 2")],
        [new_agent("Judge 1")],
        [new_agent("Editor 1")]
    ]
    st.session_state.current_row = 0
    st.session_state.loaded_sample = True
    safe_rerun()

# --- Sidebar（復活） ---
with st.sidebar:
    st.header("グローバル設定")

    # API Key
    st.session_state.OPENAI_API_KEY = st.text_input(
        "OPENAI_API_KEY",
        value=st.session_state.get("OPENAI_API_KEY", ""),
        type="password",
        help="空欄の場合は環境変数 OPENAI_API_KEY を利用"
    )

    # Defaults 編集
    d = st.session_state.defaults
    st.subheader("デフォルト値")
    d["model"] = st.text_input("Model (既定)", value=d.get("model", "gpt-4o-mini"))
    d["max_tokens"] = int(st.number_input("max_tokens (既定)", min_value=128, max_value=8192, value=int(d.get("max_tokens", 1200)), step=64))
    d["columns_per_row"] = int(st.number_input("列数 (表示ベース)", min_value=1, max_value=12, value=int(d.get("columns_per_row", 4)), step=1))

    temp_enable = st.checkbox("temperature を編集", value=True if d.get("temperature", 0.3) is not None else False)
    if temp_enable:
        d["temperature"] = float(st.slider("temperature", 0.0, 2.0, float(d.get("temperature", 0.3)), 0.1))
    else:
        d["temperature"] = 0.0

    # ループ / 状態表示
    st.markdown("---")
    st.caption("現在ステータス")
    st.write(f"Loop: {st.session_state.loop_index}")
    st.write(f"Current Row: {st.session_state.current_row+1} / {len(st.session_state.grid)}")
    st.write(f"Agents Total: {sum(len(r) for r in st.session_state.grid)}")

    # 操作
    st.markdown("---")
    if st.button("履歴のみ初期化 (Grid保持)"):
        reset_state(keep_grid=True)
        st.success("履歴とKVを初期化しました。")
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()
    if st.button("全リセット (Grid含む)"):
        reset_state(keep_grid=False)
        st.success("全て初期化しました。")
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

    if st.button("3段サンプル配置"):
        load_sample_3step()

    if st.button("Docワークフロー配置 (サンプル)"):
        load_sample_doc_workflow()
        st.success("Docワークフロー（サンプル）を配置しました")

    st.markdown("---")
    st.subheader("設定の保存 / 読み込み")

    SAVE_DIR = os.path.join(os.path.expanduser("~"), ".streamlit_app_saves")
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
    except Exception:
        SAVE_DIR = "."

    def serialize_agent(a: Agent) -> Dict[str, Any]:
        return {
            "id": a.id,
            "name": a.name,
            "user_prompt": a.user_prompt,
            "model": a.model,
            "temperature": a.temperature,
            "max_tokens": a.max_tokens,
            "seed": a.seed,
            "enabled": a.enabled,
            "history": a.history,
            "last_raw": a.last_raw,
            "last_json": a.last_json,
        }

    def deserialize_agent(d: Dict[str, Any]) -> Agent:
        ag = Agent(id=d.get("id", str(uuid.uuid4())[:8]))
        ag.name = d.get("name", ag.name)
        ag.user_prompt = d.get("user_prompt", ag.user_prompt)
        ag.model = d.get("model", ag.model)
        ag.temperature = d.get("temperature", ag.temperature)
        ag.max_tokens = d.get("max_tokens", ag.max_tokens)
        ag.seed = d.get("seed", ag.seed)
        ag.enabled = d.get("enabled", ag.enabled)
        ag.history = d.get("history", [])
        ag.last_raw = d.get("last_raw", "")
        ag.last_json = d.get("last_json", None)
        return ag

    def save_current_config(name: str) -> tuple[bool, str]:
        fname = os.path.join(SAVE_DIR, f"{name}.json")
        try:
            payload = {
                "grid": [
                    [serialize_agent(a) for a in row]
                    for row in st.session_state.grid
                ],
                "global_kv": st.session_state.global_kv,
                "defaults": st.session_state.defaults,
                "loop_index": st.session_state.loop_index,
                "current_row": st.session_state.current_row,
            }
            with open(fname, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            return True, fname
        except Exception as e:
            return False, str(e)

    def list_saved_configs():
        try:
            files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".json")]
            files.sort(reverse=True)
            return files
        except Exception:
            return []

    def load_config(fname: str) -> tuple[bool, str]:
        path = os.path.join(SAVE_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            # restore
            st.session_state.grid = [
                [deserialize_agent(d) for d in row]
                for row in payload.get("grid", [[]])
            ]
            st.session_state.global_kv = payload.get("global_kv", {})
            st.session_state.defaults = payload.get("defaults", st.session_state.defaults)
            st.session_state.loop_index = payload.get("loop_index", 0)
            st.session_state.current_row = payload.get("current_row", 0)
            st.session_state.loaded_sample = False
            safe_rerun()
            return True, path
        except Exception as e:
            return False, str(e)

    new_name = st.text_input("保存名", value="my_workflow")
    if st.button("設定を保存（現在の Grid を保存）"):
        ok, info = save_current_config(new_name)
        if ok:
            st.success(f"保存しました: {info}")
        else:
            st.error(f"保存に失敗しました: {info}")

    saved = list_saved_configs()
    if saved:
        sel = st.selectbox("保存済み設定を読み込む", options=["(選択)"] + saved)
        if sel and sel != "(選択)":
            if st.button("読み込む"):
                ok, info = load_config(sel)
                if ok:
                    st.success(f"読み込み成功: {info}")
                else:
                    st.error(f"読み込み失敗: {info}")
    else:
        st.caption("保存済み設定が見つかりません。保存してください。")

    st.markdown("---")
    st.caption("サイドバーで変更した値は即時反映されます。")


# --- 追加: エージェント操作コントローラ描画ヘルパ ---
def render_agent_controller(agent: Agent, r: int, c: int):
    """
    各セル下部に操作群をまとめて表示する。
    呼び出し側は render_main 内で既に render_agent_controller(agent, r, c) を使っています。
    """
    # render controls directly without injecting an extra wrapper DIV
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        # Enable トグル（チェック式）
        toggled = st.checkbox("Enable", value=agent.enabled, key=f"en_{agent.id}")
        if toggled != agent.enabled:
            agent.enabled = toggled
            st.session_state.grid[r][c] = agent
    with col2:
        if st.button("▶ 実行", key=f"run_{agent.id}"):
            try:
                updated = run_agent(agent, r)
                st.session_state.grid[r][c] = updated
                st.success("実行完了")
            except Exception as e:
                st.error(f"エラー: {e}")
    with col3:
        if st.button("⟳ 結果クリア", key=f"clear_{agent.id}"):
            agent.last_raw = ""
            agent.last_json = None
            st.session_state.grid[r][c] = agent
    # no extra closing DIV

def render_main():
    st.title("マルチエージェント・グリッド（紺UI・罫線グリッド）")

    # ターゲット
    st.subheader("ターゲット文書（global_kv.target_doc）")
    td = st.session_state.global_kv.get("target_doc", "")
    td_new = st.text_area("target_doc", value=td, height=150)
    c1, c2 = st.columns([1,3])
    with c1:
        if st.button("target_doc を保存/反映"):
            st.session_state.global_kv["target_doc"] = td_new
            st.success("target_doc を更新しました")
        # PR 作成用ボタン: target_doc の内容をリポジトリにアップして PR を作る
    if st.button("target_doc で PR を作成", key="create_pr_from_target_doc"):
            # use centralized loader which reads secrets.toml (preferred) -> env -> global_kv
            cfg = load_github_config()
            token = cfg.get('token')
            owner = cfg.get('owner')
            repo = cfg.get('repo')
            base_branch = cfg.get('base_branch') or 'main'
            file_path = cfg.get('file_path') or 'docs/runbooks/job-acl.md'
            if not token or not owner or not repo:
                st.error("GITHUB_TOKEN / GITHUB_OWNER / GITHUB_REPO が未設定です。環境変数か global_kv に設定してください。")
            else:
                client = GitApiClient(token=token, owner=owner, repo=repo)
                branch_name = f"streamlit/target_doc_{uuid.uuid4().hex[:8]}"
                try:
                    client.create_branch(branch_name, str(base_branch))
                    content = td_new or st.session_state.global_kv.get("target_doc", "")
                    commit_msg = f"Update target_doc from Streamlit {branch_name}"
                    client.put_file(path=str(file_path), content=content, branch=branch_name, message=commit_msg)
                    pr = client.create_pull_request(
                        title="Update target_doc via Streamlit",
                        head=branch_name,
                        base=str(base_branch),
                        body="Automated update from Streamlit app",
                    )
                    # GitApiClient.create_pull_request may return a URL string or a dict
                    pr_url = None
                    if isinstance(pr, dict):
                        pr_url = pr.get("html_url")
                    elif isinstance(pr, str):
                        pr_url = pr
                    if pr_url:
                        st.success(f"Pull Request created: {pr_url}")
                    else:
                        st.success("Pull Request 作成リクエストは送信されました（詳細はログを確認してください）。")
                except Exception as e:
                    st.error(f"PR 作成に失敗しました: {e}")
    with c2:
        st.caption("Editor が kv_patch.target_doc を返すと、ここが自動更新されます。")
    # --- リポジトリ読み書きボタン ---
    repo_path = st.session_state.global_kv.get('repo_info', {}).get('path')
    rp_display = repo_path or '(repo_info.path が未設定)'
    st.caption(f"リポジトリファイル: {rp_display}")
    r1, r2 = st.columns([1,1])
    with r1:
        if st.button("リポジトリから読み込む"):
            if not repo_path:
                st.error('st.session_state.global_kv["repo_info"]["path"] が未設定です')
            else:
                # show source info
                cfg = load_github_config()
                owner = cfg.get("owner")
                repo = cfg.get("repo")
                base_branch = cfg.get("base_branch") or 'main'
                src_display = f"{owner}/{repo}@{base_branch}" if owner and repo else "(リモート未設定)"
                st.info(f"取得元: {src_display}")
                ok, data, used_path, src, target_url, git_msg = read_repo_file(repo_path)
                # persist last fetch info so it doesn't disappear
                st.session_state['last_repo_fetch'] = {
                    'ts': datetime.now().isoformat(),
                    'ok': bool(ok),
                    'used_path': used_path,
                    'source': src or src_display,
                    'target_url': target_url,
                    'git_message': git_msg,
                    'message': data if not ok else '読み込み成功',
                }
                if ok:
                    st.session_state.global_kv['target_doc'] = data
                    # indicate whether data came from local path or remote
                    p = repo_fs_path(repo_path)
                    if os.path.exists(p):
                        st.success(f"読み込み成功 (ローカル): {p}")
                    else:
                        st.success(
                            f"読み込み成功 (リモート {st.session_state['last_repo_fetch']['source']}): {used_path or repo_path}"
                        )
                    safe_rerun()
                else:
                    # include branch/source hint in error
                    st.error(f"読み込み失敗 ({st.session_state['last_repo_fetch']['source']}): {data}")
    with r2:
        if st.button("リポジトリに書き込む"):
            if not repo_path:
                st.error('st.session_state.global_kv["repo_info"]["path"] が未設定です')
            else:
                content = td_new or st.session_state.global_kv.get('target_doc', '')
                ok, info = write_repo_file(repo_path, content)
                if ok:
                    st.success(f"書き込み成功: {info}")
                else:
                    st.error(f"書き込み失敗: {info}")

    # persistent display of last repo fetch info
    last = st.session_state.get('last_repo_fetch')
    if last:
        with st.expander("前回のリポジトリ取得情報", expanded=True):
            st.write(f"時刻: {last.get('ts')}")
            st.write(f"成功: {last.get('ok')}")
            st.write(f"取得元: {last.get('source')}")
            st.write(f"使用パス: {last.get('used_path')}")
            st.write(f"対象URL: {last.get('target_url')}")
            st.write(f"git message: {last.get('git_message')}")
            st.write(f"備考: {last.get('message')}")

    # --- Doc KV 編集エリア ---
    with st.expander("情報源とルールを編集 (sources / style / lint)", expanded=False):
        cols = st.columns([1,1])
        with cols[0]:
            st.subheader("sources_slack (JSON array or plain lines)")
            ss = "\n".join(st.session_state.global_kv.get("sources_slack", []))
            ss_new = st.text_area("Slack ソース (1行ごと)", value=ss, height=120)
            if st.button("保存: Slack sources を上書き"):
                st.session_state.global_kv["sources_slack"] = [line for line in ss_new.splitlines() if line.strip()]
                st.success("sources_slack を更新しました")
        with cols[1]:
            st.subheader("sources_confluence (JSON) / sources_pdfs")
            sc = json_pretty(st.session_state.global_kv.get("sources_confluence", []))
            sc_new = st.text_area("Confluence JSON (配列)", value=sc, height=120)
            ok, err = False, None
            if st.button("保存: Confluence JSON を解析して保存"):
                ok, err = set_kv_json("sources_confluence", sc_new)
                if ok:
                    st.success("sources_confluence を更新しました")
                else:
                    st.error(f"JSON パース失敗: {err}")
            sp = "\n".join(st.session_state.global_kv.get("sources_pdfs", []))
            sp_new = st.text_area("PDF ソース (1行ごと)", value=sp, height=80)
            if st.button("保存: PDF sources を上書き"):
                st.session_state.global_kv["sources_pdfs"] = [line for line in sp_new.splitlines() if line.strip()]
                st.success("sources_pdfs を更新しました")

        st.markdown("---")
        st.subheader("Style / Lint ルール")
        sr = json_pretty(st.session_state.global_kv.get("style_rules", []))
        lr = json_pretty(st.session_state.global_kv.get("lint_rules", []))
        sr_new = st.text_area("Style rules (JSON array)", value=sr, height=80)
        lr_new = st.text_area("Lint rules (JSON array)", value=lr, height=80)
        if st.button("保存: rules をJSONで保存"):
            ok1, err1 = set_kv_json("style_rules", sr_new)
            ok2, err2 = set_kv_json("lint_rules", lr_new)
            if ok1 and ok2:
                st.success("style_rules と lint_rules を更新しました")
            else:
                st.error(f"保存失敗: {err1 or err2}")

    # 実行系（既存のまま）
    st.markdown("---")
    a,b,c,d,e = st.columns([1.2,1.2,1.5,2,3])
    with a:
        if st.button("Step 実行（現在行）"):
            try:
                run_one_step(); st.success("Step完了")
            except Exception as er:
                st.error(f"エラー: {er}")
    with b:
        if st.button("最後まで実行（1ループ）"):
            try:
                run_to_end(); st.success("ループ完了")
            except Exception as er:
                st.error(f"エラー: {er}")
    with c:
        loops_n = st.number_input("ループ回数", 1, 100, 1)
        if st.button("Nループ実行"):
            try:
                run_n_loops(int(loops_n)); st.success(f"{int(loops_n)} ループ完了")
            except Exception as er:
                st.error(f"エラー: {er}")
    with d:
        if st.button("サンプルをロード（3段）"):
            load_sample_3step()
            st.success("3段サンプルを配置しました")
        if st.button("Docワークフローをロード (サンプル)"):
            load_sample_doc_workflow()
            st.success("Docワークフロー（サンプル）を配置しました")
    with e:
        st.write(f"現在の行: {st.session_state.current_row+1} / {len(st.session_state.grid)} | ループ: {st.session_state.loop_index}")

    st.markdown("---")

    base_cols = int(st.session_state.defaults["columns_per_row"])

    # 行（Step）ごとに描画
    for r, row in enumerate(st.session_state.grid):
        # 行ヘッダ（見出し＋行/並列追加）
        # Use native Streamlit header to avoid injecting an extra small wrapper div
        st.subheader(f"Step {r+1}")
        h1, h2 = st.columns([6,2])
        with h2:
            tcol1, tcol2 = st.columns(2)
            with tcol1:
                if st.button("＋ 並列を右に", key=f"addcol_btn_{r}"):
                    st.session_state.grid[r].append(new_agent(f"Agent {len(st.session_state.grid[r])+1}"))
                    safe_rerun()
            with tcol2:
                if st.button("＋ 行を下に", key=f"addrow_btn_{r}"):
                    st.session_state.grid.insert(r+1, [])
                    safe_rerun()

        # 横のセル数（＋追加分で必ず+1）
        grid_cols = max(base_cols, len(row) + 1)
        cols = st.columns(grid_cols, gap="small")

        for c in range(grid_cols):
            # use a dedicated container attached to this column so all outputs stay in the same column
            cell_container = cols[c].container()
            with cell_container:
                # セル外枠（罫線と背景）開始
                has_agent = c < len(row)
                cell_cls = "agent-cell" + (" has-agent" if has_agent else " empty")
                if has_agent:
                    agent = row[c]
                    # まず結果テキストを整形
                    result_content = ""
                    if agent.last_json and isinstance(agent.last_json, dict):
                        message = agent.last_json.get("message", {})
                        if message:
                            result_content = json_pretty(message)
                            if len(result_content) > 400:
                                result_content = result_content[:400] + "..."
                        else:
                            result_content = "(no message)"
                    elif agent.last_raw:
                        result_content = agent.last_raw[:400] + "..." if len(agent.last_raw) > 400 else agent.last_raw
                    else:
                        result_content = "(no result)"

                    import html
                    escaped_content = html.escape(result_content)

                    # result を Streamlit の code/text ブロックで表示（これにより必ずこのセル内に描画される）
                    # build result-preview block; add 'selected' class when this agent is targeted
                    is_selected = (st.session_state.get("ui_target_agent_id") == agent.id)
                    sel_cls = " selected" if is_selected else ""
                    # Combine result and a narrow spacer so result preview gets most width
                    left_col, _spacer = cell_container.columns([4,1], gap="small")
                    # left: result preview
                    with left_col:
                        if agent.last_json and isinstance(agent.last_json, dict):
                            msg = agent.last_json.get("message")
                            if msg:
                                import html as _html
                                left_col.markdown(f'<div class="result-preview{sel_cls}"><pre>{_html.escape(json_pretty(msg))}</pre></div>', unsafe_allow_html=True)
                            else:
                                left_col.markdown(f'<div class="result-preview no-result{sel_cls}">(no message)</div>', unsafe_allow_html=True)
                        elif agent.last_raw:
                            import html as _html
                            left_col.markdown(f'<div class="result-preview{sel_cls}"><pre>{_html.escape(agent.last_raw)}</pre></div>', unsafe_allow_html=True)
                        else:
                            left_col.markdown(f'<div class="result-preview no-result{sel_cls}">(no result)</div>', unsafe_allow_html=True)
                    # controllers: render a single horizontal row under the result preview inside left_col
                    with left_col:
                        ctrl_cols = left_col.columns([1,1,1,1,1,1], gap="small")
                        with ctrl_cols[0]:
                            if st.button("+", key=f"plus_{agent.id}"):
                                st.session_state.grid[r].insert(
                                    c+1, new_agent(f"Agent {len(st.session_state.grid[r])+1}"))
                                safe_rerun()
                        with ctrl_cols[1]:
                            if st.button("-", key=f"minus_{agent.id}"):
                                try:
                                    st.session_state.grid[r].pop(c)
                                except Exception:
                                    pass
                                safe_rerun()
                        with ctrl_cols[2]:
                            if st.button("🔍", key=f"tile_detail_{agent.id}"):
                                go_detail(agent.id)
                        with ctrl_cols[3]:
                            toggled = st.checkbox(
                                "\u00A0",
                                value=agent.enabled,
                                key=f"en_{agent.id}",
                                label_visibility="collapsed",
                            )
                        with ctrl_cols[4]:
                            if st.button("▶", key=f"run_{agent.id}"):
                                try:
                                    updated = run_agent(agent, r)
                                    st.session_state.grid[r][c] = updated
                                    st.success("実行完了")
                                except Exception as e:
                                    st.error(f"エラー: {e}")
                        with ctrl_cols[5]:
                            if st.button("■", key=f"clear_{agent.id}"):
                                agent.last_raw = ""
                                agent.last_json = None
                                st.session_state.grid[r][c] = agent

                        # small label for enable under the buttons area
                        left_col.markdown(
                            '<div style="font-size:0.78rem;margin-top:4px;">Enable</div>',
                            unsafe_allow_html=True,
                        )
                        if toggled != agent.enabled:
                            agent.enabled = toggled
                            st.session_state.grid[r][c] = agent

                    # (removed outer tile-area wrapper)
                else:
                    # 空セル（色を変え、dashed枠）
                    # render the add button directly without an extra wrapper div
                    if cell_container.button("＋ 追加\nこの位置にエージェントを追加", key=f"addcol_{r}_{c}", use_container_width=True):
                        # 指定位置 c に挿入する（末尾 append ではない）
                        st.session_state.grid[r].insert(c, new_agent(f"Agent {len(row)+1}"))
                        safe_rerun()

                # previously closed an injected agent-cell div; removed to avoid extra empty element

    # 行間の空白帯（avoid inserting an extra div; use natural spacing)
    st.write('')
    st.markdown("---")

    # --- 追加: 現在行のエージェントごとに別ウィンドウ表示（追加時に NO RESULT を見やすくするため）
    st.subheader("エージェントウィンドウ（現在行）")
    cur_row_idx = st.session_state.current_row
    if cur_row_idx < len(st.session_state.grid):
        cur_row = st.session_state.grid[cur_row_idx]
        if cur_row:
            aw_cols = st.columns(len(cur_row))
            for ai, ag in enumerate(cur_row):
                with aw_cols[ai]:
                    # one-line window: left=result, right=compact controls
                    aw_left, aw_right = st.columns([4,1], gap="small")
                    with aw_left:
                        if ag.last_json and isinstance(ag.last_json, dict):
                            msg = ag.last_json.get("message")
                            if msg:
                                aw_left.markdown(f'<div class="result-preview">{html.escape(json_pretty(msg))}</div>', unsafe_allow_html=True)
                            else:
                                aw_left.markdown('<div class="result-preview no-result">(no message)</div>', unsafe_allow_html=True)
                        elif ag.last_raw:
                            aw_left.markdown(f'<div class="result-preview">{html.escape(ag.last_raw)}</div>', unsafe_allow_html=True)
                        else:
                            aw_left.markdown('<div class="result-preview no-result">(no result)</div>', unsafe_allow_html=True)
                    with aw_right:
                        # compact buttons/icons
                        toggled = aw_right.checkbox("\u00A0", value=ag.enabled, key=f"aw_en_{ag.id}", label_visibility="collapsed")
                        aw_right.markdown('<div style="font-size:0.85rem;margin-left:6px;display:inline-block">Enable</div>', unsafe_allow_html=True)
                        if toggled != ag.enabled:
                            ag.enabled = toggled
                            st.session_state.grid[cur_row_idx][ai] = ag
                        if aw_right.button("▶", key=f"aw_run_{ag.id}"):
                            try:
                                updated = run_agent(ag, cur_row_idx)
                                st.session_state.grid[cur_row_idx][ai] = updated
                                aw_right.success("実行完了")
                            except Exception as e:
                                aw_right.error(f"エラー: {e}")
                        if aw_right.button("⟳", key=f"aw_clear_{ag.id}"):
                            ag.last_raw = ""
                            ag.last_json = None
                            st.session_state.grid[cur_row_idx][ai] = ag
        else:
            st.info("この行にはまだエージェントがいません。タイルの +追加 で追加してください。")
    else:
        st.info("現在行にエージェントがありません。")
    st.subheader("現在ループの進捗（unified）")
    st.write(f"完了済みステップ数: {len(st.session_state.current_loop_steps)} / {len(st.session_state.grid)}")
    st.code(json_pretty(st.session_state.current_loop_steps), language="json")

    st.subheader("過去ループ履歴（最新から最大N件表示）")
    depth = st.slider("表示件数", 0, 20, 5)
    view = st.session_state.history_loops[-depth:] if depth > 0 else []
    st.code(json_pretty(view), language="json")

# -------- Detail View --------
def render_detail():
    st.title("エージェント詳細")
    if not st.session_state.ui_target_agent_id:
        st.warning("エージェントが選択されていません。メインに戻ってタイルをクリックしてください。")
        if st.button("メインに戻る"):
            go_main()
        return

    pos = find_agent_pos(st.session_state.ui_target_agent_id)
    if not pos:
        st.error("選択されたエージェントが見つかりません。")
        if st.button("メインに戻る"):
            go_main()
        return

    r, c, agent = pos

    tb1, tb2, tb3 = st.columns([1,2,5])
    with tb1:
        if st.button("← メインへ"):
            go_main()
    with tb2:
        st.write(f"Step {r+1} / 列 {c+1}")
    with tb3:
        pass

    agent.enabled = st.checkbox("有効", value=agent.enabled, key=f"detail_enabled_{agent.id}")
    agent.name = st.text_input("名前", value=agent.name, key=f"detail_name_{agent.id}")

    with st.expander("モデル設定", expanded=False):
        agent.model = st.text_input("model", value=agent.model, key=f"detail_model_{agent.id}")
        agent.temperature = st.slider("temperature", 0.0, 2.0, float(agent.temperature), 0.1, key=f"detail_temp_{agent.id}")
        agent.max_tokens = int(st.number_input("max_tokens", 128, 8192, int(agent.max_tokens), 64, key=f"detail_maxtok_{agent.id}"))
        seed_used = st.checkbox("このエージェント専用seedを使う", value=(agent.seed is not None), key=f"detail_seedflag_{agent.id}")
        if seed_used:
            agent.seed = int(st.number_input("seed", 0, 2**31-1, int(agent.seed or 0), 1, key=f"detail_seedval_{agent.id}"))
        else:
            agent.seed = None

    # テンプレボタン
    t_worker = "対象本文は Context.global_kv.target_doc。最小限の改善提案を最大3件: message.type='proposal', message.changes=[{desc,before,after}]。"
    t_judge = "直前行のWorkersは Context.resp[-1]。良い変更だけ選び message.type='selection', message.accepted_changes=[{from_col,index}]。"
    t_editor = "Judgeは Context.resp[-1][0].message。Workersは Context.resp[-2]。accepted_changes を before→after で適用し kv_patch.target_doc を更新。message.type='commit'。"
    bt1, bt2, bt3 = st.columns(3)
    with bt1:
        if st.button("テンプレ: Worker", key=f"detail_tmplW_{agent.id}"):
            agent.user_prompt = t_worker
    with bt2:
        if st.button("テンプレ: Judge", key=f"detail_tmplJ_{agent.id}"):
            agent.user_prompt = t_judge
    with bt3:
        if st.button("テンプレ: Editor", key=f"detail_tmplE_{agent.id}"):
            agent.user_prompt = t_editor

    agent.user_prompt = st.text_area("指示プロンプト（Context とスキーマ制約は自動付与）", value=agent.user_prompt, height=220, key=f"detail_user_{agent.id}")

    # --- 追加: 詳細画面で LLM に送られる『生データ』を確認できる折りたたみ表示 ---
    # construct the same user_prompt / messages as run_agent would do so the user can inspect
    try:
        ctx_preview = build_context_for_row(r)
    except Exception:
        ctx_preview = {"error": "context build failed"}

    constructed_user_prompt = f"""{SCHEMA_ENFORCER}

指示:
{agent.user_prompt}

[Context]
{json_pretty(ctx_preview)}
"""

    # history slice like run_agent uses
    hist_preview = agent.history[-40:] if len(agent.history) > 40 else agent.history
    messages_preview = hist_preview + [{"role": "user", "content": constructed_user_prompt}]

    with st.expander("送信される生データ（Raw LLM prompt / messages）", expanded=False):
        st.caption(
            "下は実際にLLMへ送られるユーザープロンプト文字列と、"
            "履歴を含めた messages 配列のプレビューです。解析用に表示しています。"
        )
        st.subheader("Raw prompt (string)")
        st.code(constructed_user_prompt or "(empty)")
        st.subheader("Messages (history + current)")
        st.code(
            json_pretty(messages_preview),
            language="json",
        )

    # レスポンスナビ
    st.markdown("---")
    st.subheader("レスポンスナビ（クリックでプロンプトへ挿入）")
    st.caption("例: response[0][-1][0].message = 現ループ直前行の0列のJSON。")
    row_idx = st.session_state.current_row

    if row_idx == 0:
        st.caption("まだ前行がありません。Stepを実行するとここに並びます。")
    else:
        for rel in range(-1, -row_idx-1, -1):
            abs_row = row_idx + rel
            cols = st.columns(len(st.session_state.grid[abs_row]) + 1)
            with cols[0]:
                st.markdown(f"行 {abs_row+1}（相対 {rel}）")
            for col in range(len(st.session_state.grid[abs_row])):
                with cols[col+1]:
                    path = f"response[0][{rel}][{col}].message"
                    if st.button(f"col {col} を挿入", key=f"detail_path_{abs_row}_{col}"):
                        agent.user_prompt = (agent.user_prompt.rstrip() + "\n" + path).strip()
                        st.success(f"挿入: {path}")
                    cell_agent = st.session_state.grid[abs_row][col]
                    st.caption(cell_agent.name)
                    preview_msg = None
                    if cell_agent.last_json:
                        preview_msg = cell_agent.last_json.get("message")
                    st.code(json_pretty(preview_msg) if preview_msg else "(なし)", language="json")

    with st.expander("直前ループの最終行（あれば）", expanded=False):
        if st.session_state.history_loops:
            last = st.session_state.history_loops[-1]
            snap = last.get("grid_snapshot", [])
            if snap:
                final_row_idx = len(snap) - 1
                cols = st.columns(len(snap[final_row_idx]) + 1)
                with cols[0]:
                    st.markdown("前ループ 最終行（相対 -1）")
                for col in range(len(snap[final_row_idx])):
                    with cols[col+1]:
                        path = f"response[-1][-1][{col}].message"
                        if st.button(f"col {col} を挿入(前L)", key=f"detail_prev_{col}"):
                            agent.user_prompt = (agent.user_prompt.rstrip() + "\n" + path).strip()
                            st.success(f"挿入: {path}")
                        step_map = last["steps"][final_row_idx]
                        meta = snap[final_row_idx][col]
                        msg = (step_map.get(meta["id"]) or {}).get("message")
                        st.code(json_pretty(msg) if msg else "(なし)", language="json")
        else:
            st.caption("過去ループはまだありません。")

    # 実行/クリア
    st.markdown("---")
    ac1, ac2 = st.columns([1,1])
    with ac1:
        if st.button("このエージェントを実行", key=f"detail_run_{agent.id}"):
            try:
                updated = run_agent(agent, r)
                st.session_state.grid[r][c] = updated
                st.success("実行完了")
            except Exception as e:
                st.error(f"エラー: {e}")
    with ac2:
        if st.button("結果クリア", key=f"detail_clear_{agent.id}"):
            agent.last_raw = ""
            agent.last_json = None
            st.session_state.grid[r][c] = agent

    st.caption("JSON結果（unified schema）")
    st.code(
        json_pretty(agent.last_json) if agent.last_json is not None else "(なし)",
        language="json",
    )
    with st.expander("生テキスト（解析前）", expanded=False):
        st.code(agent.last_raw or "(なし)")

    st.session_state.grid[r][c] = agent

# Route（単一）
if st.session_state.view == "main":
    render_main()
else:
    render_detail()

# ===== 末尾クリーンアップ完了 =====
