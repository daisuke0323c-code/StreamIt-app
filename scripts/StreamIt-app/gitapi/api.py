# -*- coding: utf-8 -*-
"""
Minimal GitHub REST API helper used by Streamlit app to create branches, update files
and create pull requests. This module provides a small, dependency-light client that
uses the GitHub REST API (requests).
"""

import base64
from typing import Any, Dict, Optional

import requests

API_BASE = "https://api.github.com"


class GitApiClient:
    """Small GitHub REST client.

    Usage:
        client = GitApiClient(token, owner, repo)
        client.create_branch("my-branch", from_branch="main")
        client.put_file(path, content, branch="my-branch", message="Add file")
        client.create_pull_request(title, body, head="my-branch", base="main")
    """

    def __init__(self, token: str, owner: str, repo: str):
        self.token = token
        self.owner = owner
        self.repo = repo
        self.headers = {
            "Accept": "application/vnd.github+json",
        }
        # Only include Authorization header when a token is provided.
        if token:
            self.headers["Authorization"] = f"token {token}"

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = API_BASE + path
        r = requests.request(
            method,
            url,
            headers=self.headers,
            params=params,
            json=json,
            timeout=30,
        )
        if r.status_code >= 400:
            raise RuntimeError(
                f"{method} {url} -> {r.status_code}: {r.text}"
            )
        # Some endpoints return no content; handle gracefully
        if r.text:
            return r.json()
        return None

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request("GET", path, params=params)

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        return self._request("POST", path, json=payload)

    def _put(self, path: str, payload: Dict[str, Any]) -> Any:
        return self._request("PUT", path, json=payload)

    def get_branch_sha(self, branch: str) -> str:
        data = self._get(
            f"/repos/{self.owner}/{self.repo}/git/ref/heads/{branch}"
        )
        return data["object"]["sha"]

    def create_branch(self, new_branch: str, from_branch: str) -> None:
        base_sha = self.get_branch_sha(from_branch)
        payload = {"ref": f"refs/heads/{new_branch}", "sha": base_sha}
        self._post(f"/repos/{self.owner}/{self.repo}/git/refs", payload)

    def get_file_sha(self, path: str, ref: str) -> Optional[str]:
        try:
            data = self._get(
                f"/repos/{self.owner}/{self.repo}/contents/{path}",
                params={"ref": ref},
            )
            # API returns a dict describing the file when it exists
            return data.get("sha") if isinstance(data, dict) else None
        except RuntimeError as e:
            # If not found, the API returns 404; surface as None
            if "404" in str(e):
                return None
            raise

    def get_file(self, path: str, ref: Optional[str] = None) -> Optional[str]:
        """Return decoded file content for a path on the repository (or None if not found)."""
        try:
            params = {"ref": ref} if ref else None
            data = self._get(f"/repos/{self.owner}/{self.repo}/contents/{path}", params=params)
            if isinstance(data, dict):
                content = data.get("content")
                encoding = data.get("encoding", "base64")
                if content and encoding == "base64":
                    return base64.b64decode(content.encode("utf-8")).decode("utf-8")
            return None
        except RuntimeError as e:
            if "404" in str(e):
                return None
            raise

    def put_file(
        self,
        path: str,
        content: str,
        branch: str,
        message: str,
        sha: Optional[str] = None,
    ) -> Any:
        b64 = base64.b64encode(content.encode("utf-8")).decode()
        # If sha wasn't provided, attempt to read the file's sha on the target branch.
        # When updating an existing file GitHub requires the file's blob sha. If the
        # file does not exist on the branch, sha should be omitted (create case).
        if sha is None:
            try:
                existing = self.get_file_sha(path, ref=branch)
            except Exception:
                existing = None
            if existing:
                sha = existing

        payload: Dict[str, Any] = {
            "message": message,
            "content": b64,
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha
        return self._put(f"/repos/{self.owner}/{self.repo}/contents/{path}", payload)

    def create_pull_request(
        self, title: str, body: str, head: str, base: str
    ) -> Optional[str]:
        payload = {"title": title, "body": body, "head": head, "base": base}
        data = self._post(f"/repos/{self.owner}/{self.repo}/pulls", payload)
        return data.get("html_url") if isinstance(data, dict) else None

