"""
Simple admin-only authentication.

For testing purposes, uses a hardcoded admin token.
No user registration — single admin with full permissions.
"""

import os
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "mragagent-admin-token-2026")
security = HTTPBearer(auto_error=False)


def verify_admin(credentials: HTTPAuthorizationCredentials | None = Security(security)) -> bool:
    """Verify the request has a valid admin token."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="需要认证")
    if credentials.credentials != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="无效的管理员令牌")
    return True
