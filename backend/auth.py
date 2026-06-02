"""
简单的仅限管理员的认证机制。

出于测试目的，使用硬编码的管理员令牌。
无用户注册 — 单一管理员拥有全部权限。
"""

import os
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "mragagent-admin-token-2026")
security = HTTPBearer(auto_error=False)


def verify_admin(credentials: HTTPAuthorizationCredentials | None = Security(security)) -> bool:
    """验证请求是否携带有效的管理员令牌。"""
    if credentials is None:
        raise HTTPException(status_code=401, detail="需要认证")
    if credentials.credentials != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="无效的管理员令牌")
    return True
