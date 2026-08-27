"""
Tests for GET /api/health.
"""

from __future__ import annotations


class TestHealth:
    async def test_returns_200(self, client):
        resp = await client.get("/api/health")
        assert resp.status_code == 200

    async def test_content_type_is_json(self, client):
        resp = await client.get("/api/health")
        assert "application/json" in resp.headers["content-type"]

    async def test_status_is_ok(self, client):
        body = (await client.get("/api/health")).json()
        assert body["status"] == "ok"

    async def test_backend_is_missionmind(self, client):
        body = (await client.get("/api/health")).json()
        assert body["backend"] == "missionmind"

    async def test_version_present(self, client):
        body = (await client.get("/api/health")).json()
        assert "version" in body
        assert body["version"]  # non-empty string
