"""Frontend static file mounting."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.resources import resource_path

STATIC_DIR = resource_path("frontend")


def mount_static_frontend(app: FastAPI) -> None:
    static_dir = STATIC_DIR
    if not static_dir.exists():
        return

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon_ico():
        return FileResponse(str(static_dir / "favicon.svg"), media_type="image/svg+xml")

    @app.head("/favicon.ico", include_in_schema=False)
    async def favicon_ico_head():
        favicon_path = static_dir / "favicon.svg"
        return Response(
            status_code=200,
            headers={
                "content-length": str(favicon_path.stat().st_size),
                "content-type": "image/svg+xml",
            },
        )

    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")
