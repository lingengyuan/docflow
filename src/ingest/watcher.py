"""
FolderWatcher — watchdog 文件夹监控，支持多目录 + 递归，自动触发 ingest。

支持在 config.yaml 中配置多个监控目录：
    paths:
      watch_dirs:
        - path: "~/Documents/DocFlow"
          recursive: false
        - path: "~/Obsidian/MyVault"
          recursive: true
          extensions: [".md"]
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from src.ingest.pipeline import IngestPipeline

logger = logging.getLogger(__name__)

# 递归扫描 Obsidian vault 时需要排除的目录名
_EXCLUDED_DIRS = {".obsidian", ".trash", ".git"}


def _is_excluded(path: Path) -> bool:
    """路径中包含排除目录则返回 True。"""
    return bool(_EXCLUDED_DIRS & set(path.parts))


@dataclass
class WatchDir:
    path: Path
    recursive: bool = False
    extensions: list[str] = field(default_factory=list)  # empty = use pipeline default


_DEBOUNCE_SECONDS = 3.0


class FileEventHandler(FileSystemEventHandler):
    def __init__(self, pipeline: IngestPipeline, extensions: list[str], ingest_queue=None):
        self.pipeline = pipeline
        self.extensions = [e.lower() for e in extensions]
        self._ingest_queue = ingest_queue
        self._last_event: dict[str, float] = {}  # path → timestamp

    def on_created(self, event: FileSystemEvent):
        self._handle(event)

    def on_modified(self, event: FileSystemEvent):
        self._handle(event)

    def _handle(self, event: FileSystemEvent):
        if event.is_directory:
            return
        path = Path(str(event.src_path))
        if path.suffix.lower() not in self.extensions:
            return
        if _is_excluded(path):
            return
        key = str(path)
        now = time.time()
        # Debounce: 距上次事件不足 N 秒则跳过
        last = self._last_event.get(key, 0)
        if now - last < _DEBOUNCE_SECONDS:
            return
        self._last_event[key] = now
        logger.info(f"[watcher] detected: {path.name}")
        if self._ingest_queue:
            self._ingest_queue.submit(path)
        else:
            result = self.pipeline.ingest(path)
            logger.info(f"[watcher] {result}")


class FolderWatcher:
    def __init__(self, pipeline: IngestPipeline, watch_dirs: list[WatchDir], ingest_queue=None):
        self.pipeline = pipeline
        self.watch_dirs = watch_dirs
        self._ingest_queue = ingest_queue
        for wd in self.watch_dirs:
            wd.path.mkdir(parents=True, exist_ok=True)
        self._observer = Observer()

    def start(self):
        default_exts = self.pipeline.registry.supported_extensions
        for wd in self.watch_dirs:
            exts = wd.extensions if wd.extensions else default_exts
            handler = FileEventHandler(self.pipeline, exts, ingest_queue=self._ingest_queue)
            self._observer.schedule(handler, str(wd.path), recursive=wd.recursive)
            logger.info(f"[watcher] watching: {wd.path} (recursive={wd.recursive}, exts={exts})")
        self._observer.start()

    def stop(self):
        self._observer.stop()
        self._observer.join()

    def scan_existing(self):
        """启动时扫描已有文件，补充入库未处理的文件。"""
        default_exts = self.pipeline.registry.supported_extensions
        for wd in self.watch_dirs:
            exts = wd.extensions if wd.extensions else default_exts
            files: list[Path] = []
            for ext in exts:
                pattern = f"**/*{ext}" if wd.recursive else f"*{ext}"
                files.extend(f for f in wd.path.glob(pattern) if not _is_excluded(f))
            logger.info(f"[watcher] scanning {len(files)} files in {wd.path}")
            for f in files:
                result = self.pipeline.ingest(f)
                logger.info(f"[watcher] {result}")

    def run_forever(self):
        """阻塞运行，Ctrl-C 退出。"""
        self.start()
        self.scan_existing()
        try:
            while True:
                time.sleep(2)
        except KeyboardInterrupt:
            self.stop()
