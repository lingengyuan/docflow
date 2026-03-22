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


@dataclass
class WatchDir:
    path: Path
    recursive: bool = False
    extensions: list[str] = field(default_factory=list)  # empty = use pipeline default


class FileEventHandler(FileSystemEventHandler):
    def __init__(self, pipeline: IngestPipeline, extensions: list[str]):
        self.pipeline = pipeline
        self.extensions = [e.lower() for e in extensions]
        self._processing: set[str] = set()

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
        if str(path) in self._processing:
            return
        self._processing.add(str(path))
        try:
            logger.info(f"[watcher] detected: {path.name}")
            result = self.pipeline.ingest(path)
            logger.info(f"[watcher] {result}")
        finally:
            self._processing.discard(str(path))


class FolderWatcher:
    def __init__(self, pipeline: IngestPipeline, watch_dirs: list[WatchDir]):
        self.pipeline = pipeline
        self.watch_dirs = watch_dirs
        for wd in self.watch_dirs:
            wd.path.mkdir(parents=True, exist_ok=True)
        self._observer = Observer()

    def start(self):
        default_exts = self.pipeline.registry.supported_extensions
        for wd in self.watch_dirs:
            exts = wd.extensions if wd.extensions else default_exts
            handler = FileEventHandler(self.pipeline, exts)
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
                files.extend(wd.path.glob(pattern))
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
