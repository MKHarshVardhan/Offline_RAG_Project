"""
benchmark/resource_monitor.py

Lightweight, non-rigorous background sampler for peak RSS memory and CPU%
during a block of code. Standalone helper for the evaluation scripts in this
folder — not used by the app itself.
"""

import threading
import time

import psutil


class ResourceSampler:
    """
    Samples the current process (and, if found, any 'ollama' process) on a
    background thread while a `with` block runs, tracking peak RSS and
    average CPU%. Best-effort only — sampling interval is coarse (100ms).
    """

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.peak_rss_mb = 0.0
        self.peak_ollama_rss_mb = 0.0
        self._cpu_samples = []
        self._proc = psutil.Process()

    def _ollama_processes(self):
        return [
            p for p in psutil.process_iter(["name"])
            if p.info.get("name") and "ollama" in p.info["name"].lower()
        ]

    def _run(self):
        self._proc.cpu_percent()  # prime the counter
        while not self._stop.is_set():
            try:
                rss_mb = self._proc.memory_info().rss / (1024 * 1024)
                self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
                self._cpu_samples.append(self._proc.cpu_percent())

                ollama_total = 0.0
                for p in self._ollama_processes():
                    try:
                        ollama_total += p.memory_info().rss / (1024 * 1024)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                if ollama_total:
                    self.peak_ollama_rss_mb = max(self.peak_ollama_rss_mb, ollama_total)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(self.interval)

    @property
    def avg_cpu_percent(self) -> float:
        return sum(self._cpu_samples) / len(self._cpu_samples) if self._cpu_samples else 0.0

    def __enter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        return False
