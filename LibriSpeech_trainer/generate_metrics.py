import time
import threading
import csv
from contextlib import contextmanager

import psutil
import torch

class TrainingProfiler:
    """
    Usage:
      prof = TrainingProfiler(report_interval=10, csv_path="metrics.csv")
      for batch in loader:
          prof.start_batch()
          with prof.time_section("forward"):
              out = model(x)
          with prof.time_section("backward"):
              loss.backward(); opt.step()
          prof.end_batch(batch_size)
          # it auto-reports every report_interval seconds, or call prof.print_report()
    """
    def __init__(self, report_interval=10.0, csv_path=None, device=None):
        self.report_interval = report_interval
        self.csv_path = csv_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # counters / timers
        self.total_samples = 0
        self.total_batches = 0
        self.total_time = 0.0  # sum of batch durations
        self.section_times = {}  # name -> cumulative time
        self._batch_start = None
        self._section_start = {}
        self._last_report = time.time()

        # last batch summary
        self.last_batch_time = 0.0
        self.lock = threading.Lock()

        # optional csv
        if self.csv_path:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp","device","cpu_pct","mem_pct","rss_mb","gpu_mem_mb","throughput_sps","avg_batch_ms","last_batch_ms"])

    def start_batch(self):
        self._batch_start = time.perf_counter()

    def end_batch(self, batch_size: int):
        now = time.perf_counter()
        if self._batch_start is None:
            raise RuntimeError("start_batch() must be called before end_batch()")
        batch_time = now - self._batch_start
        with self.lock:
            self.total_time += batch_time
            self.total_batches += 1
            self.total_samples += int(batch_size)
            self.last_batch_time = batch_time
            self._batch_start = None

        # auto report
        if time.time() - self._last_report >= self.report_interval:
            self.print_report()
            self._last_report = time.time()

    @contextmanager
    def time_section(self, name: str):
        s = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - s
            with self.lock:
                self.section_times[name] = self.section_times.get(name, 0.0) + dt

    def _get_system_metrics(self):
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        rss_mb = psutil.Process().memory_info().rss / (1024**2)
        mem_pct = vm.percent
        gpu_mem_mb = None
        # CUDA
        if torch.cuda.is_available():
            try:
                # report for current device
                dev = torch.cuda.current_device()
                gpu_mem_mb = torch.cuda.memory_allocated(dev) / (1024**2)
            except Exception:
                gpu_mem_mb = None
        return cpu, mem_pct, rss_mb, gpu_mem_mb

    def throughput(self):
        with self.lock:
            t = self.total_time if self.total_time > 0 else 1e-9
            return self.total_samples / t

    def avg_batch_ms(self):
        with self.lock:
            if self.total_batches == 0:
                return 0.0
            return (self.total_time / self.total_batches) * 1000.0

    def print_report(self):
        cpu, mem_pct, rss_mb, gpu_mem_mb = self._get_system_metrics()
        thr = self.throughput()
        avg_ms = self.avg_batch_ms()
        last_ms = self.last_batch_time * 1000.0
        s = (
            f"[METRICS] device={self.device} cpu={cpu:.1f}% mem={mem_pct:.1f}% rss={rss_mb:.1f}MB "
            f"gpu_mem={gpu_mem_mb if gpu_mem_mb is None else f'{gpu_mem_mb:.1f}MB'} "
            f"throughput={thr:.1f} samples/s avg_batch={avg_ms:.1f}ms last_batch={last_ms:.1f}ms"
        )
        print(s)

        if self.csv_path:
            row = [time.time(), self.device, cpu, mem_pct, rss_mb, gpu_mem_mb or 0.0, thr, avg_ms, last_ms]
            try:
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
            except Exception:
                pass

    def reset_epoch_stats(self):
        with self.lock:
            self.total_samples = 0
            self.total_batches = 0
            self.total_time = 0.0
            self.section_times = {}

# small helper for timing a single call
@contextmanager
def measure_latency(prof: TrainingProfiler, batch_size: int):
    prof.start_batch()
    try:
        yield
    finally:
        prof.end_batch(batch_size)

# ...existing code...
if __name__ == "__main__":
    import time
    print("Profiler self-test — ensure 'psutil' is installed: pip install psutil")
    prof = TrainingProfiler(report_interval=1.0, csv_path=None)
    print("Running 5 simulated batches...")
    for i in range(5):
        prof.start_batch()
        with prof.time_section("simulate_forward"):
            time.sleep(0.05 + 0.01 * i)
        with prof.time_section("simulate_backward"):
            time.sleep(0.02 + 0.005 * i)
        prof.end_batch(batch_size=32)
    prof.print_report()
    print("Self-test complete.")