"""Context manager for timing code blocks."""
import time


class Timer:
    """
    Context manager that times a code block.

    Usage:
        with Timer("Training XGBoost") as t:
            model.fit(X, y)
        print(f"Elapsed: {t.elapsed:.2f}s")
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        if self.label:
            print(f"[Timer] {self.label} started...")
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.label:
            print(f"[Timer] {self.label} finished in {self.elapsed:.2f}s")
