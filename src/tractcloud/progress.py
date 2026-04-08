"""Progress reporting for TractCloud pipeline.

Provides reporters that emit JSON lines on stdout (for QProcess parsing)
or call Python callbacks (for direct-import use in Slicer).
"""

import json
import sys
import time


class ProgressReporter:
    """Writes JSON-line progress messages to stdout.

    All non-progress output (logging, warnings) should go to stderr.
    stdout is reserved exclusively for the JSON progress protocol.
    """

    def __init__(self, stream=None):
        self._stream = stream or sys.stdout
        self._step_start_times = {}

    def status(self, message, step=None, total_steps=None):
        msg = {"type": "status", "message": message}
        if step is not None:
            msg["step"] = step
            self._step_start_times[step] = time.time()
        if total_steps is not None:
            msg["total_steps"] = total_steps
        self._emit(msg)

    def progress(self, fraction, step=None):
        msg = {"type": "progress", "fraction": round(fraction, 4)}
        if step is not None:
            msg["step"] = step
            # Estimate remaining time
            start = self._step_start_times.get(step)
            if start and fraction > 0.01:
                elapsed = time.time() - start
                estimated_total = elapsed / fraction
                msg["elapsed"] = round(elapsed, 1)
                msg["estimated_remaining"] = round(
                    estimated_total - elapsed, 1)
        self._emit(msg)

    def result(self, tracts_created, output_dir, total_time=None):
        msg = {
            "type": "result",
            "tracts_created": tracts_created,
            "output_dir": output_dir,
        }
        if total_time is not None:
            msg["total_time"] = round(total_time, 1)
        self._emit(msg)

    def _emit(self, msg):
        self._stream.write(json.dumps(msg) + "\n")
        self._stream.flush()


class CallbackReporter:
    """Reports progress via Python callbacks (for direct-import in Slicer)."""

    def __init__(self, status_callback=None, progress_callback=None):
        self._status_cb = status_callback
        self._progress_cb = progress_callback

    def status(self, message, step=None, total_steps=None):
        if self._status_cb:
            self._status_cb(message)

    def progress(self, fraction, step=None):
        if self._progress_cb:
            self._progress_cb(fraction)

    def result(self, tracts_created, output_dir, total_time=None):
        if self._status_cb:
            self._status_cb(
                f"Done! Created {tracts_created} tract bundles"
                + (f" in {total_time:.1f}s" if total_time else ""))


class NullReporter:
    """Silent reporter for batch/quiet mode."""

    def status(self, message, step=None, total_steps=None):
        pass

    def progress(self, fraction, step=None):
        pass

    def result(self, tracts_created, output_dir, total_time=None):
        pass
