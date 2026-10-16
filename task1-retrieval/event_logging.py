"""UniAgent event traces, including unsuccessful local model calls."""
from datetime import datetime, timezone
import gzip
import json


class EventTrace:
    def __init__(self, path, model):
        self.stream = gzip.open(path, "wt", encoding="utf-8")
        self.model, self.sequence = model, 0

    def emit(self, case_id, event_type, parent=None, inputs=None, output=None,
             status="ok", error=None):
        self.sequence += 1
        event_id = f"evt-{self.sequence:07d}"
        event = {"case_id": str(case_id), "event_id": event_id, "parent_event_id": parent,
                 "timestamp": datetime.now(timezone.utc).isoformat(), "event_type": event_type,
                 "model": self.model, "tool": None, "input": inputs, "output": output,
                 "status": status, "error": error}
        self.stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()
        return event_id

    def close(self):
        self.stream.close()
