import json
from pathlib import Path
from datetime import datetime, timezone
import uuid
import platform
import sklearn

def log_experiment(
    log_path: Path,
    record: dict,
):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    base = {
        "run_id": record.get("run_id", uuid.uuid4().hex[:10]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
        },
    }
    base.update(record)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(base, ensure_ascii=False) + "\n")
