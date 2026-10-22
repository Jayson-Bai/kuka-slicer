"""CUT completion barriers; the wire bridge applies extrusion scale once."""
import math


def cut_holds(legacy=3.0, lift=None, retract=None):
    values = tuple(float(legacy if v is None else v) for v in (lift, retract))
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("CUT hold times must be finite and non-negative")
    return values


def cut_sync_payload(source_payload=1, expected_syncs=2):
    """Declare the CUT completion-barrier contract on the CUT wire event."""
    count = int(expected_syncs)
    if count < 1:
        raise ValueError("expected_syncs must be positive")
    source = str(source_payload).strip() or "1"
    contract = f"sync_v1:{count}"
    declared = [token for token in source.split() if token.startswith("sync_v1:")]
    if declared:
        if declared != [contract]:
            raise ValueError("CUT payload declares a conflicting sync contract")
        return source
    return f"{source} {contract}"


def sync_payload(tool, target, wait):
    # A short UI window never weakens completion checking. Timeout fails closed.
    timeout_ms = min(600000, max(30000, math.ceil((float(wait) + 10.0) * 1000)))
    return f"{int(tool)} {float(target):.9f} {timeout_ms}"
