"""Aircraft class from an ICAO type designator: piston, turboprop, jet, helicopter or unknown (ICAO Doc 8643
descriptions, table in data/ref/icao_aircraft_types.json)."""
import json
from functools import cache
from pathlib import Path

TABLE = Path("data/ref/icao_aircraft_types.json")


@cache
def table():
    return json.loads(TABLE.read_text()) if TABLE.exists() else {}


def aircraft_class(type_code, tab=None):
    desc = ((table() if tab is None else tab).get(type_code) or {}).get("desc", "")
    if not desc:
        return "unknown"
    if desc[0] == "H":
        return "helicopter"
    return {"P": "piston", "T": "turboprop", "J": "jet"}.get(desc[-1], "unknown")
