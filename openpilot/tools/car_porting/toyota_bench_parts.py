#!/usr/bin/env python3
"""Generate a Toyota hardware-in-the-loop bench sourcing report.

openpilot does not ship a bench parts list, but opendbc already contains everything needed
to build one: the FW fingerprints are literal Toyota part numbers keyed by ECU and
diagnostic address, and the car interface says which CAN messages openpilot reads and
writes. This tool joins the two, so you know which modules to pull from a donor car and
which of them actually have to be alive on the bench.

Everything is parsed out of the opendbc sources with the stdlib, so this runs on a fresh
checkout without building opendbc or activating the openpilot venv.

Usage:
  openpilot/tools/car_porting/toyota_bench_parts.py                        # default platform, text
  openpilot/tools/car_porting/toyota_bench_parts.py --list                 # all Toyota platforms
  openpilot/tools/car_porting/toyota_bench_parts.py TOYOTA_RAV4_TSS2
  openpilot/tools/car_porting/toyota_bench_parts.py TOYOTA_RAV4 TOYOTA_RAV4H --format markdown
"""

import argparse
import ast
import re
import string
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOYOTA_DIR = REPO_ROOT / "opendbc_repo" / "opendbc" / "car" / "toyota"
DBC_GENERATOR_DIR = REPO_ROOT / "opendbc_repo" / "opendbc" / "dbc" / "generator" / "toyota"

DEFAULT_PLATFORM = "TOYOTA_RAV4"
MAX_PART_NUMBERS = 6

INCLUDE_RE = re.compile(r'CM_ "IMPORT (.*?)";')
MESSAGE_RE = re.compile(r"^BO_ (\d+) (\w+) *: *\d+ *(\w+)?", re.M)

# What a part number prefix means to someone at a salvage yard. Curated, not from opendbc.
PART_PREFIX_HINTS = {
  "8965B": "EPS / power steering ECU",
  "8821F": "millimeter wave radar sensor",
  "8646F": "forward recognition camera",
  "88150": "driving support ECU (DSU)",
  "88151": "driving support ECU (DSU)",
  "F1526": "brake actuator / skid control ECU",
  "89663": "engine control module",
}

# Fallback when a part number prefix is not recognized, e.g. the TSS-P engine ECU entries,
# which are calibration identifiers rather than an 89663-xxxxx part number.
ECU_HINTS = {
  "abs": "brake actuator / skid control ECU",
  "dsu": "driving support ECU (DSU)",
  "engine": "engine control module",
  "eps": "EPS / power steering ECU",
  "fwdCamera": "forward recognition camera",
  "fwdRadar": "millimeter wave radar sensor",
  "hybrid": "hybrid control ECU",
  "hvac": "air conditioning amplifier",
  "srs": "SRS airbag ECU",
  "transmission": "transmission control ECU",
}


@dataclass
class Platform:
  name: str
  config_cls: str = "PlatformConfig"
  models: list[str] = field(default_factory=list)
  dbc: dict[str, str] = field(default_factory=dict)
  flags: set[str] = field(default_factory=set)


@dataclass
class ConfigClass:
  """A PlatformConfig subclass in values.py: the defaults it applies to every platform using it."""
  flags: set[str] = field(default_factory=set)
  dbc: dict[str, str] = field(default_factory=dict)
  # dbc overrides applied inside init() when a flag is set, e.g. RADAR_ACC drops the radar DBC
  conditional_dbc: list[tuple[str, dict[str, str]]] = field(default_factory=list)


@dataclass
class EcuEntry:
  ecu: str
  address: int
  subaddr: int | None
  versions: list[bytes]

  @property
  def part_numbers(self) -> list[str]:
    return sorted({clean_fw(v) for v in self.versions})


def clean_fw(raw: bytes) -> str:
  """Render a fingerprint blob roughly the way it is stamped on the part.

  Fingerprints are null padded, and many carry a leading count byte and concatenate several
  sub-versions. Non-printable bytes are escaped rather than dropped, so nothing is silently
  misrepresented.
  """
  out = []
  for b in raw.replace(b"\x00", b""):
    c = chr(b)
    out.append(c if c in string.printable and c not in string.whitespace else f"\\x{b:02x}")
  return "".join(out)


def part_hint(ecu: str, parts: list[str]) -> str:
  for part in parts:
    # skip any leading count byte so the prefix lines up with the printed part number
    stripped = re.sub(r"^(\\x[0-9a-f]{2})+", "", part)
    for prefix, hint in PART_PREFIX_HINTS.items():
      if stripped.startswith(prefix):
        return hint
  return ECU_HINTS.get(ecu, "-")


def _dbc_from_node(node: ast.AST) -> dict[str, str]:
  """Read a dbc_dict('pt', 'radar') call or a {Bus.pt: 'pt'} literal."""
  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dbc_dict":
    return {bus: a.value for bus, a in zip(("pt", "radar"), node.args, strict=False) if isinstance(a, ast.Constant)}
  if isinstance(node, ast.Dict):
    return {k.attr: v.value for k, v in zip(node.keys, node.values, strict=False)
            if isinstance(k, ast.Attribute) and isinstance(v, ast.Constant)}
  return {}


def _flag_names(node: ast.AST) -> set[str]:
  return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "ToyotaFlags"}


def _parse_config_class(node: ast.ClassDef) -> ConfigClass:
  cls = ConfigClass()

  for stmt in node.body:
    # dbc_dict: dict = field(default_factory=lambda: dbc_dict(...))
    if isinstance(stmt, ast.AnnAssign) and getattr(stmt.target, "id", None) == "dbc_dict" and isinstance(stmt.value, ast.Call):
      for kw in stmt.value.keywords:
        if kw.arg == "default_factory" and isinstance(kw.value, ast.Lambda):
          cls.dbc = _dbc_from_node(kw.value.body)

    if not (isinstance(stmt, ast.FunctionDef) and stmt.name == "init"):
      continue

    for sub in stmt.body:
      # self.flags |= ToyotaFlags.A | ToyotaFlags.B -- only augmented assignment sets flags,
      # so the ToyotaFlags references inside `if self.flags & ...` tests are not picked up
      if isinstance(sub, ast.AugAssign) and getattr(sub.target, "attr", None) == "flags":
        cls.flags |= _flag_names(sub.value)

      # if self.flags & ToyotaFlags.X: self.dbc_dict = {...}
      if isinstance(sub, ast.If):
        for inner in sub.body:
          if isinstance(inner, ast.Assign) and getattr(inner.targets[0], "attr", None) == "dbc_dict":
            for flag in _flag_names(sub.test):
              cls.conditional_dbc.append((flag, _dbc_from_node(inner.value)))

  return cls


def parse_platforms(path: Path) -> dict[str, Platform]:
  """Pull the CAR platform table out of values.py."""
  tree = ast.parse(path.read_text())
  config_classes = {n.name: _parse_config_class(n) for n in tree.body if isinstance(n, ast.ClassDef) and n.name.endswith("PlatformConfig")}

  car_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CAR")
  platforms: dict[str, Platform] = {}
  for node in car_cls.body:
    if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.targets[0], ast.Name)):
      continue

    call, name = node.value, node.targets[0].id
    config_cls = call.func.id if isinstance(call.func, ast.Name) else "PlatformConfig"
    defaults = config_classes.get(config_cls, ConfigClass())
    platform = Platform(name=name, config_cls=config_cls, flags=set(defaults.flags), dbc=dict(defaults.dbc))

    # the first positional arg is a list of ToyotaCarDocs("Toyota RAV4 2016", ...)
    if call.args and isinstance(call.args[0], ast.List):
      platform.models = [doc.args[0].value for doc in call.args[0].elts
                         if isinstance(doc, ast.Call) and doc.args and isinstance(doc.args[0], ast.Constant)]

    for arg in call.args[1:]:
      if dbc := _dbc_from_node(arg):
        platform.dbc = dbc

    for kw in call.keywords:
      if kw.arg == "flags":
        platform.flags |= _flag_names(kw.value)

    for flag, dbc in defaults.conditional_dbc:
      if flag in platform.flags:
        platform.dbc = dbc

    platforms[name] = platform
  return platforms


def parse_fingerprints(path: Path) -> dict[str, list[EcuEntry]]:
  """Pull FW_VERSIONS out of fingerprints.py, keeping the ECU / address / part number join."""
  tree = ast.parse(path.read_text())
  fw_node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "FW_VERSIONS")

  out: dict[str, list[EcuEntry]] = {}
  for car_key, ecus in zip(fw_node.keys, fw_node.values, strict=True):
    if not (isinstance(car_key, ast.Attribute) and isinstance(ecus, ast.Dict)):
      continue

    entries = []
    for key, versions in zip(ecus.keys, ecus.values, strict=True):
      if not (isinstance(key, ast.Tuple) and isinstance(versions, ast.List)):
        continue
      ecu, addr, subaddr = key.elts
      entries.append(EcuEntry(
        ecu=ecu.attr,
        address=ast.literal_eval(addr),
        subaddr=ast.literal_eval(subaddr),
        versions=[ast.literal_eval(v) for v in versions.elts],
      ))
    out[car_key.attr] = entries
  return out


def parse_dbc_messages(dbc_dir: Path, dbc_name: str) -> dict[str, tuple[int, str]]:
  """Map message name -> (address, transmitting node) for one platform's powertrain DBC.

  The generator inputs are read, following their IMPORT directives the way
  opendbc/dbc/generator/generator.py does, rather than the *_generated.dbc outputs, so this
  works with nothing built. 'XXX' means the DBC does not attribute the message to a node.
  """
  messages: dict[str, tuple[int, str]] = {}
  pending = [dbc_name.removesuffix("_generated") + ".dbc"] if dbc_name else []
  seen: set[str] = set()

  while pending:
    filename = pending.pop()
    path = dbc_dir / filename
    if filename in seen or not path.exists():
      continue
    seen.add(filename)

    src = path.read_text()
    pending += INCLUDE_RE.findall(src)
    for addr, name, node in MESSAGE_RE.findall(src):
      messages.setdefault(name, (int(addr), node or "XXX"))
  return messages


def parse_bus_map(carstate: Path) -> dict[str, int]:
  """Map the carstate parser names to their CAN bus numbers."""
  tree = ast.parse(carstate.read_text())
  buses: dict[str, int] = {}
  for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "get_can_parsers":
      for sub in ast.walk(node):
        if isinstance(sub, ast.Dict):
          for key, value in zip(sub.keys, sub.values, strict=False):
            if isinstance(key, ast.Attribute) and isinstance(value, ast.Call) and len(value.args) == 3:
              buses[key.attr] = ast.literal_eval(value.args[2])
  return buses


def parse_rx_messages(carstate: Path) -> dict[str, set[str]]:
  """Messages openpilot reads, grouped by the parser (and so the bus) they are read from."""
  src = carstate.read_text()
  rx: dict[str, set[str]] = defaultdict(set)
  for parser, bus in (("cp_cam", "cam"), ("cp", "pt")):
    for match in re.finditer(rf'(?<![\w.]){parser}\.vl(?:_all)?\["(\w+)"\]', src):
      name = match.group(1)
      if not any(name in msgs for msgs in rx.values()):
        rx[bus].add(name)
  return rx


def parse_tx_messages(toyotacan: Path) -> dict[str, int]:
  """Messages openpilot transmits, from the message builders in toyotacan.py."""
  return {name: int(bus) for name, bus in re.findall(r'make_can_msg\("(\w+)", *(\d+)', toyotacan.read_text())}


def render(platforms: list[Platform], fingerprints, rx, tx, buses, markdown: bool, all_versions: bool) -> str:
  h1, h2, code = ("## ", "### ", "`") if markdown else ("== ", "-- ", "")
  lines: list[str] = []

  def table(header: list[str], rows: list[list[str]]) -> None:
    if markdown:
      lines.append("| " + " | ".join(header) + " |")
      lines.append("|" + "|".join(["---"] * len(header)) + "|")
      lines.extend("| " + " | ".join(r) + " |" for r in rows)
    else:
      widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
      lines.append("  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)).rstrip())
      lines.append("  ".join("-" * w for w in widths))
      lines.extend("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows)
    lines.append("")

  for platform in platforms:
    lines += [f"{h1}{platform.name}", ""]
    lines.append(f"Donor cars: {', '.join(platform.models) or 'unknown'}")
    lines.append(f"Platform flags: {', '.join(sorted(platform.flags)) or 'none'}")
    lines.append(f"DBC: {', '.join(f'{k}={code}{v}{code}' for k, v in platform.dbc.items()) or 'unknown'}")
    lines += ["", f"{h2}Modules to source", ""]

    rows = []
    for entry in sorted(fingerprints.get(platform.name, []), key=lambda e: (e.ecu, e.address)):
      parts = entry.part_numbers
      shown = parts if all_versions else parts[:MAX_PART_NUMBERS]
      rendered = ", ".join(f"{code}{p}{code}" for p in shown)
      if len(shown) < len(parts):
        rendered += f", ... (+{len(parts) - len(shown)})"
      rows.append([entry.ecu, f"0x{entry.address:x}", "-" if entry.subaddr is None else f"0x{entry.subaddr:x}",
                   str(len(parts)), part_hint(entry.ecu, parts), rendered])
    table(["ECU", "Request", "Subaddr", "Known", "Part", "Part numbers"], rows)

    # Only the messages this platform's DBC defines: carstate.py covers every Toyota, so e.g.
    # the SecOC and LTA messages do not exist on an older TSS-P car.
    dbc_messages = parse_dbc_messages(DBC_GENERATOR_DIR, platform.dbc.get("pt", ""))

    lines += [f"{h2}CAN messages openpilot needs", ""]
    rows = []
    for bus_name, names in sorted(rx.items()):
      for name in sorted(names & dbc_messages.keys()):
        addr, node = dbc_messages[name]
        rows.append(["read", str(buses.get(bus_name, "?")), name, f"0x{addr:x}", node])
    for name, bus in sorted(tx.items()):
      if name in dbc_messages:
        addr, node = dbc_messages[name]
        rows.append(["write", str(bus), name, f"0x{addr:x}", node])
    table(["Dir", "Bus", "Message", "Address", "DBC node"], rows)

  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("platforms", nargs="*", default=[DEFAULT_PLATFORM], help=f"platform names (default: {DEFAULT_PLATFORM})")
  parser.add_argument("--list", action="store_true", help="list Toyota platforms with fingerprints and exit")
  parser.add_argument("--format", choices=["text", "markdown"], default="text", help="output format")
  parser.add_argument("--all-versions", action="store_true", help=f"print every known part number, not just the first {MAX_PART_NUMBERS}")
  args = parser.parse_args()

  if not TOYOTA_DIR.exists():
    print(f"opendbc is not checked out at {TOYOTA_DIR}\nrun: git submodule update --init opendbc_repo", file=sys.stderr)
    return 1

  all_platforms = parse_platforms(TOYOTA_DIR / "values.py")
  fingerprints = parse_fingerprints(TOYOTA_DIR / "fingerprints.py")

  if args.list:
    for name in sorted(all_platforms):
      print(f"{name:28} {len(fingerprints.get(name, [])):2d} ECUs  {', '.join(all_platforms[name].models)}")
    return 0

  if unknown := [p for p in args.platforms if p not in all_platforms]:
    print(f"unknown platform(s): {', '.join(unknown)}\nrun with --list to see the options", file=sys.stderr)
    return 1

  print(render(
    [all_platforms[p] for p in args.platforms],
    fingerprints,
    parse_rx_messages(TOYOTA_DIR / "carstate.py"),
    parse_tx_messages(TOYOTA_DIR / "toyotacan.py"),
    parse_bus_map(TOYOTA_DIR / "carstate.py"),
    markdown=args.format == "markdown",
    all_versions=args.all_versions,
  ))
  return 0


if __name__ == "__main__":
  sys.exit(main())
