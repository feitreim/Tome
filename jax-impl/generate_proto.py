"""Generate Python gRPC code from protobuf schema."""

import subprocess
import sys
from pathlib import Path

PROTO_DIR = Path(__file__).parent.parent / "scheduler" / "proto"
OUT_DIR = Path(__file__).parent / "generated"

OUT_DIR.mkdir(exist_ok=True)

proto_file = PROTO_DIR / "inference.proto"

cmd = [
    sys.executable,
    "-m",
    "grpc_tools.protoc",
    f"--proto_path={PROTO_DIR}",
    f"--python_out={OUT_DIR}",
    f"--grpc_python_out={OUT_DIR}",
    f"--pyi_out={OUT_DIR}",
    str(proto_file),
]

print(f"Generating Python gRPC code from {proto_file}")
print(f"Output directory: {OUT_DIR}")
subprocess.run(cmd, check=True)

(OUT_DIR / "__init__.py").touch()

print("Done! Generated files:")
for f in OUT_DIR.glob("*.py"):
    print(f"  - {f.relative_to(Path(__file__).parent)}")
