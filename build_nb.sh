#!/bin/bash
# usage: ./build_nb.sh 01   -> converts src/01_*.md to ./01_*.ipynb and executes it in place
set -e
cd "$(dirname "$0")"
src=$(ls src/${1}_*.md)
name=$(basename "$src" .md)
jupytext --to ipynb --output "${name}.ipynb" "$src" >/dev/null
start=$(date +%s)
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "${name}.ipynb" 2>&1 | grep -vE "^\[NbConvertApp\] (Converting|Writing)" || true
end=$(date +%s)
python3 - "$name.ipynb" <<'EOF'
import json, sys
nb = json.load(open(sys.argv[1]))
n_err = 0
for i, c in enumerate(nb["cells"]):
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            n_err += 1
            print(f"ERROR in cell {i}: {o['ename']}: {o['evalue']}")
        if o.get("output_type") == "stream" and o.get("name") == "stderr":
            txt = "".join(o["text"])
            print(f"stderr in cell {i}: {txt[:300]}")
print(f"{sys.argv[1]}: {len(nb['cells'])} cells, {n_err} errors")
EOF
echo "executed in $((end-start)) s"
