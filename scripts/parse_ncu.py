#!/usr/bin/env python3
"""Sum dram__bytes_read/write from an `ncu --csv` log into GB moved per profiled decode step."""
import argparse
import csv
import io
import os

UNIT = {"byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,
        "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9}


def parse(path: str) -> tuple[float, float, int]:
    """Returns (read_bytes, write_bytes, n_kernels). Tolerates the non-CSV lines ncu prints before the header."""
    text = open(path).read().splitlines()
    start = next(i for i, l in enumerate(text) if l.startswith('"ID"'))
    rd = csv.DictReader(io.StringIO("\n".join(text[start:])))
    read = write = 0.0
    kernels = set()
    for r in rd:
        name, unit, val = r["Metric Name"], r["Metric Unit"], r["Metric Value"].replace(",", "")
        b = float(val) * UNIT.get(unit, 1.0)
        kernels.add(r["ID"])
        if name == "dram__bytes_read.sum":
            read += b
        elif name == "dram__bytes_write.sum":
            write += b
    return read, write, len(kernels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--key", required=True)
    ap.add_argument("--hw", required=True)
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--out", default="data/hbm_bytes.csv")
    a = ap.parse_args()
    r, w, n = parse(a.log)
    print(f"{n} kernels: read {r / 1e9:.2f} GB, write {w / 1e9:.2f} GB, total {(r + w) / 1e9:.2f} GB per decode step")
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["model", "hw", "S", "B", "hbm_gb_per_step", "hbm_read_gb", "hbm_write_gb", "n_kernels"])
        wr.writerow([a.key, a.hw, a.S, a.B, f"{(r + w) / 1e9:.4f}", f"{r / 1e9:.4f}", f"{w / 1e9:.4f}", n])


if __name__ == "__main__":
    main()
