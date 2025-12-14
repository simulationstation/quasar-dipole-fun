#!/usr/bin/env python3
"""Inspect model_comparison.json produced by metropolis_hastings_sampler_new.py --compare-models."""
import json, sys

def main():
    fn = sys.argv[1] if len(sys.argv) > 1 else "model_comparison.json"
    with open(fn, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        print("Empty comparison file.")
        return
    best = data[0]
    base = best["waic"]
    print(f"Best model: {best['model']}  WAIC={best['waic']:.3f}  p_waic={best['p_waic']:.3f}  lppd={best['lppd']:.3f}")
    print("\nRanked:")
    for row in data:
        d = row["waic"] - base
        print(f"  {row['model']:<10} WAIC={row['waic']:.3f}  Δ={d:+.3f}  n_params={row['n_params']}")
if __name__ == "__main__":
    main()
