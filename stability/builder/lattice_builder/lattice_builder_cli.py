#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 10:19:54 2025

@author: steven
"""
# lattice_builder_cli.py
import argparse
import yaml
import numpy as np
from openmm import unit


from lattice_builder import LatticeBuilder
# --- Custom YAML constructor for !Quantity ---
def quantity_constructor(loader, node):
    mapping = loader.construct_mapping(node)
    value = float(mapping["value"])
    unit_str = mapping["unit"]
    # Map unit strings to OpenMM units
    unit_map = {
        "dalton": unit.dalton,
        "nanometer": unit.nanometer,
        "millivolt": (unit.milli*unit.volt),
        "elementary_charge": unit.elementary_charge,
    }
    if unit_str not in unit_map:
        raise ValueError(f"Unknown unit in YAML: {unit_str}")
    return value * unit_map[unit_str]


yaml.SafeLoader.add_constructor("!Quantity", quantity_constructor)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_structure(builder, run_cfg):
    """Load from CIF or lattice+positions+numbers"""

    if "type_map" in run_cfg:
        builder.type_map = run_cfg["type_map"]

    if "cif" in run_cfg:
        builder.load_from_cif(run_cfg["cif"])
    elif all(k in run_cfg for k in ("lattice", "positions", "numbers")):
        lattice = np.array(run_cfg["lattice"], dtype=float)
        positions = np.array(run_cfg["positions"], dtype=float)
        numbers = run_cfg["numbers"]
        builder.set_structure(lattice, positions, numbers)
    else:
        raise ValueError("run_config must have either 'cif' or ('lattice','positions','numbers')")


def main():
    parser = argparse.ArgumentParser(description="Build lattice with particle+run configs")
    parser.add_argument("run_config", help="YAML run configuration file")
    parser.add_argument("particle_config", help="YAML particle configuration file")
    args = parser.parse_args()

    run_cfg = load_yaml(args.run_config)
    part_cfg = load_yaml(args.particle_config)

    builder = LatticeBuilder()
    build_structure(builder, run_cfg)

    # Resize
    result = builder.resize_to_match_radii(
        radii_dict = part_cfg["radii"],
        brush_length=run_cfg["resize"].get("brush_length", 10.0),
        matrix=tuple(run_cfg["resize"].get("supercell", (3, 3, 3))),
        spacing=run_cfg["resize"].get("spacing", 10.0),
        factor=run_cfg["resize"].get("scale_factor", 1.1),
        start=run_cfg["resize"].get("scale_start", 0.0),
        padding = run_cfg["resize"].get("padding", 1000.0)
    )
    
    
    # print("Radii fit accuracy %.1f{}")
    print("Radii fit accuracy: %0.1f" % result['min_gap'])
    if result['min_gap']>1:
        print("Radii fit failed, try a lower scale_start and/or scale_factor")
    positions, types, box = result["positions"], result["types"], result["box"]


    # Outputs
    outputs = run_cfg.get("outputs", {})
    if "lammps" in outputs:
        builder.write_lammps_data(
            filename=outputs["lammps"],
            positions=positions,
            types=types,
            box=box,
        )


    print("✅ Done! Wrote:", outputs)


if __name__ == "__main__":
    main()

