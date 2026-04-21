#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
from . import EnergyCalculator  # your class file
from openmm import unit


def quantity_constructor(loader, node):
        value = loader.construct_mapping(node)
        val = value['value']
        u = value.get('unit', None)
        if u == 'nanometer':
            return val * unit.nanometer
        elif u == '/(nanometer**2)':
            return val / (unit.nanometer**2)
        elif u == 'kelvin':
            return val * unit.kelvin
        elif u == 'kilojoule/mole':
            return val * unit.kilojoule_per_mole
        elif u == 'picosecond':
            return val * unit.picosecond
        elif u == '/picosecond':
            return val / unit.picosecond
        elif u == 'meter/(second**2)':
            return val * unit.meter/unit.second**2
        elif u == 'gram/(centimeter**3)':
            return val * unit.gram/unit.centimeter**3
        elif u == 'millivolt':
            return val * (unit.milli*unit.volt)
        elif u == "dalton": 
            return val *unit.dalton
        else:
            return val


yaml.SafeLoader.add_constructor("!Quantity", quantity_constructor)
yaml.SafeLoader.add_constructor('!Copy', lambda loader, node: loader.construct_mapping(node)['key'])

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# --- CLI wrapper ---
class EnergyCLI:
    def __init__(self, run_file, config_file,particle_file):
        self.run_params = load_yaml(run_file)
        self.config_params = load_yaml(config_file)
        self.part_cfg= load_yaml(particle_file)
        
        # Initialize calculator
        debye = self.run_params.get('debye_length', 5.0)  # already openmm unit
        self.calculator = EnergyCalculator(debye)

        
    def run(self):
        input_file = self.config_params.get('input_file')
        output_file = self.config_params.get('output_file', 'optimized.gsd')
        scale_rate = self.config_params.get('scale_rate', 0.005)
        optimize_lattice = self.config_params.get('optimize_lattice', False)
        platform_name = self.config_params.get('platform_name', 'CPU')

        if not input_file:
            raise ValueError("Please specify an 'input_file' in run_config.yaml")

        # Determine file type
        ext = os.path.splitext(input_file)[1].lower()
        if ext == '.gsd':
            positions, types, box, radii, charges, masses = self.calculator.read_gsd(input_file)
        elif ext == '.lmp':
            positions, types, box = self.calculator.read_lmp(input_file)
            r_pos = self.part_cfg["radii"]["1"]
            r_neg = self.part_cfg["radii"]["2"]
            q_pos = self.part_cfg["surface_potentials"]["1"]
            q_neg = self.part_cfg["surface_potentials"]["2"]
            
            self.calculator.set_params(r_pos, r_neg, q_pos, q_neg)
            
            radii = {'1': self.calculator.r_positive, '2': self.calculator.r_negative}
            charges = {'1': self.calculator.charge_positive, '2': self.calculator.charge_negative}
            masses = {'1': self.part_cfg["masses"]["1"], '2': self.part_cfg["masses"]["2"]}
            
            # Set particle parameters from run_config.yaml if present
            
            

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # Compute energy
        energy, opt_positions = self.calculator.compute_energy(
            positions, types, box, platform_name=platform_name,
            optimize_lattice=optimize_lattice, scale_rate=scale_rate
        )

        # Write optimized frame
        self.calculator.write_gsd(output_file, opt_positions+1, types, box, radii, charges, masses)
        print(f"Energy: {energy} kT, results written to {output_file}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Energy computation for colloid configurations")
    parser.add_argument("run", help="Path to run.yaml (physical parameters)")
    parser.add_argument("run_config", help="Path to run_config.yaml (input/output etc.)")
    parser.add_argument("configuration", help="YAML particle configuration file")
    args = parser.parse_args()

    cli = EnergyCLI(args.run, args.run_config,args.configuration)
    cli.run()


if __name__ == "__main__":
    main()