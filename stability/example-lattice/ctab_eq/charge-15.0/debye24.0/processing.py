#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 16:13:54 2025

@author: steven
"""
import os
import yaml
from energy_calculator import EnergyCalculator  # your class file
import numpy as np
from openmm import unit
import matplotlib.pyplot as plt
import pandas as pd

import freud

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
    
dir_path =  os.path.dirname(os.path.abspath(__file__))

all_items = os.listdir('.') 
run_params = load_yaml(dir_path+r'/run.yaml')
part_parms = load_yaml(dir_path+r'/configuration.yaml')

cal = EnergyCalculator(run_params['debye_length'])
positions, types, box, radii, charges, masses= cal.read_gsd(dir_path+r'/trajectory.gsd')
types = np.array(types)

r_pos = part_parms["radii"]["1"]
r_neg =part_parms["radii"]["2"]
q_pos = part_parms["surface_potentials"]["1"]
q_neg = part_parms["surface_potentials"]["2"]

cal.set_params(r_pos, r_neg, q_pos, q_neg)


energy, pos = cal.compute_energy(positions.value_in_unit(unit.nanometer), types,box,optimize_lattice = True)


pos_1 = positions[types == '1']
types_1 = types[types == '1']
energy_1, pos = cal.compute_energy(pos_1.value_in_unit(unit.nanometer), types_1,box)

pos_2 = positions[types == '2']
types_2 = types[types == '2']
energy_2, pos = cal.compute_energy(pos_2.value_in_unit(unit.nanometer), types_2,box)



box = freud.box.Box(30000, 30000, 30000)

r_max = 1000
bins = 400
rdf = freud.density.RDF(bins,r_max = r_max)

rdf.compute(system=(box, pos_2.value_in_unit(unit.nanometer)), reset=False)

fig,ax = plt.subplots(figsize = [4,3],dpi = 300)
fig.tight_layout()

rdf = freud.density.RDF(bins,r_max = r_max)
rdf.compute(system=(box, positions.value_in_unit(unit.nanometer)), reset=False)
rdf_all = getattr(rdf, 'rdf')

ax.plot(rdf.bin_centers, getattr(rdf, 'rdf'), 'k-',lw = 0.5,alpha = 0.5 ,label = 'all')


rdf = freud.density.RDF(bins,r_max = r_max)
rdf.compute(system=(box, pos_1.value_in_unit(unit.nanometer)), reset=False)
rdf_1with1 = getattr(rdf, 'rdf')

ax.plot(rdf.bin_centers, getattr(rdf, 'rdf'), 'b-',lw = 1,alpha = 0.5 ,label = '1-1')

rdf = freud.density.RDF(bins,r_max = r_max)
rdf.compute(system=(box, pos_2.value_in_unit(unit.nanometer)), reset=False)
rdf_2with2 = getattr(rdf, 'rdf')

ax.plot(rdf.bin_centers, getattr(rdf, 'rdf'), 'r-',lw = 1,alpha = 0.5, label = '2-2')


rdf_data = pd.DataFrame({'q nm':rdf.bin_centers, 'all': rdf_all,'1-1':rdf_1with1,'2-2':rdf_2with2} )

energies = pd.DataFrame([['total',energy], ['1-1', energy_1], ['2-2', energy_2 ]])

ax.set(xlabel = 'q (nm)', ylabel = 'rdf')
ax.legend()
fig.savefig(dir_path+"/rdf.svg")
fig.savefig(dir_path+"/rdf.png")
rdf_data.to_csv(dir_path+"/rdf_data.csv")
energies.to_csv(dir_path+"/energies.csv")

print("processing complete: %0.1f  KbT" %energy)