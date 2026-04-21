#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 12:20:19 2025

@author: steven
"""


import numpy as np
import gc
import time
from contextlib import contextmanager
import matplotlib as mpl
import matplotlib.pyplot as plt
# Pymatgen imports for structures
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

# OpenMM imports
import openmm
from openmm import unit
from colloids.colloid_potentials_algebraic import ColloidPotentialsAlgebraic
from colloids.colloid_potentials_parameters import ColloidPotentialsParameters


from pymatgen.core import Lattice, Structure, Molecule
from scipy.optimize import minimize_scalar

import gsd.hoomd
import os

class EnergyCalculator:
    

    def __init__(self, debye = 5.0):
        self.debye = debye
        # Default colloid brush parameters
        self.brush_length = 10.0 * unit.nanometer
        self.brush_density = 0.09 / (unit.nanometer ** 2)
        self.temperature = 298.0 * unit.kelvin
        self.eps = 80.0

    def set_params(self,r_pos, r_neg, q_pos, q_neg):
        self.r_positive = r_pos 
        self.r_negative = r_neg 
        self.charge_positive = q_pos
        self.charge_negative = q_neg 
        
    def get_12_labels(self,types):
         """Label atoms as 'A' or 'B' based on first atomic number."""
         first = types[0]
         return ['1' if x == first else '2' for x in types]

        
    def read_gsd(self, filename, mass_unit=unit.dalton, length_unit=unit.nanometer, electric_potential_unit=(unit.milli * unit.volt)):
        """
        Read a .gsd file and return positions, types, box dimensions, radii, charges, masses.
        """
        with gsd.hoomd.open(name=filename, mode='r') as traj:
            frame = traj[-1]  # last frame
            positions = np.array(frame.particles.position) * length_unit
            types = [frame.particles.types[i] for i in frame.particles.typeid]
            box = np.array(frame.configuration.box[:3])+ 3000.0  # Lx, Ly, Lz
            types_ls = self.get_12_labels(frame.particles.typeid)
            # Extract masses, charges, diameters and convert to units
            masses = {t: frame.particles.mass[i]*mass_unit for i,t in enumerate(types_ls)}
            charges = {t: frame.particles.charge[i]*electric_potential_unit for i,t in enumerate(types_ls)}
            radii = {t: 0.5*frame.particles.diameter[i]*length_unit for i,t in enumerate(types_ls)}
        

        self.set_params(radii['2'],radii['1'],charges['2'],charges['1'])
        
        return positions, types, box, radii, charges, masses

    def read_lmp(self,filename):
        """Read a simple LAMMPS data file with 'Atoms # full' style written by ASE."""
        with open(filename, "r") as f:
            lines = f.readlines()
    
        # --- parse box dimensions ---
        xline = [l for l in lines if "xlo xhi" in l][0].split()
        yline = [l for l in lines if "ylo yhi" in l][0].split()
        zline = [l for l in lines if "zlo zhi" in l][0].split()
    
        xlo, xhi = float(xline[0]), float(xline[1])
        ylo, yhi = float(yline[0]), float(yline[1])
        zlo, zhi = float(zline[0]), float(zline[1])
    
        # triclinic tilt (ASE writes zeros usually, but we parse anyway)
        tilt_line = [l for l in lines if "xy xz yz" in l]
        xy, xz, yz = (0.0, 0.0, 0.0)
        if tilt_line:
            parts = tilt_line[0].split()
            xy, xz, yz = float(parts[0]), float(parts[1]), float(parts[2])
    
        # box as a 3x3 matrix
        # box = np.array([
        #     [xhi - xlo, xy, xz],
        #     [0.0, yhi - ylo, yz],
        #     [0.0, 0.0, zhi - zlo]
        # ], dtype=float)
        
        box = [xhi - xlo+ 3000.0,yhi - ylo+ 3000.0,zhi - zlo+ 3000.0]
        # --- find atom section ---
        try:
            atom_start = lines.index("Atoms # full\n") + 2
        except ValueError:
            raise RuntimeError("Could not find 'Atoms # full' section in LAMMPS file")
    
        atoms = []
        for line in lines[atom_start:]:
            if not line.strip():
                break
            parts = line.split()
            atom_id = int(parts[0])
            atom_type = int(parts[2])  # type index (1,2,...)
            x, y, z = map(float, parts[4:7])
            atoms.append((atom_id, atom_type, x, y, z))
    
        atoms.sort(key=lambda t: t[0])  # ensure order by ID
        types = np.array([t[1] for t in atoms], dtype=str)
        positions = np.array([[t[2], t[3], t[4]] for t in atoms], dtype=float)
    
        return positions, types, box
    
    def compute_energy(self, positions, types, box, platform_name="Reference", optimize_lattice=False, scale_rate=0.005):
        """
        Compute potential energy per particle for a given structure.
        Optionally optimize the lattice scaling factor to minimize energy.
    
        Parameters
        ----------
        positions : ndarray
            N x 3 Cartesian coordinates
        types : list of 'A' or 'B'
            Particle types
        box : list/tuple
            Simulation box [Lx, Ly, Lz]
        platform_name : str
            OpenMM platform ("Reference" or "CUDA")
        optimize_lattice : bool
            If True, performs 1D lattice scaling optimization before returning energy.
        scale_bounds : tuple
            Bounds for lattice scale search (min, max).
    
        Returns
        -------
        float
            Potential energy per particle (kT units)
        """
        system = openmm.System()
        system.setDefaultPeriodicBoxVectors(
       [box[0], 0, 0], [0, box[1], 0], [0, 0, box[2]]
   )
        
        if optimize_lattice:
            self.brush_length = 11.0 * unit.nanometer

        params = ColloidPotentialsParameters(
            brush_density=self.brush_density, brush_length=self.brush_length,
            debye_length=self.debye, temperature=self.temperature,
            dielectric_constant=self.eps)
        
        colloid_potentials = ColloidPotentialsAlgebraic(
            colloid_potentials_parameters=params, use_log=False)

    
        for i,type_1 in enumerate(types):
            if type_1 == '1':
                system.addParticle(mass=1.0)
                colloid_potentials.add_particle(radius =self.r_positive,surface_potential =self.charge_positive)
            else:
                system.addParticle(mass=1.8)
                colloid_potentials.add_particle(radius =self.r_negative,surface_potential =self.charge_negative)
                
        # Add forces.
        for potential in colloid_potentials.yield_potentials():
           system.addForce(potential)
            
        for force in system.getForces():
            force.setCutoffDistance(1000*unit.nanometer)
            assert force.usesPeriodicBoundaryConditions()
            assert not force.getUseLongRangeCorrection()
        
        # Set up platform and context. The platform_name is typically Reference or CUDA.
        platform = openmm.Platform.getPlatformByName(platform_name)
        dummy_integrator = openmm.LangevinIntegrator(
            params.temperature.value_in_unit(unit.kelvin), 0.0, 0.0)    
        
        
        context = openmm.Context(system, dummy_integrator, platform)
        context.setPositions(positions)
        state = context.getState(getEnergy=True)
        U = state.getPotentialEnergy()
        U = U / (unit.BOLTZMANN_CONSTANT_kB * params.temperature * unit.AVOGADRO_CONSTANT_NA)
        U_temp = U
        best_positions =positions.copy()
        
        
        if optimize_lattice:
            
            scale = 0.95
            base_positions = positions.copy()
            while U>=U_temp:
                
                scaled_positions = base_positions * scale
                context.setPositions(scaled_positions)
                state = context.getState(getEnergy=True)
                U_temp = state.getPotentialEnergy()
                
                print(U_temp)
                # print("Optimizing: U = %.1f" % U_temp
                U_temp = U_temp / (unit.BOLTZMANN_CONSTANT_kB * params.temperature * unit.AVOGADRO_CONSTANT_NA)
                print("(per particle %.1f KbT)" % (float(U_temp)/len(positions)) )
                scale = scale-scale_rate
                if U_temp<= U:
                    U = U_temp
                
            best_positions = base_positions * scale
        return U/len(positions), best_positions
    
    def write_gsd(self, filename, positions, types, box, radii_dict, charges_dict, masses_dict,
                  mass_unit=unit.dalton, length_unit = unit.nanometer, electric_potential_unit= (unit.milli * unit.volt)):
     
         
         N = len(positions)
        
         with gsd.hoomd.open(name=filename, mode='w') as traj:
             frame = gsd.hoomd.Frame()
             frame.configuration.step = 0
             frame.configuration.dimensions = 3
             frame.configuration.box = [box[0], box[1], box[2], 0.0, 0.0, 0.0]
            
             # Particles
             frame.particles.N = N
             frame.particles.types = sorted(set(types))
             frame.particles.typeid = np.array(
             [frame.particles.types.index(t) for t in types], dtype=np.int32
             )
             frame.particles.position = np.array(positions, dtype=np.float32)
            
             # Mass, charge, diameter using units
             frame.particles.mass = np.array(
             [masses_dict[frame.particles.types[i]].value_in_unit(mass_unit)
              for i in frame.particles.typeid], dtype=np.float32
             )
             frame.particles.charge = np.array(
             [charges_dict[frame.particles.types[i]].value_in_unit(electric_potential_unit)
              for i in frame.particles.typeid], dtype=np.float32
             )
             frame.particles.diameter = np.array(
             [2.0 * radii_dict[frame.particles.types[i]].value_in_unit(length_unit)
              for i in frame.particles.typeid], dtype=np.float32
             )
            
             traj.append(frame)
 