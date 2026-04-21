#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 10:22:05 2025

@author: steven
"""


import numpy as np
import gc
import time
import sys
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

class LatticeBuilder:
    """
    Handles structure creation from CIF or coordinates,
    supercell expansion, ASE/pymatgen structure conversion,
    and resizing to match colloid radii.
    """

    def __init__(self, cif_path=None):
        self.structure = None
        self.positions = None
        self.types = None
        self.scale = None
        self.box = None
        self.type_map = None

        if cif_path:
            self.load_from_cif(cif_path)

    def load_from_cif(self, cif_path):
        """Load structure from CIF file using pymatgen."""
        parser = CifParser(cif_path)
        self.structure = parser.get_structures()[0]

    def set_structure(self, lattice, coords, species):
        """Manually define a structure (instead of CIF)."""
        self.structure = Structure(lattice, species, coords)

    def make_supercell(self, matrix=(3, 3, 3)):
        """Return a new supercell structure."""
        if self.structure is None:
            raise ValueError("No structure defined")
        return self.structure.make_supercell(matrix, in_place=False)

    def set_colloid_labels_atomicnum(self, atomic_numbers):
        """Label atoms as '1' ... 'N' based on atomic number."""
        element_list = np.unique(atomic_numbers).tolist()
        type_map = {atomic_number: element_list.index(atomic_number)+1 for atomic_number in atomic_numbers}
        print("\tElement map:",type_map)
        type_list = [ type_map[atomic_number] for atomic_number in atomic_numbers ]
        return type_list

    def set_colloid_labels_typemap(self, atomic_species):
        """Label atoms as '1' ... 'N' based on species using pymatgen atom names."""
        element_list = np.unique([str(x.name) for x in atomic_species]).tolist()
        print("\tElement map:",self.type_map)
        try:
            type_list = [ self.type_map[species.name] for species in atomic_species ]
            return type_list
        except KeyError:
            print("Error: You must specify a type index for all elements in the element list:",element_list)
            sys.exit(1)

    # -------------------------
    # NEW resizing functionality
    # -------------------------

    def _calculate_distances(self, positions):
        """Pairwise distance matrix."""
        try:
            from scipy.spatial import distance_matrix
            dists = distance_matrix(positions,positions)
        except ImportError:
            n = len(positions)
            dists = np.zeros((n, n))
            for i in range(n):
                dists[i,:] = np.linalg.norm(positions-positions[i],axis=1)
        return dists

    def _check_overlap(self, distances, radii):
        """Check if any particles overlap."""
        n = len(radii)
        for i in range(n):
            for j in range(i + 1, n):
                if distances[i, j] <= (radii[i] + radii[j]):
                    return True
        return False

    def give_smallest_connection(self, distances, radii):
        """
        Return smallest effective distance (gap between particle surfaces).

        Effective distance = d - (r_i + r_j).
        """
        n = len(radii)
        min_dist = np.inf
        min_pair = None

        eff_matrix = np.zeros((n,n))

        for i in range(n):
            for j in range(i + 1, n):
                eff = distances[i, j] - (radii[i] + radii[j])
                eff_matrix[i,j] = eff
                if eff < min_dist:
                    min_dist = eff
                    min_pair = (i, j)

        return min_dist, min_pair

    # -------------------------
    # Resizing
    # -------------------------

    def resize_to_match_radii(
        self, radii_dict, brush_length=10.0, matrix=(3, 3, 3),
        spacing=10.0, factor=1.1, start=0.0, padding = 1000, center_origin=True, test_matrix=(3,3,3)):
        """
        Expand supercell until no overlaps remain given target radii.

        Parameters
        ----------
        radii_dict : dict
            Particle radii (in nanometers).
        brush_length : float
            Extra radius from polymer brush (nm).
        spacing : float
            Extra gap added to effective radii.
        factor : float
            Scale-up increment factor.
        start : float
            Starting scale factor.
        center_origin : bool
            Put center-of-gravity at 0,0,0
        """
        sc = self.make_supercell(test_matrix)
        positions = sc.cart_coords
        print("Test system size:",positions.shape)

        if self.type_map is None:
            print("Setting atom types based on elements in structure file")
            types = self.set_colloid_labels_atomicnum(sc.atomic_numbers)
        else:
            print("Setting atom types based on elements in type_map")
            types = self.set_colloid_labels_typemap(sc.species)

        # Effective radii in same units as positions
        radii = [radii_dict[str(t)].value_in_unit(unit.nanometer) for t in types]
        radii = np.array(radii) + brush_length + spacing

        dists = self._calculate_distances(positions)
        i = 0

        while self._check_overlap(dists, radii):
            scale = start + i * factor
            positions_exp = positions * scale
            dists = self._calculate_distances(positions_exp)
            i += 1

        print("Optimal scale factor is",scale)

        # Check smallest connection after scaling
        print("Computing closest pairs...")
        min_gap, pair = self.give_smallest_connection(dists, radii)
        print("...closest distance is",min_gap)

        self.structure = self.make_supercell(matrix)
        positions2 = self.structure.cart_coords
        positions_exp = positions2*scale

        print("Final system size:",positions2.shape)

        if center_origin is True:
            print("Centering the crystal origin at 0,0,0")
            positions_exp -= positions_exp.mean(axis=0)

        if self.type_map is None:
            types = self.set_colloid_labels_atomicnum(self.structure.atomic_numbers)
        else:
            types = self.set_colloid_labels_typemap(self.structure.species)

#        radii = [radii_dict[str(t)].value_in_unit(unit.nanometer) for t in types]
#        radii = np.array(radii) + brush_length + spacing
        
        # Store resized system
        self.positions = positions_exp
        self.types = types
        self.scale = scale
        self.box =  np.max(self.positions, axis=0) + np.max(radii) + padding

        return {
            "positions": self.positions,
            "types": self.types,
            "box": self.box,
            "scale": self.scale,
            "min_gap": min_gap,
            "min_pair": pair,
        }
    
    
    def write_lammps_data(self, filename, positions, types, box, atom_type_map=None, triclinic=True):
         """
         Write a LAMMPS .lmp data file in 'full' style (no bonds).
         
         Parameters
         ----------
         filename : str
             Path to output file.
         positions : ndarray (N,3)
             Atomic positions in Cartesian coordinates.
         types : list of str
             Particle type labels (e.g. ['A','B',...]).
         box : array_like (3,)
             Simulation box size along x,y,z (for orthogonal).
         atom_type_map : dict, optional
             Map from type labels (e.g. {'A':1, 'B':2}).
             If None, assigned in order of appearance.
         triclinic : bool
             If True, writes xy, xz, yz tilt factors (all set to 0).
         """
         positions = np.asarray(positions)
         n_atoms = len(positions)
         uniq_types = sorted(set(types))
     
         if atom_type_map is None:
             atom_type_map = {t: i+1 for i, t in enumerate(uniq_types)}
     
         n_types = len(atom_type_map)
     
         with open(filename, "w") as f:
             # Header counts
             f.write("(written by ASE)\n\n")
             f.write(f"{n_atoms} atoms\n")
             f.write("0 bonds\n")
             f.write(f"{n_types} atom types\n")
             f.write("0 bond types\n\n")
     
             # Box boundaries
             f.write(f"0.0 {box[0]:.6f} xlo xhi\n")
             f.write(f"0.0 {box[1]:.6f} ylo yhi\n")
             f.write(f"0.0 {box[2]:.6f} zlo zhi\n")
             if triclinic:
                 f.write("0.0 0.0 0.0 xy xz yz\n")
             f.write("\n")
     
             # Atom section
             f.write("Atoms # full\n\n")
             for i, (pos, t) in enumerate(zip(positions, types), start=1):
                 type_id = atom_type_map[t]
                 f.write(f"{i:6d}   0   {type_id:d}   0.0   {pos[0]:.6f}   {pos[1]:.6f}   {pos[2]:.6f}\n")
                 



#%%
# import os
# dir_path =  os.path.dirname(os.path.abspath(__file__))
# cif_path = os.path.normpath(dir_path + r"/CsCl.cif")
# lmp_path = os.path.normpath(dir_path + r"/CIF/LAMMPS/CsCl.lmp")
# gsd_path = os.path.normpath(dir_path + r"/CIF/OpenMM/CsCl/first_frame.gsd")



# builder = LatticeBuilder(cif_path)
# result = builder.resize_to_match_radii(r_pos=102, r_neg=120, brush_length=18 ,factor = 0.5,start=60,matrix=[8,8,8])
