from math import acos, pi
from random import choices, uniform
import re
from typing import Sequence, Union
from ase import Atoms, cell
from gsd.hoomd import Frame
import numpy as np
from colloids.colloids_create import ConfigurationGenerator


class ClusterGenerator(ConfigurationGenerator):
    """
    Generator for an initial configuration in a gsd.hoomd.Frame instance for a colloid simulation based on several
    clusters of colloids.

    Every cluster of colloids is assumed to have the same cell vectors. To generate the initial configuration,
    the clusters are first centered. Then, the shared cell vectors of the clusters are repeated in all three directions.
    Every replica of the cell is then filled with a randomly selected cluster from the list of clusters. The clusters
    are selected based on their relative weights. Every cluster can optionally be randomly rotated.

    The clusters are specified by Atoms objects generated from a lammps-data file.
    See https://docs.lammps.org/2001/data_format.html for information about this file format.

    To space out the clusters, one can increase a cluster padding factor that scales the lattice vectors before
    replication. This will also scale the box size. Additionally, one can increase a padding factor that scales just the
    overall box size and thus increases the distance between the outwards facing colloids and the walls. To make the
    simulation box smaller, use a padding factor less than 1.

    This dataclass assumes that the distances in the cluster are in units of nanometers.

    Any bonds in the cluster are added as constraints, with the constraint distance equal to the current bond length in
    the cluster definition. The bond lengths are not modified during the simulation.

    :param clusters:
        A sequence of clusters of colloids with equal cell vectors that are used to generate the initial configuration.
        All collooids in the centered clusters should lie in the unit cell defined by the lattice vectors.
    :type clusters: Sequence[Atoms]
    :param cluster_relative_weights:
        The relative weights of the clusters. The weights are used to randomly select a cluster from the list of
        clusters when generating the initial configuration.
        The weights should be positive.
    :type cluster_relative_weights: Sequence[float]
    :param lattice_repeats:
        The number of repeats of the lattice in the three directions of the lattice vectors of the cluster.
        If only a single integer is given, the same number of repeats is used in all directions.
        Every repeat should be positive.
    :type lattice_repeats: Union[int, list[int]]
    :param cluster_padding_factor:
        The factor by which the lattice vectors of every replicated cluster are scaled to space out the clusters.
        The cluster padding factor should be greater than zero.
    :type cluster_padding_factor: float
    :param padding_factor:
        The factor by which the overall lattice vectors are scaled to increase the distance between the outwards facing
        colloids and the walls. This will scale the box dimensions specified in the cluster specification file without
        changing the spacing in between clusters.
        The padding factor should be greater than zero.
    :type padding_factor: float
    :param random_rotation:
        Specifies whether replicas of the original clusters should be rotated in the initial configuration.
        If False, the orientation of the original cluster is preserved for all replicas when generating the initial
        configuration.
    :type random_rotation: bool

    :raises ValueError:
        If the cluster padding factor is not greater than zero.
        If the padding factor is not greater than zero.
        If no clusters are provided.
        If the number of clusters does not match the number of cluster probabilities.
        If any cluster probability is not greater than zero.
        If the clusters do not have the same cell vectors.
    """

    def __init__(self, clusters: Sequence[Atoms], cluster_relative_weights: Sequence[float],
                 lattice_repeats: Union[int, Sequence[int]], cluster_padding_factor: float,
                 padding_factor: float, random_rotation: bool) -> None:
        """Constructor of the ClusterGenerator class."""
        super().__init__()
        # The format of these arguments is already checked in configuration_parameters.py.
        self._clusters = clusters
        self._cluster_relative_weights = cluster_relative_weights
        self._lattice_repeats = lattice_repeats
        self._cluster_padding_factor = cluster_padding_factor
        self._padding_factor = padding_factor
        self._random_rotation = random_rotation
        if self._cluster_padding_factor <= 0.0:
            raise ValueError("The cluster padding factor must be greater than zero.")
        if self._padding_factor <= 0.0:
            raise ValueError("The padding factor must be greater than zero.")
        if not len(clusters) > 0:
            raise ValueError("At least one cluster must be provided.")
        if len(clusters) != len(cluster_relative_weights):
            raise ValueError("The number of clusters must match the number of cluster probabilities.")
        if not all(prob >= 0.0 for prob in cluster_relative_weights):
            raise ValueError("All cluster probabilities must be non-negative.")
        if not all(np.allclose(c.cell, clusters[0].cell) for c in clusters):
            raise ValueError("All clusters must have the same cell vectors.")

    @staticmethod
    def _extract_bonded_indices(bond_string: str) -> list[int]:
        """
        Extract the bonded indices from a bond string in the format of the ASE Atoms.arrays["bonds"] attribute.

        The bond string in ASE that was constructed from a lammps-data file looks like '1(1),2(1),3(1)',
        '2(1)', or '_'. If the string is '_', no bonds are present and an empty list is returned. Otherwise, the bonded
        indices are given by the integers in front of the parentheses (with the ignored bond type within the
        parentheses). These are extracted and returned as a list of integers.

        :param bond_string:
            The bond string in the format of the ASE Atoms.arrays["bonds"] attribute.
        :type bond_string: str

        :return:
            The bonded indices.
        :rtype: list[int]
        """
        if bond_string == "_":
            return []
        return [int(num) for num in re.findall(r"(\d+)\(", bond_string)]

    def generate_configuration(self) -> Frame:
        """
        Generate the initial positions of the colloids in a gsd.hoomd.Frame instance together with constraints.

        The generated frame should contain the following attributes:
        - frame.particles.N
        - frame.particles.position
        - frame.particles.types
        - frame.particles.typeid
        - frame.configuration.box
        - frame.constraints.N (optionally if constraints are present)
        - frame.constraints.value (optionally if constraints are present)
        - frame.constraints.group (optionally if constraints are present)

        The generated frame should not populate the following attributes:
        - frame.particles.type_shapes
        - frame.particles.diameter
        - frame.particles.charge
        - frame.particles.mass attributes

        :return:
            The initial configuration of the colloids.
        :rtype: gsd.hoomd.Frame

        :raises ValueError:
            If some positions in the centered and padded cluster are outside the unit cell.
        """
        centered_padded_clusters = [c.copy() for c in self._clusters]
        for c in centered_padded_clusters:
            # Do not use the vacuum argument of the center method, as it does not simply add vacuum
            # but can also decrease the size of the cell to get the desired vaccum.
            c.set_cell(np.array([self._cluster_padding_factor * c.cell[i] for i in range(3)]))
            c.center()
        unwrapped_positions = [c.get_positions(wrap=False) for c in centered_padded_clusters]
        wrapped_positions = [c.get_positions(wrap=True) for c in centered_padded_clusters]
        for i, (unwrapped, wrapped) in enumerate(zip(unwrapped_positions, wrapped_positions)):
            if not np.allclose(unwrapped, wrapped):
                raise ValueError(f"Some positions in the cluster {i} are outside of the unit cell. "
                                 f"Increase the cluster padding factor to allow for rotations.")

        if isinstance(self._lattice_repeats, int):
            r = (self._lattice_repeats, self._lattice_repeats, self._lattice_repeats)
        else:
            r = self._lattice_repeats

        # All cells are assumed to be the same so we can use the first one.
        cell = centered_padded_clusters[0].get_cell()
        individual_cell = np.array([self._padding_factor * cell[c] for c in range(3)])

        final_types_list = tuple(sorted(list(set([str(atom.number) for cluster_types in centered_padded_clusters for atom in cluster_types]))))

        # Generate a grid of displacement indices to be applied to the repeated clusters in each direction.
        cell_indices_x, cell_indices_y, cell_indices_z = np.meshgrid(range(r[0]), range(r[1]), range(r[2]), indexing="ij") # Shape: (r[0], r[1], r[2])
        cell_indices = np.stack((cell_indices_x, cell_indices_y, cell_indices_z), axis=-1) # Shape: (r[0], r[1], r[2], 3)

        # Create the actual displacements by multiplying the cell vectors with the grid of displacement indices.
        position_displacements = np.einsum("ij,xyzj->xyzi", cell, cell_indices) # Shape: (r[0], r[1], r[2], 3)

        # We need to pad the wrapped positions of the clusters, types, and bonds to be able to apply the random rotations to them.
        # The padding value is set to infinity so that any positions outside the original box are easily identified.
        max_cluster_size = max(len(c) for c in centered_padded_clusters)
        zero_min_positions = [c.get_positions(wrap=False) - np.min(c.get_positions(wrap=False), axis=0) for c in centered_padded_clusters]
        wrapped_positions_padded = np.array([np.pad(wrapped, ((0, max_cluster_size - len(wrapped)), (0, 0)), mode="constant", constant_values=np.nan) 
                                                   for wrapped in zero_min_positions])
        type_ids = [np.array([final_types_list.index(str(atom.number)) for atom in c], dtype=float) for c in centered_padded_clusters]
        type_ids_padded = np.array([np.pad(type_ids[i], (0, max_cluster_size - len(c)), mode="constant", constant_values=np.nan)
                                       for i, c in enumerate(centered_padded_clusters)])        

        bond_pairs = []
        bond_values = []
        bond_pair_offset_indices_count = []

        for c in centered_padded_clusters:
            if "bonds" not in c.arrays or len(c.arrays["bonds"]) == 0:
                bond_pairs += [np.empty((0, 2), dtype=float)]
                bond_values += [np.empty((0,), dtype=float)]
                bond_pair_offset_indices_count += [0]
                continue
            
            bond_pair = [(first_index, second_index) 
                         for first_index, bonds in enumerate(c.arrays["bonds"]) 
                         for second_index in self._extract_bonded_indices(bonds)]
            bond_pairs += [np.array(bond_pair, dtype=float)]
            bond_values += [np.array([c.get_distance(first_index, second_index) for first_index, second_index in bond_pair], dtype=float)]
            bond_pair_offset_indices_count += [len(bond_pair)]

        max_n_bonds = max(bond_pair_offset_indices_count) if bond_pair_offset_indices_count else 0
        bond_pairs_padded = np.array([np.pad(bond_pair, ((0, max_n_bonds - bond_pair.shape[0]), (0, 0)), mode="constant", constant_values=np.nan)
                                           for bond_pair in bond_pairs])
        bond_values_padded = np.array([np.pad(bond_value, (0, max_n_bonds - bond_value.size), mode="constant", constant_values=np.nan)
                                           for bond_value in bond_values])

        if bond_values_padded.ndim == 1:
            bond_values_padded = bond_values_padded[np.newaxis, :]
                
        # The offset for the bond pairs of the next cluster should repeat for each bond in the cluster its value
        # is given by the number of particles in the current cluster.
        bond_pair_offset_indices_padded = np.array([np.pad(np.ones(bond_pair_offset_index_count, dtype=float), (0, max_n_bonds - bond_pair_offset_index_count),
                                                           mode="constant", constant_values=np.nan)for bond_pair_offset_index_count in bond_pair_offset_indices_count])
        bond_pair_offset_values = np.array([len(c) for c in centered_padded_clusters]) 

        # Select a cluster for each displacement based on the relative weights of the clusters.
        cluster_selections = np.random.choice(len(centered_padded_clusters), size=r, p=self._cluster_relative_weights)

        # Assign the positions and types of the selected clusters to the corresponding displacements. The positions and bonds of the
        # clusters are padded to the same size so that they can be indexed by the cluster selections. Offsets must be applied to positions and
        # bond pairs to account for the displacements of the clusters. The bond values and types do not need to be modified as they are just 
        # the bond lengths in the original cluster definition.
        positions = wrapped_positions_padded[cluster_selections] # Shape: (r[0], r[1], r[2], max_cluster_size, 3)
        type_ids = type_ids_padded[cluster_selections] # Shape: (r[0], r[1], r[2], max_cluster_size)    

        if self._random_rotation:
            # Test to see if any positions in the clusters are outside the unit cell. If so, random rotations cannot be applied as 
            # they could cause some colloids to be outside the box. In this case, an error is raised and the user should increase 
            # the cluster padding factor to allow for rotations.
            outside_unit_cell = np.any((np.linalg.norm(zero_min_positions, axis=-1)[:, np.newaxis] - 0.5 * np.linalg.norm(individual_cell, axis=-1))[np.newaxis, :] > 0)
            if outside_unit_cell:
                raise ValueError("Some positions in the centered and padded clusters are outside of the unit cell. "
                                 "Random rotations cannot be applied as they could cause some colloids to be outside the box. "
                                 "Increase the cluster padding factor to allow for rotations.")

            # Generate uniformly randomized rotations for each cluster and apply them to the unwrapped positions of the clusters.
            # See Properties section here: https://en.wikipedia.org/wiki/Rotation_matrix
            random_rotation_matrices = np.random.rand(r[0], r[1], r[2], 3, 3) # Shape: (r[0], r[1], r[2], 3, 3)
            u, _, vh = np.linalg.svd(random_rotation_matrices)
            random_rotation_matrices = u @ vh # Shape: (r[0], r[1], r[2], 3, 3)
            positions = positions @ random_rotation_matrices # Shape: (r[0], r[1], r[2], max_cluster_size, 3)

        positions += position_displacements[:, :, :, np.newaxis, :] # Shape: (r[0], r[1], r[2], max_cluster_size, 3)
        
        # Reshape the positions and types to be a list of positions and types for all clusters in all displacements. 
        # Remove any positions that are NaN (i.e., padding values).
        final_positions = positions.reshape(-1, 3) # Shape: (r[0] * r[1] * r[2] * max_cluster_size, 3)
        final_positions = final_positions[~np.any(np.isnan(final_positions), axis=1)]
        final_type_ids = type_ids[~np.isnan(type_ids)].astype(int)

        # Set the cell of the repeated cluster.
        full_cell = individual_cell * r

        # Shift the positions so that the center of the box is at the origin.
        final_positions -= (np.max(final_positions, axis=0) - np.min(final_positions, axis=0)) / 2

        frame = Frame()
        frame.particles.N = len(final_positions)
        frame.particles.types = final_types_list

        frame.particles.typeid = np.array(final_type_ids, dtype=np.uint32)
        frame.particles.position = final_positions.astype(np.float32)
        # See http://docs.openmm.org/7.6.0/userguide/theory/05_other_features.html
        # See https://hoomd-blue.readthedocs.io/en/v2.9.3/box.html
        frame.configuration.box = np.array(
            [full_cell[0][0], full_cell[1][1], full_cell[2][2],
             full_cell[1][0] / full_cell[1][1],
             full_cell[2][0] / full_cell[2][2],
             full_cell[2][1] / full_cell[2][2]], dtype=np.float32)

        if bond_values_padded.size > 0:
            # Assign the positions and types of the selected clusters to the corresponding displacements. The positions and bonds of the
            # clusters are padded to the same size so that they can be indexed by the cluster selections. Offsets must be applied to positions and
            # bond pairs to account for the displacements of the clusters. The bond values and types do not need to be modified as they are just 
            # the bond lengths in the original cluster definition.
            bond_pairs = bond_pairs_padded[cluster_selections] # Shape: (r[0], r[1], r[2], max_n_bonds, 2)
            bond_values = bond_values_padded[cluster_selections] # Shape: (r[0], r[1], r[2], max_n_bonds)
            bond_pair_offset_values = bond_pair_offset_values[cluster_selections] # Shape: (r[0], r[1], r[2])
            bond_pair_offset_indices = bond_pair_offset_indices_padded[cluster_selections] # Shape: (r[0], r[1], r[2], max_n_bonds)
        
            # Reshape the bonds to be a list of positions and types for all clusters in all displacements. 
            # Remove any positions that are NaN (i.e., padding values).
            final_bond_values = bond_values.reshape(-1) # Shape: (r[0] * r[1] * r[2] * max_n_bonds,)
            final_bond_values = final_bond_values[~np.isnan(final_bond_values)]

            # The offset that should be applied to the bond pairs associated with each cluster is given by the cumulative 
            # sum of the number of particles in the previous clusters. 
            bond_pair_offset_values = np.cumsum(bond_pair_offset_values.reshape(-1)) - max_cluster_size # Shape: (r[0] * r[1] * r[2],)

            # Get the indices for the bond pair offset indices
            bond_pair_offset_indices = (bond_pair_offset_indices * np.arange(r[0] * r[1] * r[2]).reshape(r)[:, :, :, np.newaxis]).reshape(-1) # Shape: (r[0] * r[1] * r[2] * max_n_bonds)
            bond_pair_offset_indices = bond_pair_offset_indices[~np.isnan(bond_pair_offset_indices)].astype(int)

            bond_pair_offsets = bond_pair_offset_values[bond_pair_offset_indices] # Shape: (n_bonds,)

            # Reshape the bond pairs to be a list of bond pairs for all clusters in all displacements. 
            # Remove any bond pairs that are NaN (i.e., padding values).
            # Apply the offsets to the bond pairs to get the correct indices for the repeated clusters. 
            bond_pairs = bond_pairs.reshape(-1, 2) # Shape: (r[0] * r[1] * r[2] * max_n_bonds, 2)
            bond_pairs = bond_pairs[~np.any(np.isnan(bond_pairs), axis=1)].astype(int) 

            final_bond_pairs = bond_pairs + bond_pair_offsets[:, np.newaxis] # Shape: (n_bonds, 2)

            all_constraints = np.array(final_bond_pairs, dtype=np.uint32)
            all_values = np.array(final_bond_values, dtype=np.float32)
            frame.constraints.N = len(final_bond_pairs)
            frame.constraints.group = all_constraints
            frame.constraints.value = all_values

            # Useful for visualization although not necessary for the simulation.
            frame.bonds.N = len(final_bond_pairs)
            frame.bonds.types = ["b"]
            frame.bonds.typeid = np.zeros(frame.bonds.N, dtype=np.uint32)
            frame.bonds.group = all_constraints

        return frame
