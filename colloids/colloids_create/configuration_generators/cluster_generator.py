from math import acos, pi
from random import choices, uniform
import re
from typing import Sequence, Union
import warnings
from ase.io.lammpsdata import read_lammps_data
from gsd.hoomd import Frame
import numpy as np
from openmm import unit
from .abstracts import ConfigurationGenerator


class ClusterGenerator(ConfigurationGenerator):
    """
    Generator for an initial configuration in a gsd.hoomd.Frame instance for a colloid simulation based on several
    clusters of colloids.

    Each cluster is defined in a lammps-data file together with cell vectors. Every cluster of colloids is assumed to
    have the same cell vectors. To generate the initial configuration, the clusters are first centered. Then, the shared
    cell vectors of the clusters are repeated in all three directions. Every replica of the cell is then filled with a
    randomly selected cluster from the list of clusters. The clusters are selected based on their relative weights.
    Every cluster can optionally be randomly rotated.

    The clusters are specified as file paths to lammps-data files.
    See https://docs.lammps.org/2001/data_format.html for information about this file format.

    All colloid positions in the centered clusters must lie in the unit cell defined by the lattice vectors.

    To space out the clusters, one can increase a cluster padding factor that scales the lattice vectors before
    replication. This will also scale the box size. Additionally, one can increase a padding factor that scales just the
    overall box size and thus increases the distance between the outwards facing colloids and the walls. To make the
    simulation box smaller, use a padding factor less than 1.

    Any bonds in the cluster definition in the lammps-data file are added as constraints, with the constraint distance
    equal to the current bond length in the cluster definition. The bond lengths are not modified during the simulation.

    This dataclass assumes that the style of units in the lammps-data file is "nano" (see
    https://docs.lammps.org/units.html), that is, positions are in nanometers.

    In the lammps-data file, only the lattice vectors, the positions of the colloids in the Atoms section, and the bonds
    in the Bonds section are used. All other sections and information are ignored. In particular, the masses, radii, and
    surface potentials of the different types of colloidal particles appearing in the lammps-data file should be
    specified in the masses, radii, and surface_potentials dictionaries in the yaml file of this data class (and, for
    instance, not in the Masses section of the lammps-data file).

    See https://docs.lammps.org/Howto_triclinic.html for more information about the lattice vectors in the lammps-data
    files.

    In the Atoms section of the lammps-data files, the different columns from left to right are as follows:
        Atom Index (should go from 1 to number of atoms).
        Molecule-ID (ignored)
        Atom type (these are the types appearing as keys in the mass/diameter/surface potential dictionaries in the yaml file)
        Charge (ignored)
        x position
        y position
        z position

    In the Bonds section of the lammps-data files, the different columns from left to right are as follows:
        Bond index (should go from 1 to number of bonds)
        Bond ID (ignored)
        Index of first atom involved in the bond.
        Index of second atom involved in the bond.

    :param masses:
        The masses dictionary with the particle types as keys and the masses as values.
    :type masses: dict[str, unit.Quantity]
    :param radii:
        The radii dictionary with the particle types as keys and the radii as values.
    :type radii: dict[str, unit.Quantity]
    :param surface_potentials:
        The surface potentials dictionary with the particle types as keys and the surface potentials as values.
    :type surface_potentials: dict[str, unit.Quantity]
    :param cluster_specifications:
        The filenames of the cluster definitions in lammps-data format.
        All clusters must have the same cell vectors.
        All (centered) clusters should lie in the unit cell defined by the cell vectors.
    :type cluster_specifications: list[str]
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
        If any cluster specification file does not end with ".lmp".
        If no cluster specification file is provided.
        If the length of the cluster specifications does not match the length of the cluster relative weights.
        If any cluster relative weight is negative.
        If the cluster padding factor is not greater than zero.
        If the padding factor is not greater than zero.
        If no clusters are provided.
        If the number of clusters does not match the number of cluster probabilities.
        If any cluster probability is not greater than zero.
        If the clusters do not have the same cell vectors.
        If the lattice repeats is not a positive integer or a list of three positive integers.
    """

    def __init__(self, masses: dict[str, unit.Quantity], radii: dict[str, unit.Quantity],
                 surface_potentials: dict[str, unit.Quantity], cluster_specifications: list[str],
                 cluster_relative_weights: Sequence[float], lattice_repeats: Union[int, Sequence[int]],
                 cluster_padding_factor: float, padding_factor: float, random_rotation: bool) -> None:
        """Constructor of the ClusterGenerator class."""
        super().__init__(masses=masses, radii=radii, surface_potentials=surface_potentials)
        if not all(cluster_specification.endswith(".lmp") for cluster_specification in cluster_specifications):
            raise ValueError("The cluster specification file must be of the lammps-data file format.")
        if not len(cluster_specifications) > 0:
            raise ValueError("At least one cluster must be provided.")
        if len(cluster_specifications) != len(cluster_relative_weights):
            raise ValueError("The number of clusters must match the number of cluster probabilities.")
        if not all(prob >= 0.0 for prob in cluster_relative_weights):
            raise ValueError("All cluster probabilities must be non-negative.")
        if any(prob == 0.0 for prob in cluster_relative_weights):
            warnings.warn("Some cluster probabilities are zero. These clusters will not be used in the initial "
                          "configuration.")
        # We assume that the lammps-data file uses "nano" units where distances are measured in nanometers.
        # However, ase would transform the distances in the lammps-data file to Angstroms by multiplying them
        # by 10 if we specify units="nano". For units="metal", the ase distances are equal to the distances in
        # the lammps-data file. We then just pretend that the distances are in nanometers.
        self._clusters = [read_lammps_data(spec, units="metal") for spec in cluster_specifications]
        self._cluster_relative_weights = cluster_relative_weights
        self._lattice_repeats = lattice_repeats
        self._cluster_padding_factor = cluster_padding_factor
        self._padding_factor = padding_factor
        self._random_rotation = random_rotation
        if self._cluster_padding_factor <= 0.0:
            raise ValueError("The cluster padding factor must be greater than zero.")
        if self._padding_factor <= 0.0:
            raise ValueError("The padding factor must be greater than zero.")
        if not len(self._clusters) > 0:
            raise ValueError("At least one cluster must be provided.")
        if len(self._clusters) != len(cluster_relative_weights):
            raise ValueError("The number of clusters must match the number of cluster probabilities.")
        if not all(prob >= 0.0 for prob in cluster_relative_weights):
            raise ValueError("All cluster probabilities must be non-negative.")
        if not all(np.allclose(c.cell, self._clusters[0].cell) for c in self._clusters):
            raise ValueError("All clusters must have the same cell vectors.")
        if isinstance(self._lattice_repeats, int):
            if self._lattice_repeats <= 0:
                raise ValueError("The number of lattice repeats must be positive.")
        else:
            if not (isinstance(self._lattice_repeats, list)
                    and all(isinstance(repeat, int) for repeat in self._lattice_repeats)
                    and len(self._lattice_repeats) == 3):
                raise TypeError("The lattice repeats must be an integer or a list of three integers.")
            if not all(repeat > 0 for repeat in self._lattice_repeats):
                raise ValueError("All lattice repeats must be positive.")

        self._found_types = set()
        for atoms in self._clusters:
            # If no cell is set in the lammps-data file, the lattice vectors are set to the identity matrix.
            if np.equal(atoms.get_cell(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).all():
                warnings.warn(
                    "The lattice vectors of the cluster are probably not set in the lammps-data file."
                    "The identity matrix is used as lattice vectors.")
            types = set(str(atom.number) for atom in atoms)
            self._found_types.update(types)

    def types(self) -> set[str]:
        """
        Return the set of particle types that will be generated by this configuration generator.

        :return:
            The set of particle types that will be generated by this configuration generator.
        :rtype: set[str]
        """
        return self._found_types

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

        repeated_cluster = None
        bond_pairs = []
        bond_values = []
        # All cells are assumed to be the same so we can use the first one.
        cell = centered_padded_clusters[0].get_cell()
        # Adapted from ase's repeat function.
        i0 = 0
        for r0 in range(r[0]):
            for r1 in range(r[1]):
                for r2 in range(r[2]):
                    new_cluster = choices(centered_padded_clusters, weights=self._cluster_relative_weights,
                                          k=1)[0].copy()
                    if self._random_rotation:
                        # Generate uniformly randomized rotations.
                        # See Properties section here: https://en.wikipedia.org/wiki/Euler_angles
                        random_phi_deg = uniform(0.0, 2.0 * pi) * 180.0 / pi  # alpha on Wikipedia.
                        random_theta_deg = acos(uniform(-1.0, 1.0)) * 180.0 / pi  # beta on Wikipedia.
                        random_psi_deg = uniform(0.0, 2.0 * pi) * 180.0 / pi  # gamma on Wikipedia.
                        new_cluster.euler_rotate(phi=random_phi_deg, theta=random_theta_deg, psi=random_psi_deg,
                                                    center="COP")
                    unwrapped_positions = new_cluster.get_positions(wrap=False)
                    if self._random_rotation:
                        wrapped_positions = new_cluster.get_positions(wrap=True)
                        if not np.allclose(wrapped_positions, unwrapped_positions):
                            raise ValueError("Parts of the rotated cluster lie outside the original box, increase the "
                                             "rotation padding distance to allow for rotations.")
                    # Translate the unwrapped positions of the new cluster to the correct repeat.
                    unwrapped_positions += np.dot((r0, r1, r2), cell)
                    if repeated_cluster is None:
                        # If this is the first cluster, we create the repeated cluster.
                        assert r0 == r1 == r2 == i0 == 0
                        repeated_cluster = new_cluster.copy()
                        if "bonds" in repeated_cluster.arrays:
                            # Extract the bonds and adjust the indices to account for the repetitions.
                            new_bond_pairs = [(first_index, second_index)
                                              for first_index, bonds in enumerate(new_cluster.arrays["bonds"])
                                              for second_index in self._extract_bonded_indices(bonds)]
                            bond_pairs += new_bond_pairs
                            bond_values += [(new_cluster.get_distance(first_index, second_index))
                                            for first_index, second_index in new_bond_pairs]
                    else:
                        # Concatenate the arrays of the new cluster to the repeated cluster.
                        for name, a in new_cluster.arrays.items():
                            if name != "bonds":
                                repeated_cluster.arrays[name] = np.concatenate((repeated_cluster.arrays[name], a),
                                                                               axis=0)
                            else:
                                # For the bonds, we need to adjust the indices to account for the repetitions.
                                new_base_bond_pairs = [(first_index, second_index)
                                                  for first_index, bonds in enumerate(new_cluster.arrays["bonds"])
                                                  for second_index in self._extract_bonded_indices(bonds)]
                                new_bond_pairs = [(first_index + i0, second_index + i0)
                                                  for first_index, second_index in new_base_bond_pairs]
                                bond_pairs += new_bond_pairs
                                bond_values += [repeated_cluster.get_distance(first_index, second_index)
                                                for first_index, second_index in new_base_bond_pairs]
                        # Set the positions of the repeated cluster to the unwrapped positions.
                        repeated_cluster.arrays["positions"][i0:i0 + len(new_cluster)] = unwrapped_positions
                    i0 += len(new_cluster)

        # Enlarge the cell of the repeated cluster.
        repeated_cluster.set_cell(np.array([self._padding_factor * r[c] * cell[c] for c in range(3)]) )
        repeated_cluster.center(about=(0.0, 0.0, 0.0))

        frame = Frame()
        frame.particles.N = len(repeated_cluster)
        frame.particles.types = tuple(dict.fromkeys(str(atom.number) for atom in repeated_cluster)) 
        frame.particles.typeid = np.array([frame.particles.types.index(str(atom.number))
                                           for atom in repeated_cluster], dtype=np.uint32)
        frame.particles.position = repeated_cluster.positions.astype(np.float32)
        # See http://docs.openmm.org/7.6.0/userguide/theory/05_other_features.html
        # See https://hoomd-blue.readthedocs.io/en/v2.9.3/box.html
        frame.configuration.box = np.array(
            [repeated_cluster.cell[0][0], repeated_cluster.cell[1][1], repeated_cluster.cell[2][2],
             repeated_cluster.cell[1][0] / repeated_cluster.cell[1][1],
             repeated_cluster.cell[2][0] / repeated_cluster.cell[2][2],
             repeated_cluster.cell[2][1] / repeated_cluster.cell[2][2]], dtype=np.float32)

        if len(bond_pairs) > 0:
            all_constraints = np.array(bond_pairs, dtype=np.uint32)
            all_values = np.array(bond_values, dtype=np.float32)
            frame.constraints.N = len(bond_pairs)
            frame.constraints.group = all_constraints
            frame.constraints.value = all_values

            # Useful for visualization although not necessary for the simulation.
            frame.bonds.N = len(bond_pairs)
            frame.bonds.types = ["b"]
            frame.bonds.typeid = np.zeros(frame.bonds.N, dtype=np.uint32)
            frame.bonds.group = all_constraints

        return frame
