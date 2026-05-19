from typing import Optional
import freud
import numpy as np
import gsd.hoomd
from gsd.hoomd import Frame
from openmm import unit
from colloids.colloids_create import FinalModifier
from colloids.units import length_unit


class SeedModifier(FinalModifier):
    """
    Modifier of an existing configuration in a Frame instance that seeds the configuration with particles from another
    frame.

    Particles in the base frame that overlap with the seed are removed. If such a particle is part of a constraint
    (bond), all particles in that constraint group are removed together to avoid breaking bonds.

    The overlap is determined based on the surface-to-surface distance between particles, taking into account their
    diameters. Particles are considered overlapping if their surface-to-surface distance is less than the specified
    overlap distance. Note that the surface-to-surface distance is calculated without applying periodic boundary
    conditions.

    The seed positions are taken directly from the seed frame without any transformation. The base frame's box must be
    at least as large as the seed frame's box in all dimensions. Additionally, both boxes must be orthorhombic.

    Optionally, one can filter the seed frame to keep only the largest cluster of particles before seeding. For this,
    a cutoff distance must be provided. Two particles are considered neighbors if their distance is less than this
    cutoff distance. The largest cluster is determined based on these neighbor relationships. If any particle in a bond
    belongs to the largest cluster, all particles in that bond are included in the largest cluster.

    This modifier requires that the frame already has diameter, charge, and mass attributes set. The diameters are
    needed for overlap detection.

    :param seed_filename:
        Path to the GSD file containing the seed configuration.
    :type seed_filename: str
    :param seed_frame_index:
        The frame index in the seed file to use. Negative indices are supported (e.g., -1 for last frame).
        Defaults to -1.
    :type seed_frame_index: int
    :param overlap_distance:
        The overlap tolerance for determining which particles to remove. Particles are considered overlapping if their
        surface-to-surface distance is less than this value.
        Must have units compatible with nanometers and be non-negative.
        Defaults to 0.0 nm.
    :type overlap_distance: unit.Quantity
    :param cluster_cutoff_distance:
        Optional maximum neighbor distance for clustering in the seed frame. If provided, only the largest cluster
        of particles in the seed frame will be used for seeding.
        Must have units compatible with nanometers.
        Must be positive if provided.
        Default is None, meaning no filtering of the seed frame.
    :type cluster_cutoff_distance: Optional[unit.Quantity]
    :raises TypeError:
        If overlap_distance does not have units compatible with nanometers.
    :raises ValueError:
        If overlap_distance is negative.
    """

    def __init__(self, seed_filename: str, seed_frame_index: int = -1,
                 seed_fractional_position: Optional[list[float]] = [0.5, 0.5, 0.5],
                 overlap_distance: unit.Quantity = 0.0 * length_unit,
                 cluster_cutoff_distance: Optional[unit.Quantity] = None) -> None:
        """Constructor of the SeedModifier class."""
        super().__init__()
        if not seed_filename.endswith(".gsd"):
            raise ValueError("The seed filename must end with .gsd")
        if len(seed_fractional_position) != 3:
            raise ValueError("The seed fractional position must be a 3-dimensional vector.")
        if not np.all(np.array([n >= 0.0 and n <= 1.0 for n in seed_fractional_position])):
            raise ValueError("The seed fractional position must be between 0 and 1 in all dimensions.")
        if not overlap_distance.unit.is_compatible(length_unit):
            raise TypeError("The overlap distance must have a unit compatible with nanometers.")
        if overlap_distance < 0.0 * length_unit:
            raise ValueError("The overlap distance must be non-negative.")
        if cluster_cutoff_distance is not None:
            if not cluster_cutoff_distance.unit.is_compatible(length_unit):
                raise TypeError("The filter_cluster_r_max distance must have a unit compatible with nanometers.")
            if cluster_cutoff_distance <= 0.0 * length_unit:
                raise ValueError("The filter_cluster_r_max distance must be positive.")
        self._overlap_distance = overlap_distance.value_in_unit(length_unit)
        self._seed_filename = seed_filename
        self._seed_frame_index = seed_frame_index
        self._cluster_cutoff_distance = (cluster_cutoff_distance.value_in_unit(length_unit)
                                         if cluster_cutoff_distance is not None else None)
        self._seed_fractional_position = seed_fractional_position
    @staticmethod
    def _validate_frame_compatibility(frame: Frame, seed_frame: Frame) -> None:
        """
        Validate that the seed frame is compatible with the base frame.

        If both frames share a type of the same name, this function makes sure that their diameter, mass, and charge
        are compatible

        :param frame:
            The base frame.
        :type frame: Frame
        :param seed_frame:
            The seed frame.
        :type seed_frame: Frame

        :raises ValueError:
            If a particle in the seed frame has different mass, diameter, or charge than the corresponding type in the
            base frame.
        """
        box = frame.configuration.box
        seed_box = seed_frame.configuration.box

        # Check that both boxes are orthorhombic.
        if not np.allclose(seed_box[3:], 0.0):
            raise ValueError(f"The seed frame's box must be orthorhombic (no tilt factors), "
                             f"but has tilt factors {seed_box[3:]}.")
        if not np.allclose(box[3:], 0.0):
            raise ValueError(f"The base frame's box must be orthorhombic (no tilt factors), "
                             f"but has tilt factors {box[3:]}.")

        # Check that the base frame's box is at least as large as the seed frame's box
        if np.any(box[:3] < seed_box[:3]):
            raise ValueError(f"The base frame's box {box[:3]} must be at least as large as "
                             f"the seed frame's box {seed_box[:3]} in all dimensions.")

        # Check compatibility of masses, charges, and diameter.
        for t in seed_frame.particles.types:
            if t in frame.particles.types:
                # Find index of type in types tuple.
                t_index = frame.particles.types.index(t)
                t_index_seed = seed_frame.particles.types.index(t)
                mask = (frame.particles.typeid == t_index)
                mask_seed = (seed_frame.particles.typeid == t_index_seed)

                # No actual particles found, no need to check compatibility.
                if not np.any(mask) or not np.any(mask_seed):
                    continue

                diameter = frame.particles.diameter[mask]
                diameter_seed = seed_frame.particles.diameter[mask_seed]
                assert len(diameter) > 0
                assert len(diameter_seed) > 0
                if not np.allclose(diameter, diameter[0]):
                    raise ValueError(f"The base frame contains particles with the same type {t} but different "
                                     f"diameter.")
                if not np.allclose(diameter_seed, diameter_seed[0]):
                    raise ValueError(f"The seed frame contains particles with the same type {t} but different diameter.")
                if not np.isclose(diameter[0], diameter_seed[0]):
                    raise ValueError(f"The diameter of type {t} is {diameter[0]} in the base frame but "
                                     f"{diameter_seed[0]} in the seed frame.")

                masses = frame.particles.mass[mask]
                masses_seed = seed_frame.particles.mass[mask_seed]
                assert len(masses) > 0
                assert len(masses_seed) > 0
                if not np.allclose(masses, masses[0]):
                    raise ValueError(f"The base frame contains particles with the same type {t} but different masses.")
                if not np.allclose(masses_seed, masses_seed[0]):
                    raise ValueError(f"The seed frame contains particles with the same type {t} but different masses.")
                if not np.isclose(masses[0], masses_seed[0]):
                    raise ValueError(f"The mass of type {t} is {masses[0]} in the base frame but "
                                     f"{masses_seed[0]} in the seed frame.")

                charges = frame.particles.charge[mask]
                charges_seed = seed_frame.particles.charge[mask_seed]
                assert len(charges) > 0
                assert len(charges_seed) > 0
                if not np.allclose(charges, charges[0]):
                    raise ValueError(f"The base frame contains particles with the same type {t} but different charges.")
                if not np.allclose(charges_seed, charges_seed[0]):
                    raise ValueError(f"The seed frame contains particles with the same type {t} but different charges.")
                if not np.isclose(charges[0], charges_seed[0]):
                    raise ValueError(f"The charge of type {t} is {charges[0]} in the base frame but "
                                     f"{charges_seed[0]} in the seed frame.")

    @staticmethod
    def _filter_largest_cluster(frame: Frame, cutoff_distance: float) -> None:
        """
        Modify the given frame in-place to keep only the largest cluster of particles based on the given cutoff
        distance.

        If any particle in a bond belongs to the largest cluster, all particles in that bond are included in the largest
        cluster.

        :param frame:
            The frame to modify.
        :type frame: Frame
        :param cutoff_distance:
            The maximum neighbor distance for clustering.
        :type cutoff_distance: float
        """
        positions = frame.particles.position
        # Freud box does not matter without periodic boundaries.
        freud_box = freud.box.Box(Lx=1.0, Ly=1.0, Lz=1.0)
        freud_box.periodic = False
        cluster = freud.cluster.Cluster()
        cluster.compute((freud_box, positions), neighbors={"r_max": cutoff_distance, "exclude_ii": True})
        cluster_ids = cluster.cluster_idx.copy()
        largest_cluster_id = np.bincount(cluster_ids).argmax()

        # Make sure that both particles in bonds are in the largest cluster.
        if frame.constraints.N > 0:
            for i, j in frame.constraints.group:
                if cluster_ids[i] == largest_cluster_id or cluster_ids[j] == largest_cluster_id:
                    cluster_ids[i] = largest_cluster_id
                    cluster_ids[j] = largest_cluster_id

        largest_cluster_mask = (cluster_ids == largest_cluster_id)
        filtered_indices = np.nonzero(largest_cluster_mask)[0]

        frame.particles.N = len(filtered_indices)
        frame.particles.typeid = frame.particles.typeid[filtered_indices]
        frame.particles.mass = frame.particles.mass[filtered_indices]
        frame.particles.charge = frame.particles.charge[filtered_indices]
        frame.particles.diameter = frame.particles.diameter[filtered_indices]
        frame.particles.position = frame.particles.position[filtered_indices]

        if frame.constraints.N == 0:
            return

        index_map = {old_index: new_index for new_index, old_index in enumerate(filtered_indices)}
        new_values = []
        new_groups = []
        for value, group in zip(frame.constraints.value, frame.constraints.group):
            i, j = group
            if largest_cluster_mask[i]:
                assert largest_cluster_mask[j]
                new_values.append(value)
                new_groups.append([index_map[i], index_map[j]])
            else:
                assert not largest_cluster_mask[j]
        frame.constraints.N = len(new_values)
        frame.constraints.value = np.array(new_values, dtype=np.float32)
        frame.constraints.group = np.array(new_groups, dtype=np.uint32)

        frame.bonds.N = len(new_values)
        frame.bonds.types = ["b"]
        frame.bonds.typeid = np.zeros(frame.bonds.N, dtype=np.uint32)
        frame.bonds.group = frame.constraints.group.copy()

    def _reposition_seed_frame(self, frame: Frame, seed_frame: Frame, seed_fractional_position: np.ndarray) -> None:
        """
        Modify the seed frame in-place to reposition its center of mass to the given position.

        :param frame:
            The base frame.
        :type frame: Frame
        :param seed_frame:
            The seed frame to modify.
        :type seed_frame: Frame
        :param seed_fractional_position:
            The new fractional position of the center of mass of the seed frame.
        :type seed_fractional_position: np.ndarray
        """
        positions = seed_frame.particles.position
        masses = seed_frame.particles.mass
        center_of_mass = np.average(positions, axis=0, weights=masses)
        seed_position = (np.array(seed_fractional_position) - 0.5) * frame.configuration.box[:3]
        shift = seed_position - center_of_mass
        seed_frame.particles.position += shift

    @staticmethod
    def _find_overlapping_particles(frame: Frame, seed_frame: Frame, overlap_distance: float) -> set[int]:
        """
        Return the indices of the particles in the first frame that overlap with any particle in the seed frame.

        :param frame:
            The first frame.
        :type frame: Frame
        :param seed_frame:
            The seed frame.
            The positions of seed particles (already centered in the base box).
        :type seed_frame: Frame
        :param overlap_distance:
            Distance between the surface of two particles below which they are considered overlapping.
            Should be bigger than zero.
        :type overlap_distance: float

        :return:
            Indices of particles in the first frame that overlap with any particle in the seed frame.
        :rtype: set[int]
        """
        positions = frame.particles.position
        seed_positions = seed_frame.particles.position
        radii = frame.particles.diameter / 2.0
        seed_radii = seed_frame.particles.diameter / 2.0

        # Broadcasting (N, 1, 3) - (1, N_s, 3) leads to shape (N, N_s, 3).
        # Taking norm along axis 2 leads to shape (N, N_s).
        # First index is index in positions, second index is second position.
        distance_matrix = np.linalg.norm(positions[:, None, :] - seed_positions[None, :, :], axis=2)
        # Broadcasting (N, N_s) - (N, 1) - (1, N_s) leads to (N, N_s)
        overlap_matrix = (distance_matrix - radii[:, None] - seed_radii[None, :]) < overlap_distance
        # Find all positions that are overlapping with any seed position (shape (N,)).
        positions_overlapping = np.any(overlap_matrix, axis=1)
        overlapping_indices = set(np.nonzero(positions_overlapping)[0])

        assert frame.constraints.N == frame.bonds.N
        if frame.constraints.N == 0:
            return overlapping_indices

        # Store the parents of each particle in a disjoint-set data structure.
        # See https://en.wikipedia.org/wiki/Disjoint-set_data_structure.
        # Each node that is not a root in the tree points to its parents.
        # This data structure can be used to find the particles that are connected to any particle with overlaps.
        parents = list(range(frame.particles.N))

        def find(i):
            """Find the root node of particle i by following each parent pointer in succession."""
            root = i
            while parents[root] != root:
                root = parents[root]
            # Update all nodes along the path to point directly to the root (path compression).
            j = i
            while j != root:
                parents[j], j = root, parents[j]
            return root

        def union(i, j):
            """Take the union of particles i and j by making their parents equal."""
            i = find(i)
            j = find(j)
            if i != j:
                parents[i] = j

        for pair in frame.constraints.group:
            union(pair[0], pair[1])

        groups_to_remove = set(find(i) for i in overlapping_indices)
        particles_to_remove = set()
        for i in range(frame.particles.N):
            if find(i) in groups_to_remove:
                particles_to_remove.add(i)

        return particles_to_remove

    @staticmethod
    def _remove_indices(frame: Frame, remove_indices: set[int]) -> None:
        """
        Remove the given particles from the frame in-place.

        :param frame:
            The frame to modify.
        :type frame: gsd.hoomd.Frame
        :param remove_indices:
            The particle indices to remove.
        :type remove_indices: set[int]
        """
        mask = [i not in remove_indices for i in range(frame.particles.N)]

        frame.particles.N -= len(remove_indices)
        frame.particles.typeid = frame.particles.typeid[mask]
        frame.particles.mass = frame.particles.mass[mask]
        frame.particles.charge = frame.particles.charge[mask]
        frame.particles.diameter = frame.particles.diameter[mask]
        frame.particles.position = frame.particles.position[mask]

        if frame.constraints.N == 0:
            return

        filtered_indices = np.nonzero(mask)[0]
        index_map = {old_index: new_index for new_index, old_index in enumerate(filtered_indices)}
        new_values = []
        new_groups = []
        for value, group in zip(frame.constraints.value, frame.constraints.group):
            i, j = group
            if i not in remove_indices:
                assert j not in remove_indices
                new_values.append(value)
                new_groups.append([index_map[i], index_map[j]])
            else:
                assert j in remove_indices
        frame.constraints.N = len(new_values)
        frame.constraints.value = np.array(new_values, dtype=np.float32)
        frame.constraints.group = np.array(new_groups, dtype=np.uint32)

        frame.bonds.N = len(new_values)
        frame.bonds.types = ["b"]
        frame.bonds.typeid = np.zeros(frame.bonds.N, dtype=np.uint32)
        frame.bonds.group = frame.constraints.group.copy()

    @staticmethod
    def combine_frames(frame: Frame, seed_frame: Frame) -> None:
        """
        Add the particles and bonds from the seed frame to the first frame.

        :param frame:
            The frame to modify.
        :type frame: Frame
        :param seed_frame:
            The seed frame.
        :type seed_frame: Frame
        """
        original_n = frame.particles.N
        frame.particles.N  += seed_frame.particles.N
        for t in seed_frame.particles.types:
            if t not in frame.particles.types:
                frame.particles.types += (t, )
        typeid_map = {seed_typeid: frame.particles.types.index(seed_type)
                      for seed_typeid, seed_type in enumerate(seed_frame.particles.types)}
        new_typeids = np.array([typeid_map[seed_typeid] for seed_typeid in seed_frame.particles.typeid],
                               dtype=np.uint32)
        frame.particles.typeid = np.concatenate((frame.particles.typeid, new_typeids))
        frame.particles.mass = np.concatenate((frame.particles.mass, seed_frame.particles.mass))
        frame.particles.charge = np.concatenate((frame.particles.charge, seed_frame.particles.charge))
        frame.particles.diameter = np.concatenate((frame.particles.diameter, seed_frame.particles.diameter))
        frame.particles.position = np.vstack((frame.particles.position, seed_frame.particles.position))

        if frame.constraints.N == 0 and seed_frame.constraints.N == 0:
            return

        frame_values = frame.constraints.value if frame.constraints.N > 0 else np.array([], dtype=np.float32)
        frame_groups = frame.constraints.group if frame.constraints.N > 0 else np.array([], dtype=np.uint32)
        seed_values = seed_frame.constraints.value if seed_frame.constraints.N > 0 else np.array([], dtype=np.float32)
        seed_groups = (
            np.array([[i + original_n, j + original_n] for i, j in seed_frame.constraints.group],
                     dtype=np.uint32) if seed_frame.constraints.N > 0 else np.empty((0, 2), dtype=np.uint32)
        )

        frame.constraints.N += seed_frame.constraints.N
        frame.constraints.value = np.concatenate((frame_values, seed_values))
        frame.constraints.group = np.vstack((frame_groups, seed_groups))

        frame.bonds.N = frame.constraints.N
        frame.bonds.types = ["b"]
        frame.bonds.typeid = np.zeros(frame.bonds.N, dtype=np.uint32)
        frame.bonds.group = frame.constraints.group.copy()

    def modify_configuration(self, frame: Frame) -> None:
        """
        Modify the given configuration in-place by seeding it with particles from the seed frame.

        Particles in the base frame that overlap with the seed are removed. If such a particle is part of a constraint
        (bond), all particles in that constraint group are removed together.

        The seed positions are taken directly from the seed frame without any transformation.

        This method modifies the following attributes of the given frame:
        - frame.particles.N
        - frame.particles.position
        - frame.particles.types
        - frame.particles.typeid
        - frame.particles.mass
        - frame.particles.charge
        - frame.particles.diameter
        - frame.constraints.N (optionally if constraints are present)
        - frame.constraints.value (optionally if constraints are present)
        - frame.constraints.group (optionally if constraints are present)
        - frame.bonds.N (optionally if constraints are present)
        - frame.bonds.types (optionally if constraints are present)
        - frame.bonds.typeid (optionally if constraints are present)
        - frame.bonds.group (optionally if constraints are present)

        :param frame:
            The frame to modify. Must have diameter attribute set for overlap detection.
        :type frame: gsd.hoomd.Frame

        :raises ValueError:
            If the frame does not have the diameter attribute set.
            If seed frame types are not compatible with base frame types.
            If the base frame's box is smaller than the seed frame's box in any dimension.
            If either box is triclinic (has non-zero tilt factors).
        """
        if frame.particles.diameter is None:
            raise ValueError(f"The SeedModifier class requires already populated diameter.")
        if frame.particles.charge is None:
            raise ValueError(f"The SeedModifier class requires already populated charges.")
        if frame.particles.mass is None:
            raise ValueError(f"The SeedModifier class requires already populated masses.")

        with gsd.hoomd.open(self._seed_filename, mode='r') as traj:
            seed_frame = traj[self._seed_frame_index]

        if not seed_frame.particles.N > 0:
            raise ValueError("The seed frame does not contain any particles.")

        self._validate_frame_compatibility(frame, seed_frame)

        if self._cluster_cutoff_distance is not None:
            self._filter_largest_cluster(seed_frame, self._cluster_cutoff_distance)

        if self._seed_fractional_position is not None:
            self._reposition_seed_frame(frame, seed_frame, self._seed_fractional_position)

        overlapping_indices = self._find_overlapping_particles(frame, seed_frame, self._overlap_distance)
        if len(overlapping_indices) > 0:
            self._remove_indices(frame, overlapping_indices)

        # Add seed particles
        self.combine_frames(frame, seed_frame)


if __name__ == '__main__':
    f = gsd.hoomd.Frame()
    print(f.constraints.N)
    print(f.constraints.group)