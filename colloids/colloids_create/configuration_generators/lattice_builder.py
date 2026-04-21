from typing import Optional, Sequence, Union
import warnings
from gsd.hoomd import Frame
import numpy as np
import openmm
from openmm import unit
from pymatgen.core import Element
from pymatgen.io.cif import CifParser
from scipy.spatial import distance_matrix
from colloids.colloid_potentials_algebraic import ColloidPotentialsAlgebraic
from colloids.colloid_potentials_parameters import ColloidPotentialsParameters
from colloids.run_parameters import RunParameters
from colloids.units import length_unit, mass_unit, electric_potential_unit, energy_unit
from .abstracts import ConfigurationGenerator


class LatticeBuilder(ConfigurationGenerator):
    """
    Generator for an initial configuration in a gsd.hoomd.Frame instance for a colloid simulation based on a
    crystal lattice structure defined in a CIF file.

    The lattice structure is loaded from a CIF file and expanded into a supercell. The supercell is then uniformly
    scaled so that no particles overlap, accounting for colloid radii, brush length, and an extra radii padding gap.
    The optimal scale factor is computed directly as the maximum ratio of the sum of effective radii to the distance
    for all particle pairs, using a small (3, 3, 3) test supercell for efficiency. This scale factor is then applied to
    the full supercell defined by the lattice repeats.

    Optionally, the scale factor can be further optimized by minimizing the steric + electrostatic energy evaluated in
    vacuum (non-periodic boundary conditions) using OpenMM. A grid of uniformly spaced scale factors around the
    geometric scale factor is evaluated, and the one with the minimum energy is selected.

    The scaled supercell is centered at the origin and embedded in a cubic orthorhombic simulation box. The box side
    length is chosen so that the outermost particle (including its effective radius) plus a lattice padding gap fits
    within the box in every direction.

    :param masses:
        The masses dictionary with the particle types as keys and the masses as values.
    :type masses: dict[str, unit.Quantity]
    :param radii:
        The radii dictionary with the particle types as keys and the radii as values.
    :type radii: dict[str, unit.Quantity]
    :param surface_potentials:
        The surface potentials dictionary with the particle types as keys and the surface potentials as values.
    :type surface_potentials: dict[str, unit.Quantity]
    :param lattice_specification:
        The .cif file that specifies the desired lattice structure.
    :type lattice_specification: str
    :param lattice_repeats:
        Number of repetitions of the unit cell in each direction to create the supercell. This can be specified as a
        single integer (if the same number of repetitions is desired in all directions) or as a sequence of three
        integers (if different numbers of repetitions are desired in different directions).
    :type lattice_repeats: Union[int, Sequence[int]]
    :param run_parameters_file:
        The path to the YAML file containing the run parameters. The brush length used for computing the effective
        radii is read from this file.
    :type run_parameters_file: str
    :param radii_padding:
        Extra gap added to the effective radii when checking for overlaps.
        The unit of the radii padding should be compatible with nanometers and the value must be greater than or equal
        to zero.
    :type radii_padding: unit.Quantity
    :param lattice_padding:
        Extra gap added to the box dimensions.
        The unit of the lattice padding should be compatible with nanometers and the value must be greater than or
        equal to zero.
    :type lattice_padding: unit.Quantity
    :param optimize_energy:
        If True, the geometric scale factor is replaced by the scale factor that minimizes the steric +
        electrostatic energy evaluated in vacuum (non-periodic boundary conditions). The energy is computed for a
        grid of uniformly spaced scale factors around the geometric scale factor and the one with the minimum energy
        is selected.
        Defaults to False.
    :type optimize_energy: bool
    :param energy_scale_range:
        The range of scale factors to evaluate, specified as (min_factor, max_factor) relative to the geometric
        scale factor. For example, (0.5, 1.5) means the scan range is from 50% to 150% of the geometric scale
        factor.
        Only allowed when optimize_energy is True. Defaults to (0.5, 1.5) when not specified.
    :type energy_scale_range: Optional[Sequence[float]]
    :param energy_scale_samples:
        The number of uniformly spaced scale factors to evaluate in the scan range.
        Only allowed when optimize_energy is True. Defaults to 50 when not specified.
    :type energy_scale_samples: Optional[int]

    :raises ValueError:
        If the lattice specification file is not a .cif file.
        If the CIF file does not contain exactly one structure.
        If the lattice repeats is not a positive integer or a sequence of three positive integers.
        If the run parameters file is not a .yaml file.
        If the radii padding is not compatible with nanometers or is negative.
        If the lattice padding is not compatible with nanometers or is negative.
        If energy_scale_range or energy_scale_samples is set when optimize_energy is False.
        If the energy scale range does not consist of exactly two floats satisfying 0 < min < max.
        If the energy scale samples is less than 2.
    :raises RuntimeError:
        If optimize_energy is True and no valid scale factor is found during the energy scan.
    """

    def __init__(self, masses: dict[str, unit.Quantity], radii: dict[str, unit.Quantity],
                 surface_potentials: dict[str, unit.Quantity], lattice_specification: str,
                 lattice_repeats: Union[int, Sequence[int]], run_parameters_file: str,
                 radii_padding: unit.Quantity, lattice_padding: unit.Quantity,
                 optimize_energy: bool = False, energy_scale_range: Optional[Sequence[float]] = None,
                 energy_scale_samples: Optional[int] = None) -> None:
        """Constructor of the LatticeBuilder class."""
        super().__init__(masses=masses, radii=radii, surface_potentials=surface_potentials)
        if not lattice_specification.endswith('.cif'):
            raise ValueError("The lattice specification must be a .cif file.")
        parser = CifParser(lattice_specification, site_tolerance=0.0, frac_tolerance=0.0)
        structures = parser.parse_structures(check_occu=False, primitive=False)
        if len(structures) != 1:
            raise ValueError("The CIF file must contain exactly one structure.")
        self._structure = structures[0]
        self._radii = radii
        self._surface_potentials = surface_potentials
        self._masses = masses
        if not run_parameters_file.endswith('.yaml'):
            raise ValueError("The run parameters file must be a .yaml file.")
        run_parameters = RunParameters.from_yaml(run_parameters_file)
        self._run_parameters = run_parameters
        self._brush_length = run_parameters.brush_length
        self._lattice_repeats = lattice_repeats
        self._radii_padding = radii_padding
        self._lattice_padding = lattice_padding
        self._optimize_energy = optimize_energy
        if not optimize_energy:
            if energy_scale_range is not None:
                raise ValueError("energy_scale_range must not be set when optimize_energy is False.")
            if energy_scale_samples is not None:
                raise ValueError("energy_scale_samples must not be set when optimize_energy is False.")
        self._energy_scale_range = energy_scale_range if energy_scale_range is not None else (0.5, 1.5)
        self._energy_scale_samples = energy_scale_samples if energy_scale_samples is not None else 50
        # Label atoms as their element symbols based on atomic number, e.g. 'Fe', 'O', etc.
        self._type_map = {atomic_number: str(Element.from_Z(atomic_number))
                          for atomic_number in np.unique(self._structure.atomic_numbers)}
        if isinstance(self._lattice_repeats, int):
            if not self._lattice_repeats > 0:
                raise ValueError("The lattice repeats must be greater than zero.")
        else:
            if not isinstance(self._lattice_repeats, Sequence) or len(self._lattice_repeats) != 3:
                raise ValueError("The lattice repeats must be either a single integer or a sequence of three integers.")
            if not all(r > 0 for r in self._lattice_repeats):
                raise ValueError("All values in the lattice repeats must be greater than zero.")
        if not self._radii_padding.unit.is_compatible(unit.nanometer):
            raise TypeError("The radii padding must have a unit that is compatible with nanometers.")
        if not self._radii_padding.value_in_unit(unit.nanometer) >= 0.0:
            raise ValueError("The radii padding must have a value greater than or equal to zero.")
        if not self._lattice_padding.unit.is_compatible(unit.nanometer):
            raise TypeError("The lattice padding must have a unit that is compatible with nanometers.")
        if not self._lattice_padding.value_in_unit(unit.nanometer) >= 0.0:
            raise ValueError("The lattice padding must have a value greater than or equal to zero.")
        if len(self._energy_scale_range) != 2:
            raise ValueError("The energy scale range must be a sequence of exactly two floats.")
        if not 0.0 < self._energy_scale_range[0] < self._energy_scale_range[1]:
            raise ValueError("The energy scale range must satisfy 0 < min < max.")
        if not self._energy_scale_samples >= 2:
            raise ValueError("The energy scale samples must be at least 2.")

    def types(self) -> set[str]:
        """
        Return the set of particle types that will be generated by this configuration generator.

        :return:
            The set of particle types that will be generated by this configuration generator.
        :rtype: set[str]
        """
        return set(self._type_map.values())

    @staticmethod
    def _optimize_scale_factor(base_cart_coords: np.ndarray, types: list[str], radii: dict[str, unit.Quantity],
                               surface_potentials: dict[str, unit.Quantity], masses: dict[str, unit.Quantity],
                               run_parameters: RunParameters, geometric_scale_factor: float,
                               scale_range: tuple[float, float], scale_samples: int) -> float:
        """
        Find the scale factor that minimizes the steric + electrostatic energy.

        A grid of uniformly spaced scale factors is evaluated. For each scale factor, the positions
        are computed as base_cart_coords * scale (centered at the origin), and the total energy is
        computed in vacuum (non-periodic). The scale factor with the minimum energy is returned.

        :param base_cart_coords:
            The unscaled Cartesian coordinates from pymatgen, shape (N, 3).
        :type base_cart_coords: np.ndarray
        :param types:
            The type label for each particle.
        :type types: list[str]
        :param radii:
            The radii dictionary with the particle types as keys and the radii as values.
        :type radii: dict[str, unit.Quantity]
        :param surface_potentials:
            The surface potentials dictionary with the particle types as keys and the surface potentials as values.
        :type surface_potentials: dict[str, unit.Quantity]
        :param masses:
            The masses dictionary with the particle types as keys and the masses as values.
        :type masses: dict[str, unit.Quantity]
        :param run_parameters:
            The run parameters.
        :type run_parameters: RunParameters
        :param geometric_scale_factor:
            The geometric (overlap-avoidance) scale factor.
        :type geometric_scale_factor: float
        :param scale_range:
            (min_factor, max_factor) relative to the geometric scale factor.
        :type scale_range: tuple[float, float]
        :param scale_samples:
            The number of uniformly spaced scale factors to evaluate.
        :type scale_samples: int

        :return:
            The optimal scale factor that minimizes the total energy.
        :rtype: float

        :raises RuntimeError:
            If no valid scale factor is found (all energies are infinite or NaN).
        """
        scale_min = geometric_scale_factor * scale_range[0]
        scale_max = geometric_scale_factor * scale_range[1]
        candidates = np.linspace(scale_min, scale_max, scale_samples)

        system = openmm.System()

        potentials_parameters = ColloidPotentialsParameters(
            brush_density=run_parameters.brush_density, brush_length=run_parameters.brush_length,
            debye_length=run_parameters.debye_length, temperature=run_parameters.potential_temperature,
            dielectric_constant=run_parameters.dielectric_constant)

        colloid_potentials = ColloidPotentialsAlgebraic(
            colloid_potentials_parameters=potentials_parameters, use_log=run_parameters.use_log,
            cutoff_factor=run_parameters.cutoff_factor, periodic_boundary_conditions=False,
            steric_radius_average=run_parameters.steric_radius_average,
            electrostatic_radius_average=run_parameters.electrostatic_radius_average)

        for t in types:
            system.addParticle(masses[t].value_in_unit(mass_unit))
            colloid_potentials.add_particle(radius=radii[t], surface_potential=surface_potentials[t])

        for force in colloid_potentials.yield_potentials():
            system.addForce(force)

        assert not system.usesPeriodicBoundaryConditions()
        platform = openmm.Platform.getPlatformByName(run_parameters.platform_name)
        dummy_integrator = openmm.VerletIntegrator(0.001)
        context = openmm.Context(system, dummy_integrator, platform)

        energy_table = []
        for scale in candidates:
            positions = base_cart_coords * scale
            positions -= positions.mean(axis=0)
            context.setPositions(positions * length_unit)
            state = context.getState(getEnergy=True)
            potential_energy = state.getPotentialEnergy()
            energy_table.append((float(scale), float(potential_energy.value_in_unit(energy_unit))))

        # Select the scale with minimum finite energy.
        finite_entries = [(s, e) for s, e in energy_table if np.isfinite(e)]
        if not finite_entries:
            raise RuntimeError("No valid scale factor found. All candidates had infinite or NaN energy.")
        best_scale, best_energy = min(finite_entries, key=lambda x: x[1])

        finite_scales = [s for s, _ in finite_entries]
        minimum_at_boundary = (best_scale == finite_scales[0] or best_scale == finite_scales[-1])
        if minimum_at_boundary:
            warnings.warn(f"Energy minimum at boundary of scan range (scale={best_scale:.4f}). "
                          f"Consider widening energy_scale_range.")

        return best_scale

    def generate_configuration(self) -> Frame:
        """
        Generate the initial positions of the colloids in a gsd.hoomd.Frame instance.

        :return:
            The initial configuration of the colloids.
        :rtype: gsd.hoomd.Frame
        """
        # Find optimal scale factor using a small test supercell to speed up the search.
        small_supercell = self._structure.make_supercell((3, 3, 3), in_place=False, to_unit_cell=True)
        dists = distance_matrix(small_supercell.cart_coords, small_supercell.cart_coords)
        effective_radii = [self._radii[self._type_map[atomic_number]].value_in_unit(unit.nanometer)
                           + self._brush_length.value_in_unit(unit.nanometer)
                           + self._radii_padding.value_in_unit(unit.nanometer)
                           for atomic_number in small_supercell.atomic_numbers]
        required_scale_factor = 0.0
        for i in range(len(effective_radii)):
            for j in range(i + 1, len(effective_radii)):
                required_scale_factor = max(required_scale_factor,
                                            (effective_radii[i] + effective_radii[j]) / dists[i, j])
        assert all(dists[i, j] * required_scale_factor >= effective_radii[i] + effective_radii[j]
                   for i in range(len(effective_radii)) for j in range(i + 1, len(effective_radii)))

        # Apply scale factor to the full supercell.
        structure_full = self._structure.make_supercell(self._lattice_repeats, in_place=False)
        types = [self._type_map[atomic_number] for atomic_number in structure_full.atomic_numbers]

        # Optionally optimize the scale factor by energy minimization.
        if self._optimize_energy:
            required_scale_factor = self._optimize_scale_factor(
                base_cart_coords=structure_full.cart_coords, types=types, radii=self._radii,
                surface_potentials=self._surface_potentials, masses=self._masses,
                run_parameters=self._run_parameters, geometric_scale_factor=required_scale_factor,
                scale_range=self._energy_scale_range, scale_samples=self._energy_scale_samples)

        positions = structure_full.cart_coords * required_scale_factor

        # Center at origin.
        positions -= positions.mean(axis=0)
        effective_radii = [self._radii[t].value_in_unit(unit.nanometer)
                           + self._brush_length.value_in_unit(unit.nanometer)
                           + self._radii_padding.value_in_unit(unit.nanometer)
                           for t in types]

        # Embed in cubic box with padding.
        box_length = 2.0 * (np.max(np.abs(positions)) + np.max(effective_radii)
                            + self._lattice_padding.value_in_unit(unit.nanometer))

        # --- Build the Frame ---
        frame = Frame()
        frame.particles.N = len(positions)
        frame.particles.types = sorted(self.types())
        frame.particles.typeid = np.array(
            [frame.particles.types.index(t) for t in types], dtype=np.uint32)
        frame.particles.position = np.array(positions, dtype=np.float32)
        frame.configuration.box = np.array([box_length, box_length, box_length, 0.0, 0.0, 0.0], dtype=np.float32)

        return frame
