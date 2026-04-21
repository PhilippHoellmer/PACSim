import math
from typing import Iterator, Optional, Sequence
import warnings
from openmm import CustomExternalForce, unit
from colloids.abstracts import OpenMMPotentialAbstract
from colloids import ColloidPotentialsParameters
from colloids.units import electric_potential_unit, energy_unit, length_unit


class ShiftedLennardJonesWalls(OpenMMPotentialAbstract):
    """
    This class sets up the shifted Lennard-Jones potentials for closed-wall simulations using the CustomExternalForce
    class of openmm.

    The shifted Lennard-Jones potential as a wall follows the implementation of hoomd (see
    https://hoomd-blue.readthedocs.io/en/v2.9.4/module-md-wall.html#hoomd.md.wall.slj).

    This class allows to independently switch on walls in the x, y, and z directions. The walls are placed at
    +-wall_distance / 2 for every specified direction with its specified wall_distance.

    The shifted Lennard-Jones potential acts on colloid particles within a certain cutoff distance of every wall. This
    cutoff distance depends on the particle radius and is given by r_cut - delta, where r_cut = radius * 2^(1/6) and
    delta = radius - 1. Outside of this range, the external force acting on a particle is 0.

    The Lennard-Jones potential is shifted so that it starts smoothly at zero at the cutoff distance.

    The shifted Lennard-Jones potential as a function of the distance r to the wall is given by:
    slj(r) = 4 * epsilon * ((radius / (r - delta))^12 - alpha * (radius / (r - delta))^6)
             - 4 * epsilon * ((radius / r_cut)^12 - alpha * (radius / r_cut)^6)

    :param wall_distances:
        A list of three distances specifying the dimensions of the simulation box in the x, y, and z directions.
        This is used to determine the location of the SLJ walls at +-wall_distance/2 for every active wall direction.
        For any inactive wall direction (see wall_directions parameter), the corresponding wall distance must be None.
        For any active wall direction, the corresponding wall distance must be specified.
        The unit of any wall distance must be compatible with nanometer and the value must be greater than zero.
    :type wall_distances: Sequence[Optional[unit.Quantity]]
    :param epsilon:
        The unshifted Lennard-Jones potential well-depth.
        The unit of the epsilon must be compatible with kilojoules_per_mole and the value must be greater than zero.
    :type epsilon: unit.Quantity
    :param alpha:
        Factor determining the strength of the attractive part of the Lennard-Jones potential.
        This factor has to satisfy 0 <= alpha <= 1.
        Note that the force of this potential is only continuous if alpha = 1.
    :type alpha: float
    :param wall_directions:
        A list of three booleans indicating whether the walls in the x, y, and z directions are active.
        Defaults to [False, False, False].
    :type wall_directions: list[bool]
    :param use_substrate:
        A boolean indicating whether the bottom wall is replaced by a substrate.
        This is only possible if all wall directions are active.
        Defaults to False.
    :type use_substrate: bool

    :raises TypeError:
        If epsilon or any wall distance for an active wall direction is not a Quantity with a proper unit.

    :raises ValueError:
        If epsilon or any wall distance for an active wall direction is not greater than zero.
        If alpha is not in the interval [0, 1].
        If no wall direction is active.
        If not exactly three wall directions are specified.
        If not exactly three wall distances are specified.
        If a wall distance is specified for an inactive wall direction.
        If a wall distance is not specified for an active wall direction.
        If not all wall directions are active if a substrate is used.
    """

    _name = "wall_energy"

    def __init__(self, wall_distances: Sequence[Optional[unit.Quantity]], epsilon: unit.Quantity, alpha: float,
                 wall_directions: Sequence[bool] = (True, True, True), use_substrate: bool = False, use_pbc: bool = True) -> None:
        """Constructor of the ShiftedLennardJonesWalls class."""
        super().__init__()

        if not any(wall_directions):
            raise ValueError("at least one wall direction must be active")
        if len(wall_directions) != 3:
            raise ValueError("wall directions must be specified for three dimensions")
        if len(wall_distances) != 3:
            raise ValueError("wall distances must be specified for three dimensions")
        for wdir, wdist in zip(wall_directions, wall_distances):
            if wdir:
                if wdist is None:
                    raise ValueError("wall distance must be specified for any active wall direction")
                if not wdist.unit.is_compatible(length_unit):
                    raise TypeError("any wall distance must have a unit that is compatible with nanometers")
                if not wdist.value_in_unit(length_unit) > 0.0:
                    raise ValueError("any wall distance must have a value greater than zero")
            else:
                if wdist is not None:
                    raise ValueError("wall distance must not be specified for inactive wall direction")
        if not epsilon.unit.is_compatible(energy_unit):
            raise TypeError("argument epsilon must have a unit that is compatible with kilojoules per mole")
        if not epsilon.value_in_unit(energy_unit) > 0.0:
            raise ValueError("argument epsilon must have a value greater than zero")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("argument alpha must satisfy 0 <= alpha <= 1")
        if alpha != 1.0:
            warnings.warn("The force of the shifted Lennard-Jones potential as a wall is only continuous if alpha = 1.")
        if use_substrate:
            if not all(wall_directions):
                raise ValueError("all wall directions must be active if a substrate is used")

        self._wall_distances = wall_distances
        self._epsilon = epsilon
        self._alpha = alpha
        self._wall_directions = wall_directions
        self._use_substrate = use_substrate
        self._use_pbc = use_pbc
        self._slj_potential = self._set_up_slj_potential()

    def _set_up_slj_potential(self) -> CustomExternalForce:
        """Set up the basic functional form of the shifted Lennard Jones potential."""

        slj_x = ("step(periodicdistance(x, 0, 0, 0, 0, 0) - cutoff_x) * ("
                 "four_epsilon * "
                 "((radius / (wall_distance_x_over_two_minus_delta - periodicdistance(x, 0, 0, 0, 0, 0)))^12 "
                 "- alpha * (radius / (wall_distance_x_over_two_minus_delta - periodicdistance(x, 0, 0, 0, 0, 0)))^6)"
                 "+ shift)")
        slj_y = ("step(periodicdistance(0, y, 0, 0, 0, 0) - cutoff_y) * ("
                 "four_epsilon * "
                 "((radius / (wall_distance_y_over_two_minus_delta - periodicdistance(0, y, 0, 0, 0, 0)))^12 "
                 "- alpha * (radius / (wall_distance_y_over_two_minus_delta - periodicdistance(0, y, 0, 0, 0, 0)))^6)"
                 "+ shift)")
        slj_z = ("step(periodicdistance(0, 0, z, 0, 0, 0) - cutoff_z) * ("
                 "four_epsilon * "
                 "((radius / (wall_distance_z_over_two_minus_delta - periodicdistance(0, 0, z, 0, 0, 0)))^12 "
                 "- alpha * (radius / (wall_distance_z_over_two_minus_delta - periodicdistance(0, 0, z, 0, 0, 0)))^6)"
                 "+ shift)")
        # Using periodicdistance switches on periodic boundary conditions in the OpenMM system.
        # If there are walls in all directions or for other reasons, we might not want periodic 
        # boundary conditions though.
        if self._use_pbc and self._use_substrate:
            # Only the bottom wall is replaced by a substrate.
            # The top wall is still a shifted Lennard-Jones wall.
            slj_z = slj_z.replace("periodicdistance(0, 0, z, 0, 0, 0)", "z")
        
        elif not self._use_pbc:
            slj_x = slj_x.replace("periodicdistance(x, 0, 0, 0, 0, 0)", "abs(x)")
            slj_y = slj_y.replace("periodicdistance(0, y, 0, 0, 0, 0)", "abs(y)")
            if self._use_substrate:
                slj_z = slj_z.replace("periodicdistance(0, 0, z, 0, 0, 0)", "z")
            else:
                slj_z = slj_z.replace("periodicdistance(0, 0, z, 0, 0, 0)", "abs(z)")


        slj_string = "+".join(slj for slj, wdir in zip([slj_x, slj_y, slj_z], self._wall_directions) if wdir)
        assert slj_string

        slj_potential = CustomExternalForce(slj_string)
        slj_potential.addGlobalParameter("four_epsilon",
                                         4.0 * self._epsilon.value_in_unit(energy_unit))
        slj_potential.addGlobalParameter("alpha", self._alpha)
        slj_potential.addPerParticleParameter("radius")
        slj_potential.addPerParticleParameter("shift")

        if self._wall_directions[0]:
            slj_potential.addPerParticleParameter("wall_distance_x_over_two_minus_delta")
            slj_potential.addPerParticleParameter("cutoff_x")
        if self._wall_directions[1]:
            slj_potential.addPerParticleParameter("wall_distance_y_over_two_minus_delta")
            slj_potential.addPerParticleParameter("cutoff_y")
        if self._wall_directions[2]:
            slj_potential.addPerParticleParameter("wall_distance_z_over_two_minus_delta")
            slj_potential.addPerParticleParameter("cutoff_z")

        return slj_potential

    def add_particle(self, index: int, radius: unit.Quantity) -> None:
        """
        Add a colloid with a given radius to the system.

        This method has to be called for every particle in the system before the method yield_potentials is used.

        :param index:
            The index of the particle in the OpenMM system.
        :type index: int
        :param radius:
            The radius of the colloid.
            The unit of the radius must be compatible with nanometers and the value must be greater than zero.
        :type radius: unit.Quantity
        
        :raises TypeError:
            If the radius is not a Quantity with a proper unit (via the abstract base class).
        :raises ValueError:
            If the radius is not greater than zero (via the abstract base class).
        :raises RuntimeError:
            If this method is called after the yield_potentials method (via the abstract base class).
        """
        super().add_particle()
        if not radius.unit.is_compatible(length_unit):
            raise TypeError("argument radius must have a unit that is compatible with nanometers")
        if not radius.value_in_unit(length_unit) > 0.0:
            raise ValueError("argument radius must have a value greater than zero")
        for wall_distance in self._wall_distances:
            if wall_distance is not None:
                if not wall_distance / 2.0 > radius * 2 ** (1 / 6) + radius - 1.0 * length_unit:
                    raise ValueError("The colloid radius leads to a cutoff radius * 2^(1/6) + radius - 1 in the "
                                     "shifted Lennard-Jones wall that exceeds half of the wall distance.")
        rcut = (2.0 ** (1.0 / 6.0)) * radius
        per_particle_parameters = [
            radius.value_in_unit(length_unit),  # radius
            (-4.0 * self._epsilon * ((radius / rcut) ** 12 - self._alpha * (radius / rcut) ** 6)).value_in_unit(
                energy_unit)  # shift
        ]
        if self._wall_directions[0]:
            per_particle_parameters.append(
                (self._wall_distances[0] / 2.0 - radius + 1.0 * length_unit).value_in_unit(
                    length_unit)  # wall_distance_x_over_two_minus_delta
            )
            per_particle_parameters.append(
                (self._wall_distances[0] / 2.0 - rcut - radius + 1.0 * length_unit).value_in_unit(
                    length_unit)  # cutoff_x
            )
        if self._wall_directions[1]:
            per_particle_parameters.append(
                (self._wall_distances[1] / 2.0 - radius + 1.0 * length_unit).value_in_unit(
                    length_unit)  # wall_distance_y_over_two_minus_delta
            )
            per_particle_parameters.append(
                (self._wall_distances[1] / 2.0 - rcut - radius + 1.0 * length_unit).value_in_unit(
                    length_unit)  # cutoff_y
            )
        if self._wall_directions[2]:
            per_particle_parameters.append(
                (self._wall_distances[2] / 2.0 - radius + 1.0 * length_unit).value_in_unit(
                    length_unit)  # wall_distance_z_over_two_minus_delta
            )
            per_particle_parameters.append(
                (self._wall_distances[2] / 2.0 - rcut - radius + 1.0 * length_unit).value_in_unit(
                    length_unit)  # cutoff_z
            )

        self._slj_potential.addParticle(index, per_particle_parameters)

    def yield_potentials(self) -> Iterator[CustomExternalForce]:
        """
        Generate all potentials that are necessary to properly include the shifted Lennard-Jones walls in the OpenMM
        system.

        This method has to be called after the method add_particle was called for every particle in the system.

        :return:
            A generator that yields shifted Lennard-Jones walls handled by this class.
        :rtype: Iterator[CustomExternalForce]

        :raises RuntimeError:
            If the method add_particle was not called before this method (via the abstract base class).
        """
        super().yield_potentials()
        self._slj_potential.setName(self._name)
        yield self._slj_potential


class ImplicitSubstrateWall(OpenMMPotentialAbstract):
    """
    This class sets up an implicit substrate at the bottom of the simulation box using the CustomExternalForce class of
    OpenMM. This is an alternative to explicitly modeling the substrate using a layer of fixed particles, and is
    designed to reduce computational costs. Either implicit or explicit substrate may be used but not both.

    A substrate can only be used when all SLJ walls are active. The bottom wall in the z direction will be replaced by
    the implicit substrate wall.

    The implicit substrate is modeled as a single substrate particle with a flat surface (that is, with an infinite
    radius) at the bottom of the simulation box. The substrate particle is charged and interacts with the colloidal
    particles via the same steric and electrostatic pair potentials as, e.g., defined in the ColloidPotentialsAlgebraic
    class. An infinite substrate radius implies a value of 2 * radius for the prefactors in the steric and electrostatic
    potentials, where radius is the radius of the colloidal particles.

    :param colloid_potentials_parameters:
        The parameters of the steric and electrostatic pair potentials between colloidal particles.
    :type colloid_potentials_parameters: ColloidPotentialsParameters
    :param wall_distance_z:
        A distance specifying the dimensions of the simulation box in the z direction.
        This is used to determine the location of the substrate wall at -wall_distance/2.
        The unit must be compatible with nanometer and the value must be greater than zero.
    :type wall_distance_z: unit.Quantity
    :param substrate_charge:
        The charge of the implicit substrate particles.
    :type substrate_charge: unit.Quantity
    :param use_log:
        If True, the electrostatic force uses the more accurate equation involving a logarithm [i.e., eq. (12.5.2) in
        Hunter, Foundations of Colloid Science (Oxford University Press, 2001), 2nd edition] instead of the simpler
        equation that only involves an exponential [i.e., eq. (12.5.5) in Hunter, Foundations of Colloid Science
        (Oxford University Press, 2001), 2nd edition].
    :type use_log: bool

    :raises TypeError:
        If the substrate charge for an active substrate wall is not a Quantity with a proper unit.
        If the wall distance for an active substrate wall is not a Quantity with a proper unit.
    :raises ValueError:
        If the wall distance for an active substrate wall is not greater than zero.
    """

    _steric_prefactor_unit = energy_unit / (length_unit ** 3)
    _electrostatic_prefactor_unit = energy_unit / (length_unit * electric_potential_unit ** 2)
    _name = "implicit_substrate_energy"

    def __init__(self, colloid_potentials_parameters: ColloidPotentialsParameters, wall_distance_z: unit.Quantity,
                 substrate_charge: unit.Quantity, use_log: bool) -> None:
        """Constructor of the ImplicitSubstrate class."""
        super().__init__()

        if not wall_distance_z.unit.is_compatible(length_unit):
            raise TypeError("wall distance must have a unit that is compatible with nanometers")
        if not wall_distance_z.value_in_unit(length_unit) > 0.0:
            raise ValueError("wall distance must have a value greater than zero")
        if not substrate_charge.unit.is_compatible(electric_potential_unit):
            raise TypeError("substrate charge must have a unit that is compatible with volts")

        self._parameters = colloid_potentials_parameters
        self._substrate_charge = substrate_charge
        self._wall_distance = wall_distance_z
        self._use_log = use_log
        self._substrate_wall_potential = self._set_up_substrate_wall_potential()

    def _set_up_substrate_wall_potential(self) -> CustomExternalForce:
        """Set up the basic functional form of the substrate wall."""

        """Set up the basic functional form of the steric potential from the Alexander-de Gennes polymer brush model."""
        # 2 / (1 / radius1 + 1 / radius2) = 2 * radius1 if radius2 = infinity.

        steric_potential = (
            "step(two_l - h) * "
            "steric_prefactor * 2.0 * radius * brush_length * brush_length * ("
            "28.0 * ((two_l / h)^0.25 - 1.0) "
            "+ 20.0 / 11.0 * (1.0 - (h / two_l)^2.75)"
            "+ 12.0 * (h / two_l - 1.0)) "
        )

        """Set up the basic functional form of the electrostatic potential from DLVO theory."""
        if self._use_log:
            # 2 / (1 / radius1 + 1 / radius2) = 2 * radius1 if radius2 = infinity.
            electrostatic_potential = (
                "electrostatic_prefactor * 2 * radius * psi * substrate_psi * log(1.0 + exp(-h / debye_length));")

        else:
            # 2 / (1 / radius1 + 1 / radius2) = 2 * radius1 if radius2 = infinity.
            electrostatic_potential = (
                "electrostatic_prefactor * 2 * radius * psi * substrate_psi * exp(-h / debye_length);")

        substrate_string = "+".join([steric_potential, electrostatic_potential]) #steric_potential + electrostatic_potential
        # +z so that close to zero when z~-L/2 + radius.
        # The surface separation to the implicit substrate is h = L/2 - radius + 1 + z.
        # The +1 shift is analogous to the shift in the SLJ walls.
        # This means that the repulsive potential diverges at a distance of radius - 1 to the implicit substrate wall.
        substrate_string += ("h = substrate_wall_distance + z - radius + 1.0;"
                             "two_l = 2.0 * brush_length")

        substrate_wall_potential = CustomExternalForce(substrate_string)

        # Steric prefactor is k_B * T * 16 * pi * sigma^(3/2) / 35 (see Hocky paper)
        substrate_wall_potential.addGlobalParameter(
            "steric_prefactor",
            (unit.BOLTZMANN_CONSTANT_kB * self._parameters.temperature
             * 16.0 * math.pi * (self._parameters.brush_density ** (3 / 2)) / 35.0
             * unit.AVOGADRO_CONSTANT_NA).value_in_unit(self._steric_prefactor_unit)
        )
        # Brush length L (see Hocky paper)
        substrate_wall_potential.addGlobalParameter("brush_length",
                                                    self._parameters.brush_length.value_in_unit(length_unit))

        # Electrostatic prefactor is 2 * pi * epsilon
        substrate_wall_potential.addGlobalParameter(
            "electrostatic_prefactor",
            (2.0 * math.pi * self._parameters.VACUUM_PERMITTIVITY * self._parameters.dielectric_constant
             * unit.AVOGADRO_CONSTANT_NA).value_in_unit(self._electrostatic_prefactor_unit))
        substrate_wall_potential.addGlobalParameter("debye_length",
                                                    self._parameters.debye_length.value_in_unit(length_unit))
        substrate_wall_potential.addGlobalParameter("substrate_psi",
                                                    self._substrate_charge.value_in_unit(electric_potential_unit))
        substrate_wall_potential.addGlobalParameter("substrate_wall_distance",
                                                    (self._wall_distance / 2.0).value_in_unit(length_unit))

        substrate_wall_potential.addPerParticleParameter("radius")
        # Psi should be given in millivolts.
        substrate_wall_potential.addPerParticleParameter("psi")

        return substrate_wall_potential

    def add_particle(self, index: int, radius: unit.Quantity, surface_potential: unit.Quantity) -> None:
        """
        Add a colloid with a given radius and surface potential to the system.

        This method has to be called for every particle in the system before the method yield_potentials is used.

        :param index:
            The index of the particle in the OpenMM system.
        :type index: int
        :param radius:
            The radius of the colloid.
            The unit of the radius must be compatible with nanometers and the value must be greater than zero.
        :type radius: unit.Quantity
        :param surface_potential:
            The surface potential of the colloid.
            The unit of the surface_potential must be compatible with millivolts.
        :type surface_potential: unit.Quantity

        :raises TypeError:
            If the radius or surface potential is not a Quantity with a proper unit.
        :raises ValueError:
            If the radius is not greater than zero.
        :raises RuntimeError:
            If this method is called after the yield_potentials method (via the abstract base class).
        """
        super().add_particle()
        if not radius.unit.is_compatible(length_unit):
            raise TypeError("argument radius must have a unit that is compatible with nanometers")
        if not radius.value_in_unit(length_unit) > 0.0:
            raise ValueError("argument radius must have a value greater than zero")
        if not surface_potential.unit.is_compatible(electric_potential_unit):
            raise TypeError("argument surface_potential must have a unit that is compatible with volts")

        self._substrate_wall_potential.addParticle(index, [radius.value_in_unit(length_unit),
                                                           surface_potential.value_in_unit(electric_potential_unit)])

    def yield_potentials(self) -> Iterator[CustomExternalForce]:
        """
        Generate all potentials in the systems that are necessary to properly include the substrate wall.

        This method has to be called after the method add_particle was called for every particle in the system.

        :return:
            A generator that yields the substrate wall handled by this class.
        :rtype: Iterator[CustomExternalForce]

        :raises RuntimeError:
            If the method add_particle was not called before this method (via the abstract base class).
        """
        super().yield_potentials()
        self._substrate_wall_potential.setName(self._name)
        yield self._substrate_wall_potential
