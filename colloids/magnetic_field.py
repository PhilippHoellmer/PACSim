import math
from numbers import Integral
from typing import Iterator, Optional, Sequence

import openmm.app
from openmm import CustomExternalForce, unit

from colloids.abstracts import OpenMMPotentialAbstract
from colloids.units import energy_unit, length_unit, time_unit


class MagneticField(OpenMMPotentialAbstract):
    """
    This class sets up a magnetic-field potential using the CustomExternalForce class of OpenMM.

    The magnetic field acts on a selected set of particle type ids. For a DC field, the potential is linear in x and
    y. For an AC field, each component is multiplied by a cosine with its own frequency and phase.
    """

    _name = "magnetic_energy"
    _force_unit = energy_unit / length_unit
    _frequency_unit = time_unit ** (-1)

    def __init__(self, field_type: str, amplitude_x: unit.Quantity, amplitude_y: unit.Quantity,
                 typeids: Sequence[int], frequency_x: unit.Quantity = None,
                 frequency_y: unit.Quantity = None, phase_x: float = None, phase_y: float = None,
                 use_pbc: bool = True, box_lengths: Optional[Sequence[unit.Quantity]] = None) -> None:
        super().__init__()

        if not isinstance(field_type, str):
            raise TypeError("argument field_type must be a string")
        self._field_type = field_type.upper()
        if self._field_type not in ["DC", "AC"]:
            raise ValueError("argument field_type must be 'DC' or 'AC'")
        if not amplitude_x.unit.is_compatible(self._force_unit):
            raise TypeError("argument amplitude_x must have a unit compatible with energy per unit length")
        if not amplitude_y.unit.is_compatible(self._force_unit):
            raise TypeError("argument amplitude_y must have a unit compatible with energy per unit length")
        if not isinstance(typeids, Sequence) or isinstance(typeids, (str, bytes)):
            raise TypeError("argument typeids must be a sequence of integers")
        self._typeids = set()
        for typeid in typeids:
            if not isinstance(typeid, Integral):
                raise TypeError("argument typeids must be integers")
            self._typeids.add(int(typeid))
        if len(self._typeids) == 0:
            raise ValueError("argument typeids must contain at least one particle type id")
        if self._field_type == "DC":
            if frequency_x is not None or frequency_y is not None:
                raise ValueError("frequencies must not be specified for a DC magnetic field")
            if phase_x is not None or phase_y is not None:
                raise ValueError("phases must not be specified for a DC magnetic field")
        else:
            if frequency_x is None or frequency_y is None:
                raise ValueError("frequencies must be specified for an AC magnetic field")
            if phase_x is None or phase_y is None:
                raise ValueError("phases must be specified for an AC magnetic field")
            if not frequency_x.unit.is_compatible(self._frequency_unit):
                raise TypeError("argument frequency_x must have a unit compatible with 1/picosecond")
            if not frequency_y.unit.is_compatible(self._frequency_unit):
                raise TypeError("argument frequency_y must have a unit compatible with 1/picosecond")
            if not isinstance(phase_x, (int, float)):
                raise TypeError("argument phase_x must be a number")
            if not isinstance(phase_y, (int, float)):
                raise TypeError("argument phase_y must be a number")

        self._amplitude_x = amplitude_x
        self._amplitude_y = amplitude_y
        self._frequency_x = frequency_x
        self._frequency_y = frequency_y
        self._phase_x = phase_x
        self._phase_y = phase_y
        self._use_pbc = use_pbc
        self._box_lengths = box_lengths
        self._magnetic_field = self._set_up_magnetic_field_potential()

    def _set_up_magnetic_field_potential(self) -> CustomExternalForce:
        x_coordinate = "x"
        y_coordinate = "y"
        if self._field_type == "DC":
            magnetic_field_string = (
                "type_mask * ("
                f"magnetic_field_amplitude_x * {x_coordinate} + magnetic_field_amplitude_y * {y_coordinate}"
                ")"
            )
        else:
            magnetic_field_string = (
                "type_mask * ("
                "magnetic_field_amplitude_x * cos(two_pi * magnetic_field_frequency_x * magnetic_field_time + "
                "magnetic_field_phase_x) * "
                f"{x_coordinate}"
                " + magnetic_field_amplitude_y * cos(two_pi * magnetic_field_frequency_y * magnetic_field_time + "
                "magnetic_field_phase_y) * "
                f"{y_coordinate}"
                ")"
            )

        magnetic_field = CustomExternalForce(magnetic_field_string)
        magnetic_field.addGlobalParameter("magnetic_field_amplitude_x",
                                          self._amplitude_x.value_in_unit(self._force_unit))
        magnetic_field.addGlobalParameter("magnetic_field_amplitude_y",
                                          self._amplitude_y.value_in_unit(self._force_unit))
        magnetic_field.addGlobalParameter("two_pi", 2.0 * math.pi)
        magnetic_field.addGlobalParameter("magnetic_field_time", 0.0)
        magnetic_field.addPerParticleParameter("type_mask")
        if self._field_type == "AC":
            magnetic_field.addGlobalParameter("magnetic_field_frequency_x",
                                              self._frequency_x.value_in_unit(self._frequency_unit))
            magnetic_field.addGlobalParameter("magnetic_field_frequency_y",
                                              self._frequency_y.value_in_unit(self._frequency_unit))
            magnetic_field.addGlobalParameter("magnetic_field_phase_x", self._phase_x)
            magnetic_field.addGlobalParameter("magnetic_field_phase_y", self._phase_y)

        return magnetic_field

    def add_particle(self, index: int, typeid: int) -> None:
        super().add_particle()
        if not isinstance(typeid, Integral):
            raise TypeError("argument typeid must be an integer")
        type_mask = 1.0 if int(typeid) in self._typeids else 0.0
        self._magnetic_field.addParticle(index, [type_mask])

    def yield_potentials(self) -> Iterator[CustomExternalForce]:
        super().yield_potentials()
        self._magnetic_field.setName(self._name)
        yield self._magnetic_field


class MagneticFieldTimeUpdateReporter(object):
    """
    Reporter that keeps the magnetic-field time parameter aligned with the OpenMM simulation time.
    """

    def __init__(self, update_interval: int = 1) -> None:
        if not update_interval > 0:
            raise ValueError("The update interval must be greater than zero.")
        self._update_interval = update_interval

    def describeNextReport(self, simulation: openmm.app.Simulation) -> tuple[int, bool, bool, bool, bool, bool]:
        steps = self._update_interval - simulation.currentStep % self._update_interval
        return steps, False, False, False, False, False

    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        simulation.context.setParameter("magnetic_field_time", state.getTime().value_in_unit(time_unit))