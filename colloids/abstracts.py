from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any, Iterator
from openmm import Force, unit
import yaml
from colloids.colloid_potentials_parameters import ColloidPotentialsParameters
from colloids.units import electric_potential_unit, length_unit


class OpenMMPotentialAbstract(ABC):
    """
    Abstract wrapper class for a potential of OpenMM.

    This class is a convenience wrapper for interactions that are handled by several OpenMM potentials

    The inheriting classes must implement the add_particle and yield_potentials methods so that they can be conveniently
    added to an openmm system.

    The add_particle method should be called for every particle in the system before the method yield_potentials is
    used in order to add the potential to the openmm system.
    """

    def __init__(self) -> None:
        """Constructor of the OpenMMPotentialAbstract class."""
        self._add_particle_called = False
        self._yield_potentials_called = False

    @abstractmethod
    def add_particle(self, *args: Any, **kwargs: Any) -> None:
        """
        Add a particle with the given parameters to the handled OpenMM potentials.

        This method has to be called for every particle in the system before the method yield_potentials is used.

        Note that the overriding method in the inheriting class should call this method first because it checks that the
        method yield_potentials was not called before.

        :param args:
            Parameters of the particle as positional arguments.
        :type args: Any
        :param kwargs:
            Parameters of the particle as keyword arguments.
        :type kwargs: Any

        :raises RuntimeError:
            If the method yield_potentials was called before this method.
        """
        if self._yield_potentials_called:
            raise RuntimeError("method add_particle must be called for every particle in the system before the method "
                               "yield_potentials is used")
        self._add_particle_called = True

    # noinspection PyTypeChecker
    @abstractmethod
    def yield_potentials(self) -> Iterator[Force]:
        """
        Generate all potentials that are necessary to properly include the potential in an OpenMM system.

        This method has to be called after the method add_particle was called for every particle in the system. Note
        that the overriding method in the inheriting class should call this method first because it checks that the
        method add_particle was called before.

        The generated potentials can be added to the OpenMM system using the system.addForce method.

        Note that this method should name the OpenMM potentials with their setName method. This allows to properly log
        their respective energies in the GSDReporter.

        :return:
            A generator that yields all potentials handled by this class.
        :rtype: Iterator[Force]

        :raises RuntimeError:
            If the method add_particle was not called before this method.
        """
        if not self._add_particle_called:
            raise RuntimeError("method add_particle must be called for every particle in the system before the method "
                               "yield_potentials is used")
        self._yield_potentials_called = True


class OpenMMNonbondedPotentialAbstract(OpenMMPotentialAbstract):
    """
    Abstract wrapper class for a non-bonded potential of OpenMM.

    This class is a convenience wrapper for interactions that are handled by several OpenMM CustomNonbondedForce
    potentials.

    In addition to the add_particle and yield_potentials methods, the inheriting classes must implement the
    add_exclusion method.
    """

    @abstractmethod
    def add_exclusion(self, particle_one: int, particle_two: int) -> None:
        """
        Exclude a particle pair from the non-bonded interactions handled by this class.

        :param particle_one:
            The index of the first particle.
        :type particle_one: int
        :param particle_two:
            The index of the second particle.
        :type particle_two: int
        """
        raise NotImplementedError


class ColloidPotentialsAbstract(OpenMMNonbondedPotentialAbstract):
    """
    Abstract class for the steric and electrostatic pair potentials between colloids in a solution with periodic
    boundary conditions using the CustomNonbondedForces class of openmm.

    The potentials are given in Hueckel, Hocky, Palacci & Sacanna, Nature 580, 487--490 (2020)
    (see https://doi.org/10.1038/s41586-020-2205-0). They should be implemented in the inheriting classes in one or
    several CustomNonbondedForce instances.

    The inheriting classes must implement the add_particle and yield_potentials methods.

    The steric potential from the Alexander-de Gennes polymer brush model between two colloids depends on their radii
    r_1 and r_2. Similarly, the electrostatic potential from DLVO theory between two colloids depends on their radii r_1
    and r_2 and their surface potentials psi_1 and psi_2. Before the finalized potentials are generated via the
    yield_potentials method in order to add them to the openmm system (using the system.addForce method), the
    add_particle method has to be called for each colloid in the system to define its radius and surface potential.

    :param colloid_potentials_parameters:
        The parameters of the steric and electrostatic pair potentials between colloidal particles.
    :type colloid_potentials_parameters: ColloidPotentialsParameters
    :param periodic_boundary_conditions:
        Whether this force should use periodic cutoffs for the steric and electrostatic potentials.
    :type periodic_boundary_conditions: bool
    """

    def __init__(self, colloid_potentials_parameters: ColloidPotentialsParameters,
                 periodic_boundary_conditions: bool) -> None:
        """Constructor of the ColloidPotentialsAbstract class."""
        super().__init__()
        self._parameters = colloid_potentials_parameters
        self._periodic_boundary_conditions = periodic_boundary_conditions

    @abstractmethod
    def add_particle(self, radius: unit.Quantity, surface_potential: unit.Quantity, substrate_flag: bool) -> None:
        """
        Add a colloid with a given radius and surface potential to the system.

        If the substrate flag is True, the colloid is considered to be a substrate particle. Substrate particles should
        not interact with each other. They should only interact with the other particles in the system. How this is
        implemented is up to the inheriting class.

        This method has to be called for every particle in the system before the method yield_potentials is used.

        Note that the overriding method in the inheriting class should call this method first because it checks the
        input arguments, and that the method yield_potentials was not called before (via the OpenMMPotentialAbstract
        base class).

        :param radius:
            The radius of the colloid.
            The unit of the radius must be compatible with nanometers and the value must be greater than zero.
        :type radius: unit.Quantity
        :param surface_potential:
            The surface potential of the colloid.
            The unit of the surface_potential must be compatible with millivolts.
        :type surface_potential: unit.Quantity
        :param substrate_flag:
            Whether the colloid is a substrate particle.
        :type substrate_flag: bool

        :raises TypeError:
            If the radius or surface_potential is not a Quantity with a proper unit.
        :raises ValueError:
            If the radius is not greater than zero.
        :raises RuntimeError:
            If the method yield_potentials was called before this method (via the OpenMMPotentialAbstract base class).
        """
        super().add_particle()
        if not radius.unit.is_compatible(length_unit):
            raise TypeError("argument radius must have a unit that is compatible with nanometers")
        if not radius.value_in_unit(length_unit) > 0.0:
            raise ValueError("argument radius must have a value greater than zero")
        if not surface_potential.unit.is_compatible(electric_potential_unit):
            raise TypeError("argument surface_potential must have a unit that is compatible with volts")


@dataclass(order=True, frozen=True)
class Parameters(object):
    """
    Base dataclass providing methods to store the data in a yaml file and to read the data from a yaml file.

    All standard data types (int, float, str) or standard containers (list, tuple, dict) of them can be stored in an
    inheriting data class.

    In addition, this dataclass provides methods to store OpenMM quantities in a yaml file by wrapping them in a custom
    class. In the yaml file, OpenMM quantities are stored with the tag !Quantity and two key-value pairs with the keys
    "value" and "unit".

    Furthermore, this dataclass provides methods to store references to other keys in the top level of the yaml file.
    For this the !Copy tag should be used. For example, the following excerpt from a yaml file contains a reference to
    the value of the key "potential_temperature" in the top level of the yaml file in the value of the key
    "temperature" in the dictionary "integrator_parameters":

    potential_temperature: !Quantity
        unit: kelvin
        value: 298.0
    integrator: LangevinIntegrator
    integrator_parameters:
        frictionCoeffs: !Quantity
            unit: /picosecond
            value: 0.001574074286750681
        stepSize: !Quantity
            unit: picosecond
            value: 0.00317647015905543
        temperature: !Copy
            key: potential_temperature

    This reference is resolved when the data is read from the yaml file and effectively becomes:

    potential_temperature: !Quantity
        unit: kelvin
        value: 298.0
    integrator: LangevinIntegrator
    integrator_parameters:
        frictionCoeffs: !Quantity
            unit: /picosecond
            value: 0.001574074286750681
        stepSize: !Quantity
            unit: picosecond
            value: 0.00317647015905543
        temperature: !Quantity
            unit: kelvin
            value: 298.0
    """

    class _Copy(yaml.YAMLObject):
        """
        Yaml tag for a reference to another value in the yaml file.

        This class defines the application-specific yaml tag !Copy by following the description in
        https://pyyaml.org/wiki/PyYAMLDocumentation.

        :param key:
            The key of the value to which the reference points.
        :type key: str

        :ivar key:
            The key of the value to which the reference points.
        :vartype key: str

        :cvar yaml_tag:
            The yaml tag for this class.
        :vartype yaml_tag: str
        """

        yaml_tag = u'!Copy'

        def __init__(self, key: str) -> None:
            self.key = key

    class _Quantity(yaml.YAMLObject):
        """
        Wrapper class for an OpenMM quantity that allows for a simple serialization in a yaml file.

        This class defines the application-specific yaml tag !Quantity by following the description in
        https://pyyaml.org/wiki/PyYAMLDocumentation.

        Although the !!python/object tag would in principle allow to serialize an OpenMM quantity in a yaml file, this
        tag would require specifying an enormous amount of (mostly private) attributes. This class circumvents this
        problem by defining a custom yaml tag that only stores the value and a string representation of the unit of an
        OpenMM quantity. The string representation is obtained by calling the get_name method of the unit of the OpenMM
        quantity.

        :param quantity:
            The OpenMM quantity to be wrapped.
        :type quantity: unit.Quantity

        :ivar value:
            The value of the wrapped quantity.
        :vartype value: float
        :ivar unit:
            The string representation of the unit of the wrapped quantity obtained from the get_name method.
        :vartype unit: str

        :cvar yaml_tag:
            The yaml tag for this class.
        :vartype yaml_tag: str
        """

        yaml_tag = u'!Quantity'

        def __init__(self, quantity: unit.Quantity) -> None:
            self.value = quantity.value_in_unit(quantity.unit)
            self.unit = quantity.unit.get_name()

        def to_openmm_quantity(self) -> unit.Quantity:
            """
            Convert the wrapped quantity to an openmm quantity.

            :return:
                The openmm quantity.
            :rtype: unit.Quantity
            """
            return unit.Quantity(self.value, self._openmm_unit_from_string(self.unit))

        @classmethod
        def _openmm_unit_from_string(cls, unit_string: str) -> unit.Unit:
            """Convert a string representation of a composite openmm unit (like meter/second) to an openmm unit."""

            # Remove all whitespaces from the string representation of the unit.
            string_wo_whitespaces = "".join(unit_string.split())
            # Composite units that only contain a denominator start with a slash in openmm
            if string_wo_whitespaces.startswith("/"):
                # If more than one unit is in the denominator of the composite, the units are enclosed in parentheses.
                # It appears the composite unit always ends after the closing bracket.
                if string_wo_whitespaces[1] == "(":
                    bracket_index = string_wo_whitespaces.index(")")
                    assert bracket_index == len(string_wo_whitespaces) - 1
                    return cls._openmm_unit_from_string(string_wo_whitespaces[2:bracket_index]) ** (-1)
                return cls._openmm_unit_from_string(string_wo_whitespaces[1:]) ** (-1)
            # If the composite unit does not start with a slash, it starts with one unit and ends with a
            # multiplication (*), division (/), or power (**).
            stop_index = 0
            while stop_index < len(string_wo_whitespaces) and string_wo_whitespaces[stop_index] not in ["*", "/"]:
                stop_index += 1
            try:
                openmm_unit = unit.__dict__[string_wo_whitespaces[:stop_index]]
            except KeyError:
                # The first unit in the composite unit may contain a SI prefix that must be found explicitly because
                # units like millivolt are not directly recognized by openmm.
                openmm_unit = None
                for si_prefix in unit.si_prefixes:
                    if string_wo_whitespaces.startswith(si_prefix.prefix):
                        assert stop_index > len(si_prefix.prefix)
                        openmm_unit = si_prefix * unit.__dict__[string_wo_whitespaces[len(si_prefix.prefix):stop_index]]
                        break
            # If the composite unit only contains one unit, the conversion is finished.
            if stop_index == len(string_wo_whitespaces):
                return openmm_unit
            # Check if the unit that was just found is followed by a power.
            # This power appears to be always positive.
            if string_wo_whitespaces[stop_index] == "*" and string_wo_whitespaces[stop_index + 1] == "*":
                power_index = 1
                assert len(string_wo_whitespaces) > stop_index + 2
                power = int(string_wo_whitespaces[stop_index + 2:stop_index + 2 + power_index])
                # Simply try out how many digits the power has by trying to convert the substring to an integer.
                while stop_index + 2 + power_index < len(string_wo_whitespaces):
                    try:
                        power_index += 1
                        power = int(string_wo_whitespaces[stop_index + 2:stop_index + 2 + power_index])
                    except ValueError:
                        power_index -= 1
                        break
                openmm_unit = openmm_unit ** power
                stop_index += 2 + power_index
                # Check if the conversion is finished.
                if stop_index == len(string_wo_whitespaces):
                    return openmm_unit
            # Check if the unit that was just found (possibly with a power) is followed by a multiplication or division.
            if string_wo_whitespaces[stop_index] == "*":
                return openmm_unit * cls._openmm_unit_from_string(string_wo_whitespaces[stop_index + 1:])
            if string_wo_whitespaces[stop_index] == "/":
                # If more than one unit is in the denominator of the composite, the units are enclosed in parentheses.
                # It appears the composite unit always ends after the closing bracket.
                if string_wo_whitespaces[stop_index + 1] == "(":
                    bracket_index = string_wo_whitespaces[stop_index + 1:].index(")")
                    assert stop_index + 1 + bracket_index == len(string_wo_whitespaces) - 1
                    return (openmm_unit
                            / (cls._openmm_unit_from_string(
                                string_wo_whitespaces[stop_index + 2:stop_index + 1 + bracket_index])))
                return openmm_unit / cls._openmm_unit_from_string(string_wo_whitespaces[stop_index + 1:])
            raise RuntimeError("This should not happen.")

    @classmethod
    def from_yaml(cls, filename: str) -> "Parameters":
        """
        Read the parameters of this dataclass from a yaml file.

        Instances of the _Quantity class are converted to OpenMM quantities.

        :param filename:
            The name of the yaml file.
        :type filename: str

        :return:
            The parameters.
        :rtype: RunParameters
        """
        with open(filename, "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        return cls.from_dict(params)

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "Parameters":
        """
        Read the parameters of this dataclass from a dictionary.

        Instances of the _Quantity class are converted to OpenMM quantities.

        :param params:
            The dictionary with the parameters.
        :type params: dict[str, Any]

        :return:
            The parameters.
        :rtype: RunParameters
        """
        for key, value in params.items():
            params[key] = cls._resolve_reference_values(params, value)
        for key, value in params.items():
            params[key] = cls._convert_to_openmm_quantity(value)
        # noinspection PyArgumentList
        return cls(**params)

    @staticmethod
    def _resolve_reference_values(base_params: dict[str, Any], value: Any) -> Any:
        """Recursively resolve references to other parameters in the base_params dictionary."""
        if isinstance(value, Parameters._Copy):
            if value.key not in base_params:
                raise ValueError(f"Reference to {value.key} not found at the top level of the yaml file.")
            return base_params[value.key]
        elif isinstance(value, dict):
            return_dict = value
            for key, sub_value in value.items():
                return_dict[key] = Parameters._resolve_reference_values(base_params, sub_value)
            return return_dict
        else:
            return value

    def to_yaml(self, filename: str) -> None:
        """
        Write a yaml file with the parameters of this dataclass.

        OpenMM quantities are converted to _Quantity objects.

        :param filename:
            The name of the yaml file.
        :type filename: str
        """
        with open(filename, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        """
        Represent this dataclass as a dictionary while converting all OpenMM quantities to _Quantity objects.

        :return:
            The dictionary with the parameters.
        :rtype: dict[str, Any]
        """
        result_dict = {}
        for f in fields(self):
            assert f.name not in result_dict
            result_dict[f.name] = self._convert_to_quantity(getattr(self, f.name))
        return result_dict

    @staticmethod
    def _convert_to_quantity(obj):
        """Recursively convert OpenMM quantities to _Quantity objects."""
        if isinstance(obj, (list, tuple)):
            return type(obj)(Parameters._convert_to_quantity(item) for item in obj)
        elif isinstance(obj, dict):
            return dict((Parameters._convert_to_quantity(key), Parameters._convert_to_quantity(value))
                        for key, value in obj.items())
        elif isinstance(obj, unit.Quantity):
            return Parameters._Quantity(obj)
        else:
            return deepcopy(obj)

    @staticmethod
    def _convert_to_openmm_quantity(obj):
        """Recursively convert _Quantity objects to OpenMM quantities."""
        if isinstance(obj, (list, tuple)):
            return type(obj)(Parameters._convert_to_openmm_quantity(item) for item in obj)
        elif isinstance(obj, dict):
            return dict((Parameters._convert_to_openmm_quantity(key),
                         Parameters._convert_to_openmm_quantity(value))
                        for key, value in obj.items())
        elif isinstance(obj, Parameters._Quantity):
            return obj.to_openmm_quantity()
        else:
            return deepcopy(obj)
