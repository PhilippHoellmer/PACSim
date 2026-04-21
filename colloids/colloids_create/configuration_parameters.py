from dataclasses import dataclass, field
from typing import Any, Optional
import inspect
from openmm import unit
from colloids.abstracts import Parameters
from colloids.units import electric_potential_unit, length_unit, mass_unit
from colloids.colloids_create import ConfigurationGenerator
import colloids.colloids_create.configuration_generators as configuration_generators
import colloids.colloids_create.initial_modifiers as initial_modifiers
import colloids.colloids_create.final_modifiers as final_modifiers


@dataclass(order=True, frozen=True)
class ConfigurationParameters(Parameters):
    """
    Data class for the parameters of the colloids configuration to be created for an OpenMM simulation.

    This dataclass can be written to and read from a yaml file. The yaml file contains the parameters as key-value
    pairs. Any OpenMM quantities are converted to Quantity objects that can be represented in a readable way in the
    yaml file.

    The base configuration is constructed by a configuration generator. The generator class is specified by name in the
    configuration_generator field, and its constructor parameters are provided as a dictionary in the
    configuration_generator_parameters field.

    Available configuration generators can be found in the configuration_generators package.

    After the base configuration has been created, it can be modified by applying a series of configuration modifiers.
    These modifiers can modify the positions of the colloids, add or remove colloids, or modify other properties of the
    colloids. The modifiers are applied in two stages: initial modifiers (such as adding a substrate at the bottom of
    the simulation box) are applied before setting the particle properties (diameter, charge, mass), and final modifiers
    (such as including a seed of colloids from a gsd file while removing overlapping particles from the base
    configuration) are applied after setting the particle properties.

    :param configuration_generator:
        The name of the configuration generator class to use for creating the initial configuration.
        Available choices can be found in the configuration_generators package.
        Defaults to "ClusterGenerator".
    :type configuration_generator: str
    :param configuration_generator_parameters:
        Dictionary of parameters to pass to the configuration generator's __init__ method.
        The expected parameters depend on the chosen generator class.
        Defaults to the default ClusterGenerator parameters.
    :type configuration_generator_parameters: dict[str, Any]
    :param masses:
        The masses of the different types of colloidal particles that appear in the configuration.
        The keys of the dictionary are the types of the colloidal particles and the values are the masses.
        The unit of the masses must be compatible with atomic mass units and the values must be greater than zero,
        except for immobile particles (as the substrate), which should have a mass of zero.
        Defaults to {"1": 1.0 * amu, "2": (95.0 / 105.0) ** 3 * amu}.
    :type masses: dict[str, unit.Quantity]
    :param radii:
        The radii of the different types of colloidal particles that appear in the initial configuration file.
        The keys of the dictionary are the types of the colloidal particles and the values are the radii.
        The unit of the radii must be compatible with nanometers and the values must be greater than zero.
        Defaults to {"1": 105.0 * nanometer, "2": 95.0 * nanometer}.
    :type radii: dict[str, unit.Quantity]
    :param surface_potentials:
        The surface potentials of the different types of colloidal particles that appear in the initial configuration
        file.
        The keys of the dictionary are the types of the colloidal particles and the values are the surface potentials.
        The unit of the surface potentials must be compatible with millivolts.
        Defaults to {"1": 44.0 * millivolt, "2": -54.0 * millivolt}.
    :type surface_potentials: dict[str, unit.Quantity]
    :param initial_modifiers:
        List of modifier class names to apply before setting particle properties (diameter, charge, mass).
        These modifiers run early in the configuration generation process.
        Possible choices can be found in the colloids_create.initial_modifiers module.
        If initial modifiers are specified, their parameters must be specified as well in the
        initial_modifiers_parameters list.
        Defaults to None.
    :type initial_modifiers: Optional[list[str]]
    :param initial_modifiers_parameters:
        List of dictionaries containing parameters for each initial modifier.
        Each dictionary is passed to the corresponding modifier's __init__ method.
        The list must have the same length as initial_modifiers.
        Defaults to None.
    :type initial_modifiers_parameters: Optional[list[dict[str, Any]]]
    :param final_modifiers:
        List of modifier class names to apply after setting particle properties (diameter, charge, mass).
        These modifiers run at the end of the configuration generation process.
        Possible choices can be found in the colloids_create.final_modifiers module.
        If final modifiers are specified, their parameters must be specified as well.
        Defaults to None.
    :type final_modifiers: Optional[list[str]]
    :param final_modifiers_parameters:
        List of dictionaries containing parameters for each final modifier.
        Each dictionary is passed to the corresponding modifier's __init__ method.
        The list must have the same length as final_modifiers.
        Defaults to None.
    :type final_modifiers_parameters: Optional[list[dict[str, Any]]]

    :raises TypeError:
        If the masses, radii, or surface potentials do not have the correct units.
        If the masses, radii, or surface potentials dictionaries do not have strings as keys.
    :raises ValueError:
        If the configuration generator is not found in the available generators.
        If the masses are not greater than or equal to zero.
        If the radii are not greater than zero.
        If an initial or final modifier is not found in the available modifiers.
        If initial_modifiers is specified but initial_modifiers_parameters is not, or vice versa.
        If final_modifiers is specified but final_modifiers_parameters is not, or vice versa.
        If the number of (initial or final) modifiers does not match the number of parameter dictionaries.
    """
    configuration_generator: str = "ClusterGenerator"
    configuration_generator_parameters: dict[str, Any] = field(default_factory=lambda: {
        "cluster_specifications": ["cluster.lmp"],
        "cluster_relative_weights": [1.0],
        "lattice_repeats": 8,
        "cluster_padding_factor": 1.0,
        "padding_factor": 1.0,
        "random_rotation": False,
    })
    masses: dict[str, unit.Quantity] = field(default_factory=lambda: {"1": 1.0 * mass_unit,
                                                                      "2": (95.0 / 105.0) ** 3 * mass_unit})
    radii: dict[str, unit.Quantity] = field(default_factory=lambda: {"1": 105.0 * length_unit,
                                                                     "2": 95.0 * length_unit})
    surface_potentials: dict[str, unit.Quantity] = field(
        default_factory=lambda: {"1": 44.0 * electric_potential_unit, "2": -54.0 * electric_potential_unit})
    initial_modifiers: Optional[list[str]] = None
    initial_modifiers_parameters: Optional[list[dict[str, Any]]] = None
    final_modifiers: Optional[list[str]] = None
    final_modifiers_parameters: Optional[list[dict[str, Any]]] = None

    def __post_init__(self):
        """Post-initialization method for the ConfigurationParameters class."""

        possible_generators = [name for name, obj in inspect.getmembers(configuration_generators, inspect.isclass)
                               if issubclass(obj, ConfigurationGenerator)
                               and obj is not ConfigurationGenerator]
        if self.configuration_generator not in possible_generators:
            raise ValueError(f"Configuration generator {self.configuration_generator} not found. "
                             f"Possible choices are: {', '.join(possible_generators)}.")

        for t in self.masses:
            if not isinstance(t, str):
                raise TypeError("The types of the masses dictionary must be strings.")
            if not self.masses[t].unit.is_compatible(mass_unit):
                raise TypeError(f"Mass of type {t} must have a unit compatible with atomic mass units.")
            if self.masses[t] < 0.0 * mass_unit:
                raise ValueError(f"Mass of type {t} must be greater than zero.")
            if t not in self.radii:
                raise ValueError(f"Type {t} of the masses dictionary is not in radii dictionary.")
            if t not in self.surface_potentials:
                raise ValueError(f"Type {t} of the masses dictionary is not in surface potentials dictionary.")
        for t in self.radii:
            if not isinstance(t, str):
                raise TypeError("The types of the radii dictionary must be strings.")
            if not self.radii[t].unit.is_compatible(length_unit):
                raise TypeError(f"Radius of type {t} must have a unit compatible with nanometers.")
            if self.radii[t] <= 0.0 * length_unit:
                raise ValueError(f"Radius of type {t} must be greater than zero.")
            if t not in self.masses:
                raise ValueError(f"Type {t} of the radii dictionary is not in masses dictionary.")
            if t not in self.surface_potentials:
                raise ValueError(f"Type {t} of the radii dictionary is not in surface potentials dictionary.")
        for t in self.surface_potentials:
            if not isinstance(t, str):
                raise TypeError("The types of the surface potentials dictionary must be strings.")
            if not self.surface_potentials[t].unit.is_compatible(electric_potential_unit):
                raise TypeError(f"Surface potential of type {t} must have a unit compatible with millivolts.")
            if t not in self.masses:
                raise ValueError(f"Type {t} of the surface potentials dictionary is not in masses dictionary.")
            if t not in self.radii:
                raise ValueError(f"Type {t} of the surface potentials dictionary is not in radii dictionary.")

        if self.initial_modifiers is not None:
            if self.initial_modifiers_parameters is None:
                raise ValueError("The initial_modifiers_parameters must be specified if initial_modifiers is specified.")
            if len(self.initial_modifiers) != len(self.initial_modifiers_parameters):
                raise ValueError("The number of initial modifiers must match the number of initial modifier "
                                 "parameter dictionaries.")
            possible_modifiers = [name for name, obj in inspect.getmembers(initial_modifiers, inspect.isclass)
                                  if issubclass(obj, initial_modifiers.InitialModifier)
                                  and obj is not initial_modifiers.InitialModifier]
            for initial_modifier in self.initial_modifiers:
                if initial_modifier not in possible_modifiers:
                    raise ValueError(f"Initial modifier {initial_modifier} not found. Possible choices are: "
                                     f"{', '.join(possible_modifiers)}.")
        else:
            if self.initial_modifiers_parameters is not None:
                raise ValueError("The initial_modifiers_parameters must not be specified if initial_modifiers is not "
                                 "specified.")

        if self.final_modifiers is not None:
            if self.final_modifiers_parameters is None:
                raise ValueError("The final_modifiers_parameters must be specified if final_modifiers is specified.")
            if len(self.final_modifiers) != len(self.final_modifiers_parameters):
                raise ValueError("The number of final modifiers must match the number of final modifier "
                                 "parameter dictionaries.")
            possible_modifiers = [name for name, obj in inspect.getmembers(final_modifiers, inspect.isclass)
                                  if issubclass(obj, final_modifiers.FinalModifier)
                                  and obj is not final_modifiers.FinalModifier]
            for final_modifier in self.final_modifiers:
                if final_modifier not in possible_modifiers:
                    raise ValueError(f"Final modifier {final_modifier} not found. Possible choices are: "
                                     f"{', '.join(possible_modifiers)}.")
        else:
            if self.final_modifiers_parameters is not None:
                raise ValueError("The final_modifiers_parameters must not be specified if final_modifiers is not "
                                 "specified.")
