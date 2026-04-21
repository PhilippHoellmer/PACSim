from abc import ABC, abstractmethod
from gsd.hoomd import Frame
from openmm import unit


class ConfigurationModifier(ABC):
    """
    Abstract base class for a modifier of an existing configuration in a gsd.hoomd.Frame instance for a colloid
    simulation.
    """

    def __init__(self) -> None:
        """Constructor of the ConfigurationModifier class."""
        pass

    @abstractmethod
    def modify_configuration(self, frame: Frame) -> None:
        """
        Modify the given configuration and constraints in-place.

        :param frame:
            The frame to modify.
        :type frame: gsd.hoomd.Frame
        """
        raise NotImplementedError


class InitialModifier(ConfigurationModifier):
    """
    Abstract base class for a modifier of an existing configuration in a Frame instance for a colloid simulation that is
    applied right after the initial configuration generation.

    This modifier is intended to be used before any particle attributes (like type_shapes, diameter, charge, mass) are
    set in the frame, and it should not modify these attributes.

    This modifier receives the masses, radii, and surface potentials dictionaries to be able to perform checks on the
    types of particles in the frame if necessary.

    :param masses:
        The masses dictionary with the particle types as keys and the masses as values.
    :type masses: dict[str, unit.Quantity]
    :param radii:
        The radii dictionary with the particle types as keys and the radii as values.
    :type radii: dict[str, unit.Quantity]
    :param surface_potentials:
        The surface potentials dictionary with the particle types as keys and the surface potentials as values.
    :type surface_potentials: dict[str, unit.Quantity]
    """

    # noinspection PyUnusedLocal
    def __init__(self, masses: dict[str, unit.Quantity], radii: dict[str, unit.Quantity],
                 surface_potentials: dict[str, unit.Quantity]) -> None:
        """Constructor of the InitialModifier class."""
        super().__init__()

    @abstractmethod
    def modify_configuration(self, frame: Frame) -> None:
        """
        Modify the given configuration and constraints in-place (for instance, by adding a substrate).

        The overloading method should only modify the following attributes of the given frame:
        - frame.particles.N
        - frame.particles.position
        - frame.particles.types
        - frame.particles.typeid
        - frame.configuration.box
        - frame.constraints.N (optionally if constraints are present)
        - frame.constraints.value (optionally if constraints are present)
        - frame.constraints.group (optionally if constraints are present)

        The overloading method should not modify the following attributes of the given frame:
        - frame.particles.type_shapes
        - frame.particles.diameter
        - frame.particles.charge
        - frame.particles.mass attributes

        :param frame:
            The frame to modify.
        :type frame: Frame
        """
        raise NotImplementedError


class FinalModifier(ConfigurationModifier):
    """
    Abstract base class for a modifier of an existing configuration in a Frame instance for a colloid simulation that is
    applied right before finalizing the configuration.

    This modifier is intended to be used after all particle attributes (like type_shapes, diameter, charge, mass) are
    set, and it may modify these attributes if necessary.
    """

    def __init__(self) -> None:
        """Constructor of the FinalModifier class."""
        super().__init__()

    @abstractmethod
    def modify_configuration(self, frame: Frame) -> None:
        """
        Modify the given configuration and constraints in-place (for instance, by adding a seed).

        The overloading method may modify the following attributes of the given frame:
        - frame.particles.N
        - frame.particles.position
        - frame.particles.types
        - frame.particles.typeid
        - frame.configuration.box
        - frame.constraints.N (optionally if constraints are present)
        - frame.constraints.value (optionally if constraints are present)
        - frame.constraints.group (optionally if constraints are present)
        - frame.particles.type_shapes
        - frame.particles.diameter
        - frame.particles.charge
        - frame.particles.mass attributes

        :param frame:
            The frame to modify.
        :type frame: Frame
        """
        raise NotImplementedError
