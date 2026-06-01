from typing import Optional, Sequence, Callable
from abc import abstractmethod, ABC
from dataclasses import dataclass

import openmm
import openmm.app
from openmm import unit

from colloids.helper_functions import get_cell_from_box
from colloids.units import length_unit

from openmm import CustomGBForce, CustomCVForce


_EPSILON = 1e-12

    
class SwitchingFunctions:

    @staticmethod
    def get_exponential_switching_function_str(r="r", d0="d0", r0="r0", dmax=None):
        s_r =  f"1 / (1 + exp(({r} - {d0}) / {r0}))"

        if dmax is not None:
            s_r = f"step({dmax} - {r}) * ({s_r})"

        return s_r
    
    @staticmethod
    def get_rational_switching_function_str(nn=12, mm=24, r="r", d0="d0", r0="r0"):
        
        x = f"({r} - {d0}) / {r0}"
        
        s_r = f"(1 - ({x})^{nn}) / (1 - ({x})^{mm} + eps)"

        return s_r

    @staticmethod
    def get_more_than_str(x="x", threshold="threshold", nn=12):

        mt = f"({x})^{nn}/(({threshold})^{nn} + ({x})^{nn} + eps)"

        return mt
    

class CollectiveVariableAbstract(ABC):
    """
    Abstract class for the implementation of custom collective variable forces in OpenMM.
    
    :param system:
        The OpenMM system that this force will be added to.
        The system is used to check the periodic boundary conditions.
    :type simulation: openmm.System
    :param topology:
        The topology of the OpenMM system. This is used to check the particle types. 
    :type topology: openmm.app.Topology
    """

    def __init__(self, topology=openmm.app.Topology, system=openmm.System):
        pass
    
    @abstractmethod
    def compute_cv(self) -> openmm.Force:
        raise NotImplementedError

    @abstractmethod
    def get_force(self) -> openmm.CustomCVForce:
        raise NotImplementedError
    
class HighCoordCompositionCV(CollectiveVariableAbstract):
    
    """
    Custom collective variable to determine particles of high coordination and compute the number ratio of a specified 
    target particle type among these coordinated particles. A switching function is applied to make the CV smooth and 
    differentiable.

    :param topology:
        The topology of the OpenMM system. 
    :type topology: openmm.app.Topology
    :param system:
        The OpenMM system that this force will be added to.
    :type simulation: openmm.System
    :param target_particle_type: 
        The particle type name for which the number ratio is being computed. The number of particles of this type
        that satisfy the high coordination requirement will become the numerator of this custom CV expression.
    :type target_particle_type: str
    :param coordination_r0: 
        The decay length of the exponential switching function.
    :type coordination_r0: unit.Quantity
    :param coordination_d0:
        The offset distance (midpoint) of the exponential switching function.
    :type coordination_d0: unit.Quantity
    :param coordination_dmax:
        The max cutoff distance of the exponential switching function. Beyond this distance,
        the value of the switch is zero.
    :type coordination_dmax: unit.Quantity
    :param highcoord_threshold: 
        The minimum coordination number particles must have to be counted. 
    :type highcoord_threshold: float
    :param ignore_types:
        A list of particle types to ignore when counting high-coordination particles.
        If None, all particle types are considered.
        Defaults to None.
    :type ignore_types: Optional[Sequence[str]] 

    :raises TypeError:
        If coordination_r0, coordination_d0, or coordination_dmax are not Quantities with 
        proper length units.
    :raises ValueError:
        If coordination_r0, coordination_d0, or coordination_dmax are not greater than 0.
        If the highcoord_threshold is not greater than 0.
        If the target_particle_type is not a valid particle type in the OpenMM system.     
    """

    def __init__(self, topology: openmm.app.Topology, system: openmm.System, target_particle_type: str, coordination_d0: unit.Quantity, 
                coordination_r0: unit.Quantity, coordination_dmax: unit.Quantity, highcoord_threshold: float, 
                ignore_types: Optional[Sequence[str]] = None):
        
        self._uses_pbc = system.usesPeriodicBoundaryConditions()
        self._particle_types = [atom.name for atom in topology.atoms()]
        if target_particle_type not in self._particle_types:
            raise ValueError("The target particle for which high coordination",
            "count is being determined is not in the OpenMM system")
        
        if highcoord_threshold <=0:
            raise ValueError("The high coordination threshold must be a positive value.")
        if not all(param.unit.is_compatible(length_unit) for param in (coordination_r0, coordination_d0, coordination_dmax)):
            raise TypeError("Switching function parameters must all have a unit that is compatible with nanometers")
        if coordination_r0.value_in_unit(length_unit) <=0 or coordination_d0.value_in_unit(length_unit) <=0 or coordination_dmax.value_in_unit(length_unit) <=0: 
            raise ValueError("Switching function parameters must all be positive values")

        self._nn_high_coord = 12
        self._highcoord_threshold = highcoord_threshold
        
        self._is_target = []
        self._target_particle_type = target_particle_type
    
        self._ignore_types = ignore_types
        self._include_particles = []
        
        for particle_type in self._particle_types:
            if self._ignore_types is None:
                include_particle = 1
            else:
                include_particle = int(particle_type not in self._ignore_types)

            self._include_particles.append(include_particle)
            self._is_target.append(int(particle_type == self._target_particle_type))

        
        self._coordination_d0 = coordination_d0
        self._coordination_r0 =  coordination_r0
        self._coordination_dmax = coordination_dmax


    def compute_cv(self, name: str, weight_parameter: str):
        force = CustomGBForce()
        force.setName(name)

        force.addGlobalParameter(
            "coordination_d0",
            self._coordination_d0.value_in_unit(length_unit),
        )
        force.addGlobalParameter(
            "coordination_r0",
            self._coordination_r0.value_in_unit(length_unit),
        )
        force.addGlobalParameter(
            "coordination_dmax",
            self._coordination_dmax.value_in_unit(length_unit),
        )

        force.addGlobalParameter("highcoord_threshold", self._highcoord_threshold)
        force.addGlobalParameter("nn", self._nn_high_coord)
        force.addGlobalParameter("eps", _EPSILON)

        force.addPerParticleParameter("include_particles")
        force.addPerParticleParameter("is_target")

        s_r_exp= SwitchingFunctions.get_exponential_switching_function_str("r", d0 = "coordination_d0",
                                                                 r0 = "coordination_r0", dmax = "coordination_dmax")
        
        high_coord = SwitchingFunctions.get_more_than_str(x="coord", threshold="highcoord_threshold", nn="nn")

        force.addComputedValue(
            "coord",
            f"include_particles2 * ({s_r_exp})",
            CustomGBForce.ParticlePairNoExclusions,
        )

        force.addEnergyTerm(
            f"{weight_parameter} * ({high_coord})",
            CustomGBForce.SingleParticle,
            )


        for include_particle, is_target in zip(self._include_particles,
                                                self._is_target):
            force.addParticle([include_particle, is_target])

        if self._uses_pbc:
            force.setNonbondedMethod(CustomGBForce.CutoffPeriodic)
        else:
            force.setNonbondedMethod(CustomGBForce.CutoffNonPeriodic)

        force.setCutoffDistance(
            self._coordination_dmax.value_in_unit(length_unit)
            )

        return force

    def get_force(self):

        target_highcoord = self.compute_cv(
            name="target_highcoord",
            weight_parameter="include_particles * is_target",
            )

        all_highcoord = self.compute_cv(
                name="all_highcoord",
            weight_parameter="include_particles",
            )
        
        x_force = CustomCVForce("x_i;"
                                 "x_i = target_highcoord / (all_highcoord + eps)"
                                 )
        
        x_force.setName("highcoord_composition_cv")

        x_force.addGlobalParameter("eps", _EPSILON)
        x_force.addCollectiveVariable("target_highcoord", target_highcoord)
        x_force.addCollectiveVariable("all_highcoord", all_highcoord)
        
        return x_force
    
class XPositionCV(CollectiveVariableAbstract):
    """Simple scalar CV for tests: x position of one particle."""

    def __init__(self, topology: openmm.app.Topology, system: openmm.System, particle_index: int = 0):
        self._particle_index = particle_index

    def compute_cv(self) -> openmm.Force:
        force = openmm.CustomExternalForce("x")
        force.setName("x_position_cv")
        force.addParticle(self._particle_index, [])
        return force

    def get_force(self) -> openmm.Force:
        return self.compute_cv()