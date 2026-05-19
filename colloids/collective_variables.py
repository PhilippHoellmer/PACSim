from typing import Optional
from abc import abstractmethod, ABC

import numpy as np
import numpy.typing as npt

import openmm
import openmm.app

from colloids.helper_functions import get_cell_from_box
from colloids.units import length_unit

from openmm import CustomGBForce, CustomCVForce
import freud

_EPSILON = 1e-12


def _minimum_image_displacements(positions: npt.NDArray[np.float64],
                                 box_lengths: Optional[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    displacements = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    if box_lengths is not None:
        displacements -= box_lengths * np.rint(displacements / box_lengths)
    return displacements


class CollectiveVariableAbstract(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def compute_cv(self) -> openmm.CustomGBForce:
        pass

    @abstractmethod
    def get_force(self) -> openmm.CustomCVForce:
        pass


class SteinhardtOrderParameterCV(CollectiveVariableAbstract):
    pass

class HighCoordCompositionCV(CollectiveVariableAbstract):

    def __init__(self, topology: openmm.app.Topology, system: openmm.System, positive_type_name="2", negative_type_name="1",
                 ignore_types=None, coordination_d0 = 300.0 * length_unit, coordination_r0 =  50.0 * length_unit, coordination_dmax = 400.0 * length_unit,
                 nn_switch = 12, highcoord_threshold = 12.0):
        self._coordination_d0 = coordination_d0
        self._coordination_r0 =  coordination_r0
        self._coordination_dmax = coordination_dmax
        self._nn_switch = nn_switch
        self._highcoord_threshold = highcoord_threshold

        self._positive_type_name = positive_type_name
        self._negative_type_name = negative_type_name
        self._ignore_types = ignore_types

        self._include_particles = []
        self._is_P = []
        self._is_N = []

        self._particle_types = [atom.name for atom in topology.atoms()]
        self._uses_pbc = system.usesPeriodicBoundaryConditions()

        for particle_type in self._particle_types:
            if self._ignore_types is None:
                include_particle = 1
            else:
                include_particle = int(particle_type not in self._ignore_types)

            self._include_particles.append(include_particle)
            self._is_P.append(int(particle_type == self._positive_type_name))
            self._is_N.append(int(particle_type == self._negative_type_name))
                
     
    def compute_cv(self, name, type_parameter):
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
        force.addGlobalParameter("nn", self._nn_switch)
        force.addGlobalParameter("eps", _EPSILON)

        force.addPerParticleParameter("include_particles")
        force.addPerParticleParameter("is_P")
        force.addPerParticleParameter("is_N")

        force.addComputedValue(
            "coord",
            "include_particles2 * step(coordination_dmax-r) / "
            "(1 + exp((r-coordination_d0)/coordination_r0))",
            CustomGBForce.ParticlePairNoExclusions,
        )

        force.addEnergyTerm(
            f"{type_parameter} * highcoord;"
            "highcoord = coord^nn / (highcoord_threshold^nn + coord^nn + eps)",
            CustomGBForce.SingleParticle,
        )

        for include_particle, is_P, is_N in zip(self._include_particles,
                                                self._is_P,
                                                self._is_N):
            force.addParticle([include_particle, is_P, is_N])

        if self._uses_pbc:
            force.setNonbondedMethod(CustomGBForce.CutoffPeriodic)
        else:
            force.setNonbondedMethod(CustomGBForce.CutoffNonPeriodic)

        force.setCutoffDistance(
            self._coordination_dmax.value_in_unit(length_unit)
            )

        return force

    def get_force(self, parameters):

        p_highcoord = self.compute_cv(name="p_highcoord", type_parameter="is_P")

        n_highcoord = self.compute_cv(name="n_highcoord", type_parameter="is_N")

        xP_force = CustomCVForce("x_P;"
                                 "x_P = p_highcoord / (p_highcoord + n_highcoord + eps)"
                                 )
        
        xP_force.setName("xP_crystal")

        xP_force.addGlobalParameter("eps", _EPSILON)
        xP_force.addCollectiveVariable("p_highcoord", p_highcoord)
        xP_force.addCollectiveVariable("n_highcoord", n_highcoord)
        
        return xP_force