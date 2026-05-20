from typing import Optional
from abc import abstractmethod, ABC

import openmm
import openmm.app

from colloids.helper_functions import get_cell_from_box
from colloids.units import length_unit

from openmm import CustomGBForce, CustomCVForce
import freud

_EPSILON = 1e-12

##Helper functions for implementation of Steinhardt order parameter surrogates

def _real_harmonic_definitions(x="x", y="y", z="z", r="r", l=6):
    """Return unnormalized real spherical-harmonic polynomial definitions specified by angular momentum l."""

    if l==4:
        defs = (
            f"X = {x}/({r}+eps);"
            f"Y = {y}/({r}+eps);"
            f"Z = {z}/({r}+eps);"
            "Y40 = 35*Z^4 - 30*Z^2 + 3;"
            "Y41c = X*Z*(7*Z^2 - 3);"
            "Y41s = Y*Z*(7*Z^2 - 3);"
            "Y42c = (X^2 - Y^2)*(7*Z^2 - 1);"
            "Y42s = 2*X*Y*(7*Z^2 - 1);"
            "Y43c = X*Z*(X^2 - 3*Y^2);"
            "Y43s = Y*Z*(3*X^2 - Y^2);"
            "Y44c = X^4 - 6*X^2*Y^2 + Y^4;"
            "Y44s = 4*X*Y*(X^2 - Y^2)"
        )
    elif l==6:
        defs = ()

    else:
        raise ValueError(f"Spherical harmonic definitions not implemented for angular momentum number {l}.")
        defs = None
    
    return defs


def _ql_component_names(l=6):

    if l==4:
        names = [
            "Y40",
            "Y41c",
            "Y41s",
            "Y42c",
            "Y42s",
            "Y43c",
            "Y43s",
            "Y44c",
            "Y44s",
        ]
    elif l==6:
        names = []
    
    else:
        raise ValueError(f"Spherical harmonic definitions not implemented for angular momentum number {l}.")
        names = None

    return names

def _ql_magnitude_expression():
    sum_sq = " + ".join(f"ql_{component}^2" for component in _ql_component_names())
    return f"sqrt(({sum_sq}) / (ql_neighbor_weight^2 + eps))"


class SwitchingFunctions(ABC):

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
    def __init__(self, topology=openmm.app.Topology, system=openmm.app.System):
        pass
    
    @abstractmethod
    def compute_cv(self) -> openmm.Force:
        raise NotImplementedError

    @abstractmethod
    def get_force(self) -> openmm.CustomCVForce:
        raise NotImplementedError


class SteinhardtOrderParameterCV(CollectiveVariableAbstract):
    def __init__(self, topology=openmm.app.Topology, system=openmm.app.System, l=6, included_particles=None,
            coordination_d0=300.0 * length_unit, coordination_r0=50.0 * length_unit, coordination_dmax=400.0 * length_unit,
            ql_d0=230.0 * length_unit, ql_r0=10.0 * length_unit, ql_dmax=260.0 * length_unit,
            highcoord_threshold=12.0, highcoord_nn=12):
        
        self._particle_types = [atom.name for atom in topology.atoms()]
        self._included_particles = included_particles
        self._uses_pbc = system.usesPeriodicBoundaryConditions()
        self._coordination_d0 = coordination_d0
        self._coordination_r0 = coordination_r0
        self._coordination_dmax = coordination_dmax
        self._ql_d0 = ql_d0
        self._ql_r0 = ql_r0
        self._ql_dmax = ql_dmax
        self._highcoord_threshold = highcoord_threshold
        self._highcoord_nn = highcoord_nn
        self._include_particles = [
            1 if included_particles is None or particle_type in included_particles else 0
            for particle_type in self._particle_types
        ]
    
    def compute_cv(self, name, include_particles, l, cv_type):

        force = CustomGBForce()
        force.setName(name)
        
        force.addGlobalParameter("coordination_d0", self._coordination_d0.value_in_unit(length_unit))
        force.addGlobalParameter("coordination_r0", self._coordination_r0.value_in_unit(length_unit))
        force.addGlobalParameter("coordination_dmax", self._coordination_dmax.value_in_unit(length_unit))
        force.addGlobalParameter("ql_d0", self._ql_d0.value_in_unit(length_unit))
        force.addGlobalParameter("ql_r0", self._ql_r0.value_in_unit(length_unit))
        force.addGlobalParameter("ql_dmax", self._ql_dmax.value_in_unit(length_unit))
        force.addGlobalParameter("highcoord_threshold", self._highcoord_threshold)
        force.addGlobalParameter("nn", self._highcoord_nn)
        force.addGlobalParameter("eps", _EPSILON)
        force.addPerParticleParameter("include_particles")

        for include_particle in self._include_particles:
            force.addParticle([include_particle])
        if self._uses_pbc:
            force.setNonbondedMethod(CustomGBForce.CutoffPeriodic)
        else:
            force.setNonbondedMethod(CustomGBForce.CutoffNonPeriodic)
        
        force.setCutoffDistance(max(
            self._coordination_dmax.value_in_unit(length_unit),
            self._ql_dmax.value_in_unit(length_unit),
        ))
        
        coordination_switch = SwitchingFunctions.exponential(
            r="r", d0="coordination_d0", r0="coordination_r0", dmax="coordination_dmax")
        
        highcoord = SwitchingFunctions.more_than(
            x="coord", threshold="highcoord_threshold", nn="nn")
        
        
        if cv_type == 'highcoord':
            force.addComputedValue("coord", f"include_particles2 * ({coordination_switch})",
                                CustomGBForce.ParticlePairNoExclusions,)

            force.addEnergyTerm(
                f"include_particles * ({highcoord})",
                CustomGBForce.SingleParticle,
            )
        
        elif cv_type == 'order':

            ql_switch = SwitchingFunctions.exponential(
                r="r", d0="ql_d0", r0="ql_r0", dmax="ql_dmax")
        
            ql_i = _ql_magnitude_expression()

            force.addComputedValue(
                "coord",
                f"include_particles2 * ({coordination_switch})",
                CustomGBForce.ParticlePairNoExclusions)

            harmonics = _real_harmonic_definitions(x="x", y="y", z="z", r="r", l=l)
            for component in _ql_component_names():
                force.addComputedValue(
                    f"ql_{component}",
                    f"include_particles2 * ({ql_switch}) * {component};{harmonics}",
                    CustomGBForce.ParticlePairNoExclusions,
                )
            force.addComputedValue(
                "q4_neighbor_weight",
                f"include_particles2 * ({ql_switch})",
                CustomGBForce.ParticlePairNoExclusions,
            )
            
            force.addEnergyTerm(
                f"include_particles * highcoord * ql_i;"
                f"ql_i = {ql_i};"
                f"highcoord = {highcoord}",
                CustomGBForce.SingleParticle,
            )
            
        return force

    def get_force(self, l):
        ql_cv = CustomCVForce(
            f"ql_highcoord;"
            f"ql_highcoord = ql_numerator / (highcoord_count + eps)"
        )
        ql_cv.setName(f"q{l}_highcoord")
        ql_cv.addGlobalParameter("eps", _EPSILON)
        ql_cv.addCollectiveVariable("ql_numerator", self.get_cv_force(l, cv_type="order"))
        ql_cv.addCollectiveVariable("highcoord_count", self.get_cv_force(l, cv_type="highcoord"))
        return ql_cv


class HighCoordCompositionCV(CollectiveVariableAbstract):

    def __init__(self, topology: openmm.app.Topology, system: openmm.System, positive_type_name="2", negative_type_name="1",
                 ignore_types=None, coordination_d0 = 300.0 * length_unit, coordination_r0 =  50.0 * length_unit, coordination_dmax = 400.0 * length_unit,
                 nn_switch = 12, highcoord_threshold = 12.0):
        super.__init__()
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

        s_r_exp= SwitchingFunctions.get_exponential_switching_function_str("r", d0 = "coordination_d0",
                                                                 r0 = "coordination_r0", dmax = "coordination_dmax")
        high_coord = SwitchingFunctions.get_more_than_str(x="coord", threshold="highcoord_threshold", nn="nn")

        force.addComputedValue(
            "coord",
            f"include_particles2 * ({s_r_exp})",
            CustomGBForce.ParticlePairNoExclusions,
        )

        force.addEnergyTerm(
            f"{type_parameter} * ({high_coord})",
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

    def get_force(self):

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