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
        defs = (
            f"X = {x}/({r}+eps);"
            f"Y = {y}/({r}+eps);"
            f"Z = {z}/({r}+eps);"
            "Y60 = 231*Z^6 - 315*Z^4 + 105*Z^2 - 5;"
            "Y61c = X*Z*(33*Z^4 - 30*Z^2 + 5);"
            "Y61s = Y*Z*(33*Z^4 - 30*Z^2 + 5);"
            "Y62c = (X^2 - Y^2)*(33*Z^4 - 18*Z^2 + 1);"
            "Y62s = 2*X*Y*(33*Z^4 - 18*Z^2 + 1);"
            "Y63c = X*Z*(X^2 - 3*Y^2)*(11*Z^2 - 3);"
            "Y63s = Y*Z*(3*X^2 - Y^2)*(11*Z^2 - 3);"
            "Y64c = (X^4 - 6*X^2*Y^2 + Y^4)*(11*Z^2 - 1);"
            "Y64s = 4*X*Y*(X^2 - Y^2)*(11*Z^2 - 1);"
            "Y65c = X*Z*(X^4 - 10*X^2*Y^2 + 5*Y^4);"
            "Y65s = Y*Z*(5*X^4 - 10*X^2*Y^2 + Y^4);"
            "Y66c = X^6 - 15*X^4*Y^2 + 15*X^2*Y^4 - Y^6;"
            "Y66s = 2*X*Y*(3*X^4 - 10*X^2*Y^2 + 3*Y^4)"
        )

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
        names = [
            "Y60",
            "Y61c",
            "Y61s",
            "Y62c",
            "Y62s",
            "Y63c",
            "Y63s",
            "Y64c",
            "Y64s",
            "Y65c",
            "Y65s",
            "Y66c",
            "Y66s",
        ]
    
    else:
        raise ValueError(f"Spherical harmonic definitions not implemented for angular momentum number {l}.")
        names = None

    return names

def _ql_magnitude_expression():
    sum_sq = " + ".join(f"ql_{component}^2" for component in _ql_component_names())
    return f"sqrt(({sum_sq}) / (ql_neighbor_weight^2 + eps))"

class SteinhardtOrderParameterCV(CollectiveVariableAbstract):
    def __init__(self, topology=openmm.app.Topology, system=openmm.System, l=6, included_particles=None,
            coordination_d0=300.0 * length_unit, coordination_r0=50.0 * length_unit, coordination_dmax=400.0 * length_unit,
            ql_d0=230.0 * length_unit, ql_r0=10.0 * length_unit, ql_dmax=260.0 * length_unit,
            highcoord_threshold=12.0, highcoord_nn=12):
        
        self._particle_types = [atom.name for atom in topology.atoms()]
        self._included_particles = included_particles
        self._uses_pbc = system.usesPeriodicBoundaryConditions()
        self._coordination_d0 = coordination_d0
        self._coordination_r0 = coordination_r0
        self._coordination_dmax = coordination_dmax
        self._l = l
        self._ql_d0 = ql_d0
        self._ql_r0 = ql_r0
        self._ql_dmax = ql_dmax
        self._highcoord_threshold = highcoord_threshold
        self._highcoord_nn = highcoord_nn
        self._include_particles = [
            1 if included_particles is None or particle_type in included_particles else 0
            for particle_type in self._particle_types
        ]
    
    def compute_cv(self, name, l, cv_type):

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
        
        coordination_switch = SwitchingFunctions.get_exponential_switching_function_str(
            r="r", d0="coordination_d0", r0="coordination_r0", dmax="coordination_dmax")
        
        highcoord = SwitchingFunctions.get_more_than_str(
            x="coord", threshold="highcoord_threshold", nn="nn")
        
        
        if cv_type == 'highcoord':
            force.addComputedValue("coord", f"include_particles2 * ({coordination_switch})",
                                CustomGBForce.ParticlePairNoExclusions,)

            force.addEnergyTerm(
                f"include_particles * ({highcoord})",
                CustomGBForce.SingleParticle,
            )
        
        elif cv_type == 'order':

            ql_switch = SwitchingFunctions.get_exponential_switching_function_str(
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

    def get_force(self):
        ql_cv = CustomCVForce(
            f"ql_highcoord;"
            f"ql_highcoord = ql_numerator / (highcoord_count + eps)"
        )
        ql_cv.setName(f"q{self._l}_highcoord")
        ql_cv.addGlobalParameter("eps", _EPSILON)
        ql_cv.addCollectiveVariable("ql_numerator", self.compute_cv("ql_numerator", self._l, cv_type="order"))
        ql_cv.addCollectiveVariable("highcoord_count", self.compute_cv("highcoord_count", self._l, cv_type="highcoord"))
        return ql_cv
