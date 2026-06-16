import math
import numpy as np
from typing import Iterator
from openmm import unit
from openmm import CustomNonbondedForce
from colloids.abstracts import OpenMMNonbondedPotentialAbstract
from colloids.units import energy_unit, length_unit, temperature_unit

try:
    import torch
    import torch.nn as _nn
    from torchmdnet.models.model import load_model as _load_model
    from openmmtorch import TorchForce as _TorchForce
    _ML_AVAILABLE = True

    class _MLDepletionModule(_nn.Module):
        def __init__(self, mlp_name: str, ptypes, r_d: float, kT: float):
            super().__init__()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.mlp = _load_model(mlp_name, derivative=True, device=device)
            self.ptypes = torch.tensor(ptypes, dtype=torch.long, device=device)
            self.kT: float = kT
            # Converts volume fraction phi to number density (µm^-3) at runtime.
            self.phi_scale: float = 1.0 / (4.0 / 3.0 * math.pi * (r_d / 1000.0) ** 3)

        def forward(self, positions: torch.Tensor, phi: float) -> torch.Tensor:
            positions = positions.float() / 1000.0
            excluded_vol, _ = self.mlp.forward(self.ptypes, positions)
            return -self.kT * phi * self.phi_scale * excluded_vol

except ImportError:
    _ML_AVAILABLE = False


class DepletionPotential(OpenMMNonbondedPotentialAbstract):
    """
    This class sets up the depletion potential between colloids in a solution with a non-adsorbing polymer background.
    Since the attractive force arises from the fact that the polymer molecules are depleted at the surface of the
    colloids, the force is called the depletion force. The depletion force is well-modeled by the Asakura-Oosawa
    potential. To completely describe the pair potentials in a system of colloids within solution of non-adsorbing
    polymers, this attractive depletion force can be paired with the steric and electrostatic forces between the ionic
    colloids from the Alexander-de Gennes polymer brush model and DLVO theory.

    The cutoff distance for the depletion potential is set to max(sigma_colloid) + sigma_depletant where sigma_colloid
    is the diameter of the largest particle in the system plus two lengths of the polymer brush, and sigma_depletant is
    the diameter of the depletant. The cutoff can be set to be periodic or non-periodic.

    :param depletion_phi:
        The number density of polymers in the solution.
        The value must be between 0 and 1.
    :type depletion_phi: float
    :param depletant_radius:
        The "radius" of the depletants, if treated as hard spheres.
        The unit of the depletant_radius must be compatible with nanometers and the value must be greater than zero.
    :type depletant_radius: unit.Quantity
    :param brush_length:
        The thickness of the polymer brush as described by the Alexander-de Gennes polymer brush model.
        (See ColloidPotentialsParameters class for more information.)
        The unit of the brush_length must be compatible with nanometers and the value must be greater than zero.
        Defaults to 10.0 nanometers.
    :type brush_length: unit.Quantity
    :param periodic_boundary_conditions:
        Whether this force should use periodic cutoffs for the depletion potential.
    :type periodic_boundary_conditions: bool
    """

    _name = "depletion_energy"

    def __init__(self, depletion_phi: float, depletant_radius: unit.Quantity, brush_length: unit.Quantity,
                 temperature: unit.Quantity, periodic_boundary_conditions: bool = True):
        """Constructor of the DepletionPotential class."""
        super().__init__()

        if not 0.0 <= depletion_phi <= 1.0:
            raise ValueError("phi must be between zero and one")
        if not depletant_radius.unit.is_compatible(length_unit):
            raise TypeError("depletant radius must have a unit compatible with nanometers")
        if depletant_radius <= 0.0 * length_unit:
            raise ValueError("depletant radius must be greater than zero")
        if not brush_length.unit.is_compatible(length_unit):
            raise TypeError("brush length must have a unit compatible with nanometers")
        if brush_length <= 0.0 * length_unit:
            raise ValueError("brush length must be greater than zero")
        if not temperature.unit.is_compatible(temperature_unit):
            raise TypeError("argument temperature must have a unit that is compatible with kelvin")
        if not temperature.value_in_unit(temperature_unit) > 0.0:
            raise ValueError("argument temperature must have a value greater than zero")

        self._depletion_phi = depletion_phi
        self._depletant_radius = depletant_radius
        self._brush_length = brush_length
        self._temperature = temperature
        self._periodic_boundary_conditions = periodic_boundary_conditions
        self._max_radius = -math.inf * length_unit
        self._depletion_potential = self._set_up_depletion_potential()

    def _set_up_depletion_potential(self) -> CustomNonbondedForce:
        """Set up the basic functional form of the Asakura-Oosawa depletion potential for a solution of binary colloidal 
        particles in a background of non-adsorbing polymers."""
        depletion_potential = CustomNonbondedForce(
            "select(flag1 * flag2, 0, "
            "step(depletion_q1 + depletion_q2 + 2 - n) * "
            "depletion_prefactor * (depletion_q1 + depletion_q2 + 2 - n)^2 "
            "* (n + 2 * (depletion_q1 + depletion_q2 + 2) "
            "- 3.0 / n * (depletion_q1^2 + depletion_q2^2 - 2.0 * depletion_q1 * depletion_q2)));"
            "n = r / depletant_radius;"
        )
        depletion_potential.addGlobalParameter(
            "depletion_prefactor",
            (-unit.BOLTZMANN_CONSTANT_kB * self._temperature * unit.AVOGADRO_CONSTANT_NA
             * self._depletion_phi / 16.0).value_in_unit(energy_unit))
        depletion_potential.addGlobalParameter("depletant_radius",
                                               self._depletant_radius.value_in_unit(length_unit))
        depletion_potential.addPerParticleParameter("depletion_q")
        depletion_potential.addPerParticleParameter("flag")
        return depletion_potential
    
    def add_particle(self, radius: unit.Quantity, substrate_flag: bool = False) -> None:
        """
        Add a colloid with a given radius to the system.

        If the substrate flag is True, the colloid is considered to be a substrate particle. Substrate particles do
        not interact with each other. In this class, this is achieved by setting the flag per-particle parameter to 1
        for substrate particles and to 0 for non-substrate particles. This flag is used in the algebraic expression of
        the steric and electrostatic potentials. Interaction groups would also work but are considerably slower
        (see https://github.com/openmm/openmm/issues/2698).

        This method has to be called for every particle in the system before the method yield_potentials is used.

        :param radius:
            The radius of the colloid.
            The unit of the radius must be compatible with nanometers and the value must be greater than zero.
        :type radius: unit.Quantity
        :param substrate_flag:
            Whether the colloid is a substrate particle.
        :type substrate_flag: bool

        :raises TypeError:
            If the radius is not a Quantity with a proper unit.
        :raises ValueError:
            If the radius is not greater than zero.
        :raises RuntimeError:
            If the method yield_potentials was called before this method (via the abstract base class).
        """
        super().add_particle()
        if not radius.unit.is_compatible(length_unit):
            raise TypeError("argument radius must have a unit that is compatible with nanometers")
        if not radius.value_in_unit(length_unit) > 0.0:
            raise ValueError("argument radius must have a value greater than zero")
        if radius.in_units_of(length_unit) > self._max_radius:
            self._max_radius = radius.in_units_of(length_unit)
        self._depletion_potential.addParticle([(radius + self._brush_length) / self._depletant_radius,
                                               int(substrate_flag)])

    def yield_potentials(self) -> Iterator[CustomNonbondedForce]:
        """
        Generate the depletion pair potential between colloids in a solution in an OpenMM system.

        This method has to be called after the method add_particle is called for every particle in the system.

        :return:
            A generator that yields the depletion potential handled by this class.
        :rtype: Iterator[CustomNonbondedForce]

        :raises RuntimeError:
            If the method add_particle was not called before this method (via the abstract base class).
        """
        super().yield_potentials()
        assert not math.isinf(self._max_radius.value_in_unit(length_unit))

        if self._periodic_boundary_conditions:
            self._depletion_potential.setNonbondedMethod(self._depletion_potential.CutoffPeriodic)
        else:
            self._depletion_potential.setNonbondedMethod(self._depletion_potential.CutoffNonPeriodic)
        self._depletion_potential.setCutoffDistance(
            (2.0 * self._max_radius + 2.0 * self._depletant_radius + 2.0 * self._brush_length).value_in_unit(length_unit))
        self._depletion_potential.setUseLongRangeCorrection(False)
        self._depletion_potential.setUseSwitchingFunction(False)
        self._depletion_potential.setName(self._name)

        yield self._depletion_potential

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
        self._depletion_potential.addExclusion(particle_one, particle_two)


class MLDepletionPotential(OpenMMNonbondedPotentialAbstract):
    """
    ML-based depletion potential using a TorchMD-Net model and OpenMM-Torch.

    Computes the depletion free energy via a pre-trained neural network instead of the
    analytical Asakura-Oosawa expression. Requires torch, torchmdnet, and openmmtorch.

    :param ml_model:
        Path to the TorchMD-Net checkpoint file (.ckpt).
    :type ml_model: str
    :param depletion_phi:
        Volume fraction of depletants. Must be between 0 and 1.
    :type depletion_phi: float
    :param depletant_radius:
        Radius of the depletants. Must be compatible with nanometers.
    :type depletant_radius: unit.Quantity
    :param temperature:
        System temperature. Must be compatible with kelvin.
    :type temperature: unit.Quantity
    :param particle_type_map:
        Mapping from GSD particle type name (str) to ML model type index (int).
    :type particle_type_map: dict
    :param periodic_boundary_conditions:
        Whether the TorchForce should use periodic boundary conditions.
    :type periodic_boundary_conditions: bool
    """

    _name = "ml_depletion_energy"

    def __init__(self, ml_model: str, depletion_phi: float, depletant_radius: unit.Quantity,
                 temperature: unit.Quantity, particle_type_map: dict,
                 periodic_boundary_conditions: bool = True):
        if not _ML_AVAILABLE:
            raise ImportError("MLDepletionPotential requires torch, torchmdnet, and openmmtorch.")
        super().__init__()
        if not 0.0 <= depletion_phi <= 1.0:
            raise ValueError("depletion_phi must be between zero and one")
        if not depletant_radius.unit.is_compatible(length_unit):
            raise TypeError("depletant_radius must have a unit compatible with nanometers")
        if depletant_radius <= 0.0 * length_unit:
            raise ValueError("depletant_radius must be greater than zero")
        if not temperature.unit.is_compatible(temperature_unit):
            raise TypeError("temperature must have a unit compatible with kelvin")
        if not temperature.value_in_unit(temperature_unit) > 0.0:
            raise ValueError("temperature must be greater than zero")

        self._ml_model = ml_model
        self._depletion_phi = depletion_phi
        self._depletant_radius = depletant_radius
        self._temperature = temperature
        self._particle_type_map = particle_type_map
        self._periodic_boundary_conditions = periodic_boundary_conditions
        self._type_ids = []

    def add_particle(self, type_name: str) -> None:
        """
        Record the ML type index for a particle.

        :param type_name:
            GSD particle type name, looked up in the particle_type_map provided at construction.
        :type type_name: str
        """
        super().add_particle()
        self._type_ids.append(self._particle_type_map.get(type_name, 0))

    def yield_potentials(self) -> Iterator:
        """
        Build and yield the TorchForce wrapping the ML depletion model.

        Must be called after add_particle has been called for every particle.
        """
        super().yield_potentials()
        kT = (unit.BOLTZMANN_CONSTANT_kB * self._temperature * unit.AVOGADRO_CONSTANT_NA
              ).in_units_of(unit.kilojoules_per_mole)._value
        type_ids = np.array(self._type_ids, dtype=np.int32)
        r_d = self._depletant_radius.value_in_unit(length_unit)

        module = _MLDepletionModule(self._ml_model, type_ids, r_d, kT)
        force = _TorchForce(torch.jit.script(module))
        force.addGlobalParameter("ml_depletion_phi", self._depletion_phi)
        force.setUsesPeriodicBoundaryConditions(self._periodic_boundary_conditions)
        force.setName(self._name)
        yield force

    def add_exclusion(self, particle_one: int, particle_two: int) -> None:
        """No-op: TorchForce does not support pairwise exclusions."""
        pass
