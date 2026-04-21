from dataclasses import dataclass, field
import inspect
from typing import Any, Optional
import warnings
from openmm import unit
from colloids.abstracts import Parameters
import colloids.integrators as integrators
import colloids.update_reporters as update_reporters
from colloids.units import energy_unit, length_unit, temperature_unit, time_unit, electric_potential_unit


@dataclass(order=True, frozen=True)
class RunParameters(Parameters):
    """
    Data class for the parameters of an OpenMM simulation of colloidal particles.

    This dataclass can be written to and read from a yaml file. The yaml file contains the parameters as key-value
    pairs. Any OpenMM quantities are converted to Quantity objects that can be represented in a readable way in the
    yaml file.

    The potentials are given in Hueckel, Hocky, Palacci & Sacanna, Nature 580, 487--490 (2020)
    (see https://doi.org/10.1038/s41586-020-2205-0). Any references to equations or symbols in the code refer to this
    paper.

    The initial configuration must be a single frame in a gsd file, and the filename must end with the ".gsd" extension
    (see https://gsd.readthedocs.io/en/stable/python-module-gsd.hoomd.html). The following attributes of the frame are
    used during the simulation:

    - frame.particles.N -> Total number of particles in the frame (including colloids, substrate, snowman heads, etc.).
    - frame.particles.position -> Positions of all particles in the frame.
    - frame.particles.velocity -> Velocities of all particles in the frame. Can be overridden by sampling from a
                                  Maxwell-Boltzmann distribution if velocity_seed is not None.
    - frame.particles.types -> Possible types of all particles in the frame.
    - frame.particles.typeid -> Type index within the frame.particles.types tuple of each particle in the frame.
    - frame.particles.diameter -> Diameter in nanometer of each particle in the frame that is used to infer the radius.
    - frame.particles.charge -> Surface potential in millivolt of each particle in the frame.
    - frame.particles.mass -> Mass in atomic mass units of each particle in the frame. A zero mass signals immobile
                              particles and are interpreted as the substrate.
    - frame.configuration.box -> Box dimensions of the frame. The first three entries are the box lengths in x, y, and z
                                 directions in nanometers. The next three entries are the tilt factors xy, xz, and yz.
    - frame.constraints.N -> Total number of constraints in the frame.
    - frame.constraints.value -> Constraint lengths in nanometers of all constraints in the frame.
    - frame.constraints.group -> Particle pairs for all constraints in the frame.

    Note that gsd files can store constraints directly in the frame.constraints attribute. One has to be careful,
    however, that Ovito ignores the frame.constraints attribute. This means that one has to manually store the
    constraint distances into the GSD file once a gsd file is exported from Ovito

    :param initial_configuration:
        The path to the initial configuration of the system in a gsd file.
        The filename must end with ".gsd".
        Defaults to "initial_configuration.gsd".
    :type initial_configuration: str
    :param frame_index:
        The index of the frame in the gsd file that is used as the initial configuration.
        It is also possible to use negative indices to count from the end of the file.
        Defaults to -1 (i.e., the last frame in the file).
    :type frame_index: int
    :param platform_name:
        The name of the platform to use for the simulation.
        Defaults to "Reference". Other possible choices are "CPU", "CUDA", or "OpenCL".
    :type platform_name: str
    :param potential_temperature:
        The temperature that is used for the colloid potentials.
        The unit of the temperature must be compatible with kelvin and the value must be greater than zero.
        Defaults to 298.0 * unit.kelvin.
    :type potential_temperature: unit.Quantity
    :param integrator:
        The name of the OpenMM integrator to use for the molecular-dynamics simulations.
        Possible choices are "BrownianIntegrator", "LangevinIntegrator", LangevinMiddleIntegrator",
        "NoseHooverIntegrator", "VariableLangevinIntegrator", "VariableVerletIntegrator", and "VerletIntegrator".
        Defaults to "LangevinIntegrator".
    :type integrator: str
    :param integrator_parameters:
        The parameters that are forwarded to initialize the OpenMM integrator.
        Each integrator has specific parameters, and the parameters passed in here must be compatible with the chosen
        integrator. See the corresponding integrator in the OpenMM documentation
        https://docs.openmm.org/latest/api-python/library.html#integrators for the possible arguments (or, alternatively,
        the colloids.integrators module).
        Defaults to sensible values for the LangevinIntegrator (temperature of 298 K, frictionCoeff of
        0.001574074286750681 / ps, stepSize of 0.00317647015905543 ps, and no specified random number seed).
    :type integrator_parameters: dict[str, Any]
    :param brush_density:
        The polymer surface density in the Alexander-de Gennes polymer brush model [i.e., sigma in eq. (1)].
        The unit of the brush_density must be compatible with 1/nanometer^2 and the value must be greater than zero.
        Defaults to 0.09 / ((unit.nano * unit.meter) ** 2).
    :type brush_density: unit.Quantity
    :param brush_length:
        The thickness of the brush in the Alexander-de Gennes polymer brush model [i.e., L in eq. (1)].
        The unit of the brush_length must be compatible with nanometers and the value must be greater than zero.
        Defaults to 10.6 * (unit.nano * unit.meter).
    :type brush_length: unit.Quantity
    :param debye_length:
        The Debye screening length within DLVO theory [i.e., lambda_D].
        The unit of the debye_length must be compatible with nanometers and the value must be greater than zero.
        Defaults to 5.726968 * (unit.nano * unit.meter).
    :type debye_length: unit.Quantity
    :param dielectric_constant:
        The dielectric constant of the solvent [i.e., epsilon].
        The value of the dielectric constant must be greater than zero.
        Defaults to 80.0.
    :type dielectric_constant: float
    :param use_log:
        If True, the electrostatic force uses the more accurate equation involving a logarithm [i.e., eq. (12.5.2) in
        Hunter, Foundations of Colloid Science (Oxford University Press, 2001), 2nd edition] instead of the simpler
        equation that only involves an exponential [i.e., eq. (12.5.5) in Hunter, Foundations of Colloid Science
        (Oxford University Press, 2001), 2nd edition].
        Defaults to False.
    :type use_log: bool
    :param steric_radius_average:
        The type of radius average to use for the steric potential.
        Can be either "harmonic" or "arithmetic".
        Defaults to "harmonic".
    :type steric_radius_average: str
    :param electrostatic_radius_average:
        The type of radius average to use for the electrostatic potential.
        Can be either "harmonic" or "arithmetic".
        Defaults to "harmonic".
    :type electrostatic_radius_average: str
    :param velocity_seed:
        The seed for the random number generator that is used to sample the initial velocities replacing the velocities
        in the gsd file.
        If None, the velocities in the gsd file are used.
        If negative, a random seed is used for the random number generator.
        Defaults to None.
    :type velocity_seed: Optional[int]
    :param equilibration_steps:
        The number of time steps to equilibrate the system before the production run.
        During the equilibration run, no data is written.
        The number of time steps must be greater than or equal to zero.
        Defaults to 0.
    :type equilibration_steps: int
    :param run_steps:
        The number of time steps to run the simulation.
        The number of time steps must be greater than zero.
        Defaults to 100.
    :type run_steps: int
    :param state_data_interval:
        The interval at which state data is written to a csv file.
        The interval must be greater than zero.
        Defaults to 100.
    :type state_data_interval: int
    :param state_data_filename:
        The name of the csv file to which the state data is written.
        The filename must end with ".csv".
        Defaults to "state_data.csv".
    :type state_data_filename: str
    :param trajectory_interval:
        The interval at which the trajectory is written to a gsd file.
        The interval must be greater than zero.
        Defaults to 100.
    :type trajectory_interval: int
    :param trajectory_filename:
        The name of the gsd file to which the trajectory is written.
        The filename must end with ".gsd".
        Defaults to "trajectory.gsd".
    :type trajectory_filename: str
    :param checkpoint_interval:
        The interval at which the checkpoint is written to a chk file.
        The interval must be greater than zero.
        Defaults to 100.
    :type checkpoint_interval: int
    :param checkpoint_filename:
        The name of the chk file to which the checkpoint is written.
        The filename must end with ".chk".
        Defaults to "checkpoint.chk".
    :type checkpoint_filename: str
    :param minimize_energy_initially:
        If True, the energy of the system is minimized before the simulation starts.
        Defaults to False.
    :type minimize_energy_initially: bool
    :param final_configuration_gsd_filename:
        The name of the gsd file to which the final configuration is written.
        If None, the final configuration is not written to a gsd file.
        The filename must end with ".gsd".
        Defaults to "final_frame.gsd".
    :type final_configuration_gsd_filename: Optional[str]
    :param wall_directions:
        A list of three booleans indicating whether the walls in the x, y, and z directions are active for
        closed-wall simulations with shifted Lennard-Jones potential walls.
        If any of the wall directions is active, epsilon and alpha must be specified.
        Defaults to [False, False, False].
    :type wall_directions: list[bool]
    :param use_pbc:
        If True, use periodic boundary conditions in the simulation.
        Defaults to True.
    :type use_pbc: bool
    :param epsilon:
        The unshifted Lennard-Jones potential well-depth for closed-wall simulations with shifted Lennard-Jones
        potential walls.
        If any wall direction is True, epsilon must be not None, its unit must be compatible with kilojoules per mole
        and the value must be greater than zero.
        Defaults to None.
    :type epsilon: Optional[unit.Quantity]
    :param alpha:
        Factor determining the strength of the attractive part of the Lennard-Jones potential for closed-wall
        simulations with shifted Lennard-Jones potential walls.
        If any wall direction is True, alpha must be not None and 0 <= alpha <= 1.
        Note that the force of this potential is only continuous if alpha = 1.
    :type alpha: Optional[float]
    :param use_implicit_substrate:
        A boolean indicating whether to implicitly model a substrate implicitly as a charged wall. 
        An implicit substrate can only be used when all walls are active. The bottom wall is then replaced by the
        implicit substrate.
        An implicit substrate can only be used if no explicit substrate particles have been added to the simulation box.
        Defaults to False.
    :type use_implicit_substrate: bool
    :param substrate_wall_charge:
        The charge of the substrate wall at the bottom of the simulation box. If using an implicit substrate, substrate
        wall charge must not be None and the units must be compatible with millivolts.
        Defaults to None.
    :type substrate_wall_charge: Optional[unit.Quantity]
    :param use_depletion:
        A boolean indicating whether to turn on the depletion attraction for the simulation.
        If depletion attraction is on, depletion_phi and depletant_radius must be specified.
        Defaults to False.
    :type use_depletion: bool
    :param depletion_phi:
        The number density of polymers in the solution.
        If depletion attraction is on, the value of depletion_phi must not be None and 0 <= depletion_phi <=1.
        Defaults to None.
    :type depletion_phi: Optional[float]
    :param depletant_radius:
        The radius of the polymers in solution for a system with depletion attraction.
        If depletion attraction is on, depletant_radius must not be None, its unit must be compatible with nanometers,
        and the value must be greater than zero.
    :type depletant_radius: Optional[unit.Quantity]
    :param use_gravity: bool
        A boolean indicating whether the gravitational force is turned on for the simulation.
        If true, the gravitational acceleration, particle density, and water density parameters must be specified.
        Defaults to False.
    :param gravitational_acceleration:
        The acceleration due to gravity.
        If gravity is on, the value of the gravitational constant must be specified.
        The unit must be compatible with meters per second squared.
        Defaults to None.
    :type gravitational_acceleration: Optional[unit.Quantity]
    :param water_density:
        The density of water. This is used to compute the effective particle density when calculating the gravitational
        force.
        If gravity is on, the density of water must be specified, its unit must be compatible with grams per centimeter
        cubed, and the value must be greater than zero.
        Defaults to None.
    :type water_density: Optional[unit.Quantity]
    :param particle_density:
        The density of the colloidal particles. This is used to compute the effective particle density when calculating
        the gravitational force.
        If gravity is on, the particle density must be specified, its unit must be compatible with grams per centimeter
        cubed, and the value must be greater than zero.
        Defaults to None.
    :type particle_density: Optional[unit.Quantity]
    :param update_reporter:
        The name of the update reporter used to vary the value of a force-related global parameter over time
        in a simulation.
        Possible choices can be found in the update_reporters.py file.
        If an update reporter is specified, its update reporter parameters must be specified.
        Defaults to None.
    :type update_reporter: Optional[str]
    :param update_reporter_parameters:
        The parameters that are forwarded to the initialization method of the UpdateReporter, if enabled for a
        simulation. Note that the initialization method of the UpdateReporter class expects an OpenMM simulation object
        and an append_file boolean that should not appear in this dictionary.
        Defaults to None.
    :type update_reporter_parameters: Optional[dict[str, Any]]
    :param use_plumed: 
        A boolean indicating whether to interface a simulation with the PLUMED plugin.
        If true, plumed_script must be specified and must point to a plumed input file.
        Defaults to False.
    :type use_plumed: bool
    :param plumed_script:
        A plumed input file to be used for interfacing with a simulation.
    :type plumed_script: Optional[str]

    :raises TypeError:
        If any of the quantities has an incompatible unit.
    :raises ValueError:
        If any of the parameters has an invalid value.
    """

    initial_configuration: str = "initial_configuration.gsd"
    frame_index: int = -1
    platform_name: str = "Reference"
    potential_temperature: unit.Quantity = field(default_factory=lambda: 298.0 * temperature_unit)
    integrator: str = "LangevinIntegrator"
    integrator_parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "temperature": 298.0 * temperature_unit,
            "stepSize": 0.00317647015905543 * time_unit,
            "frictionCoeff": 0.001574074286750681 / time_unit,
            "randomNumberSeed": None
        })
    brush_density: unit.Quantity = field(default_factory=lambda: 0.09 / (length_unit ** 2))
    brush_length: unit.Quantity = field(default_factory=lambda: 10.6 * length_unit)
    debye_length: unit.Quantity = field(default_factory=lambda: 5.726968 * length_unit)
    dielectric_constant: float = 80.0
    cutoff_factor: float = 21.0
    use_log: bool = False
    steric_radius_average: str = "harmonic"
    electrostatic_radius_average: str = "harmonic"
    velocity_seed: Optional[int] = None
    equilibration_steps: int = 0
    run_steps: int = 100
    state_data_interval: int = 100
    state_data_filename: str = "state_data.csv"
    trajectory_interval: int = 100
    trajectory_filename: str = "trajectory.gsd"
    checkpoint_interval: int = 100
    checkpoint_filename: str = "checkpoint.chk"
    minimize_energy_initially: bool = False
    final_configuration_gsd_filename: Optional[str] = "final_frame.gsd"
    epsilon: Optional[unit.Quantity] = None
    alpha: Optional[float] = None
    wall_directions: list[bool] = field(default_factory=lambda: [False, False, False])
    use_pbc: bool = True
    use_implicit_substrate: bool = False
    substrate_wall_charge: Optional[unit.Quantity] = None
    use_depletion: bool = False
    depletion_phi: Optional[float] = None
    depletant_radius: Optional[unit.Quantity] = None
    use_gravity: bool = False
    gravitational_acceleration: Optional[unit.Quantity] = None
    water_density: Optional[unit.Quantity] = None
    particle_density: Optional[unit.Quantity] = None
    update_reporter: Optional[str] = None
    update_reporter_parameters: Optional[dict[str, Any]] = None
    use_plumed: bool = False
    plumed_script: Optional[str] = None

    def __post_init__(self) -> None:
        """Check if the parameters are valid after initialization."""
        if not self.initial_configuration.endswith(".gsd"):
            raise ValueError("The filename of the initial configuration must end with '.gsd'.")
        if self.platform_name not in ["Reference", "CPU", "CUDA", "OpenCL"]:
            raise ValueError("The platform name must be 'Reference', 'CPU', 'CUDA', or 'OpenCL'.")
        possible_integrators = [name for name, _ in inspect.getmembers(integrators, inspect.isfunction)]
        if self.integrator not in possible_integrators:
            raise ValueError(f"Integrator {self.integrator} not available, the integrator must be one of the "
                             f"following: {', '.join(possible_integrators)}.")
        integrator_getter = getattr(integrators, self.integrator)
        try:
            integrator_getter(**self.integrator_parameters)
        except TypeError:
            raise TypeError(f"Integrator {self.integrator} does not accept the given arguments "
                            f"{self.integrator_parameters}. The expected signature is "
                            f"{inspect.signature(integrator_getter)}")
        if not self.potential_temperature.unit.is_compatible(temperature_unit):
            raise TypeError("The temperature must have a unit compatible with kelvin.")
        if self.potential_temperature <= 0.0 * temperature_unit:
            raise ValueError("The temperature must be greater than zero.")
        if not self.brush_density.unit.is_compatible(length_unit ** (-2)):
            raise TypeError("The brush density must have a unit compatible with 1/nanometer^2.")
        if self.brush_density <= 0.0 * (length_unit ** (-2)):
            raise ValueError("The brush density must be greater than zero.")
        if not self.brush_length.unit.is_compatible(length_unit):
            raise TypeError("The brush length must have a unit compatible with nanometers.")
        if self.brush_length <= 0.0 * length_unit:
            raise ValueError("The brush length must be greater than zero.")
        if not self.debye_length.unit.is_compatible(length_unit):
            raise TypeError("The Debye length must have a unit compatible with nanometers.")
        if self.debye_length <= 0.0 * length_unit:
            raise ValueError("The Debye length must be greater than zero.")
        if self.dielectric_constant <= 0.0:
            raise ValueError("The dielectric constant must be greater than zero.")
        if self.equilibration_steps < 0:
            raise ValueError("The number of equilibration steps must be greater than or equal to zero.")
        if self.run_steps == 0:
            warnings.warn("The number of time steps is zero.")
        if self.run_steps < 0:
            raise ValueError("The number of time steps must be greater than or equal to zero.")
        if self.state_data_interval <= 0:
            raise ValueError("The state data interval must be greater than zero.")
        if not self.state_data_filename.endswith(".csv"):
            raise ValueError("The filename of the state data must end with '.csv'.")
        if self.trajectory_interval <= 0:
            raise ValueError("The trajectory interval must be greater than zero.")
        if not self.trajectory_filename.endswith(".gsd"):
            raise ValueError("The filename of the trajectory must end with '.gsd'.")
        if self.checkpoint_interval <= 0:
            raise ValueError("The checkpoint interval must be greater than zero.")
        if not self.checkpoint_filename.endswith(".chk"):
            raise ValueError("The filename of the checkpoint must end with '.chk'.")
        if (self.final_configuration_gsd_filename is not None
                and not self.final_configuration_gsd_filename.endswith(".gsd")):
            raise ValueError("The filename of the final configuration must end with '.gsd'.")
        if isinstance(self.wall_directions, str):
            raise ValueError("Wall directions was parsed as a string although it should be a list of bools. "
                             "Make sure that the yaml file is correctly formatted and that there is space after each "
                             "dash in the list of wall directions.")
        if len(self.wall_directions) != 3:
            raise ValueError("Wall directions must be specified for three dimensions.")
        if any(self.wall_directions):
            if self.epsilon is None:
                raise ValueError("Epsilon must be specified if walls are active.")
            if not self.epsilon.unit.is_compatible(energy_unit):
                raise TypeError("Epsilon must have a unit compatible with kilojoules per mole.")
            if self.epsilon <= 0.0 * energy_unit:
                raise ValueError("epsilon must be greater than zero.")
            if self.alpha is None:
                raise ValueError("Alpha must be specified if walls are active.")
            if not 0.0 <= self.alpha <= 1.0:
                raise ValueError("Alpha must be between zero and one.")
        else:
            if self.epsilon is not None:
                raise ValueError("Epsilon must not be specified if walls are not active.")
            if self.alpha is not None:
                raise ValueError("Alpha must not be specified if walls are not active.")
        if self.use_depletion:
            if self.depletion_phi is None:
                raise ValueError("Depletion phi must be specified if depletion is on.")
            if not 0.0 <= self.depletion_phi <= 1.0:
                raise ValueError("Depletion phi must be between zero and one.")
            if self.depletant_radius is None:
                raise ValueError("Depletant radius must be specified if depletion is on.")
            if not self.depletant_radius.unit.is_compatible(length_unit):
                raise TypeError("Depletant radius must have a unit compatible with nanometers.")
            if self.depletant_radius <= 0.0 * length_unit:
                raise ValueError("Depletant radius must be greater than zero.")
        else:
            if self.depletion_phi is not None:
                raise ValueError("Depletion phi must not be specified if depletion potential is not on.")
            if self.depletant_radius is not None:
                raise ValueError("Depletant radius must not be specified if depletion potential is not on.")
        if self.use_implicit_substrate:
            if not all(self.wall_directions):
                raise ValueError("A substrate can only be used if all walls are active.")
            if self.substrate_wall_charge is None:
                raise ValueError("Substrate wall charge must be specified if using implicit substrate.")
            if not self.substrate_wall_charge.unit.is_compatible(electric_potential_unit):
                raise TypeError(
                    "The substrate wall charge must have a unit compatible with millivolts.")
        else:
            if self.substrate_wall_charge is not None:
                raise ValueError("Substrate wall charge must not be specified if not using implicit substrate.")
        if self.use_gravity:
            if self.gravitational_acceleration is None:
                raise ValueError("Gravitational acceleration must be specified if gravity is on.")
            if not self.gravitational_acceleration.unit.is_compatible(length_unit / time_unit ** 2):
                raise TypeError(
                    "The gravitational acceleration must have a unit compatible with meters per second squared.")
            if self.gravitational_acceleration <= 0.0 * (length_unit / time_unit ** 2):
                raise ValueError("The gravitational acceleration must be greater than zero.")
            if self.water_density is None:
                raise ValueError("Density of water must be specified if gravity is on.")
            if not self.water_density.unit.is_compatible(unit.gram / length_unit ** 3):
                raise TypeError("The water density must have a unit compatible with grams per centimeter cubed.")
            if self.water_density <= 0.0 * (unit.gram / length_unit ** 3):
                raise ValueError("The water density must be greater than zero.")
            if self.particle_density is None:
                raise ValueError("Density of particle must be specified if gravity is on.")
            if not self.particle_density.unit.is_compatible(unit.gram / length_unit ** 3):
                raise TypeError("The particle density must have a unit compatible with grams per centimeter cubed.")
            if self.particle_density <= 0.0 * (unit.gram / length_unit ** 3):
                raise ValueError("The particle density must be greater than zero.")
            if not all(self.wall_directions):
                raise ValueError("Gravity can only be turned on if all walls are active and, hence, no periodic "
                                 "boundary conditions are present.")
        else:
            if self.gravitational_acceleration is not None:
                raise ValueError("Gravitational acceleration must not be specified if gravity is not on.")
            if self.water_density is not None:
                raise ValueError("Density of water must not be specified if gravity is not on.")
            if self.particle_density is not None:
                raise ValueError("Density of particle must not be specified if gravity is not on.")
        if self.update_reporter is not None:
            possible_update_reporters = [name for name, _ in inspect.getmembers(update_reporters, inspect.isclass)
                                         if name != "ABC" and "Abstract" not in name]
            if self.update_reporter not in possible_update_reporters:
                raise ValueError(f"Update reporter {self.update_reporter} not available, the update reporter must be one of the following:",
                                 f"{', '.join(possible_update_reporters)}.")
            if self.update_reporter_parameters is None:
                raise ValueError("Update-reporter parameters must be specified if the update reporter is on.")
            if "simulation" in self.update_reporter_parameters or "append_file" in self.update_reporter_parameters:
                raise ValueError("Update-reporter parameters should not contain simulation and append_file keys.")
        else:
            if self.update_reporter_parameters is not None:
                raise ValueError("Update-reporter parameters must not be specified if the update reporter is not on.")
        if self.use_plumed:
            if self.plumed_script is None:
                raise ValueError("PLUMED input file must be specified if using PLUMED.")
        else:
            if self.plumed_script is not None:
                raise ValueError("PLUMED input file must not be specified if PLUMED is not being used.")

if __name__ == '__main__':
    RunParameters(initial_configuration="tests/first_frame.xyz").to_yaml("example.yaml")
    parameters = RunParameters.from_yaml("example.yaml")
    print(parameters)
