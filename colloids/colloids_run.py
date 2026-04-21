import argparse
import inspect
import sys
from typing import Optional, Sequence
import warnings
import gsd.hoomd
import numpy as np
import openmm
from openmm import app
from colloids import (ColloidPotentialsAlgebraic, ColloidPotentialsParameters, ShiftedLennardJonesWalls,
                      ImplicitSubstrateWall, DepletionPotential, Gravity, PlumedPotential)
from colloids.gsd_reporter import GSDReporter
from colloids.helper_functions import get_cell_from_box, read_gsd_file, write_gsd_file
import colloids.integrators as integrators
from colloids.run_parameters import RunParameters
from colloids.status_reporter import StatusReporter
import colloids.update_reporters as update_reporters
from colloids.units import electric_potential_unit, length_unit


def simple_formatwarning(msg: str, category: Warning, filename: str, lineno: int, line: Optional[str] = None) -> str:
    """
    Simpler format for warnings that excludes the line with the code that caused the warning.

    :param msg:
        The warning message.
    :type msg: str
    :param category:
        The warning category.
    :type category: Warning
    :param filename:
        The filename where the warning occurred.
    :type filename: str
    :param lineno:
        The line number where the warning occurred.
    :type lineno: int
    :param line:
        The line of code that caused the warning (not used).
    :type line: Optional[str]

    :return:
        The formatted warning message.
    :rtype: str
    """
    return f"{filename}:{lineno}: {category.__name__}: {msg}\n"


warnings.formatwarning = simple_formatwarning


class ExampleAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # TODO ADD OPTION FOR PLATFORM PROPERTIES?
        default_parameters = RunParameters()
        default_parameters.to_yaml("example.yaml")
        parser.exit()


def check_frame(parameters: RunParameters, frame: gsd.hoomd.Frame) -> None:
    """Check the frame and the run parameters."""
    for diameter in frame.particles.diameter:
        if not diameter > 0.0:
            raise ValueError("Every diameter must be greater than zero.")
    for mass in frame.particles.mass:
        if not mass >= 0.0:
            raise ValueError("Every mass must be greater than or equal to zero.")
    for constraint_value in frame.constraints.value:
        if not constraint_value > 0.0:
            raise ValueError("Every constraint distance must be greater than zero.")

    if any(parameters.wall_directions):
        # Check for orthogonal box vectors if walls should be included.
        if not all(a == 0.0 for a in frame.configuration.box[3:]):
            raise ValueError("If any wall is included, the box vectors must be parallel to the coordinate axes.")
        # If not all walls are present, the box of OpenMM needs to be enlarged because OpenMM will use periodic
        # boundaries, and we do not want to let particles interact through the walls. The enlargement of the box does
        # currently not consider the depletant radius, which is why it should be small enough.
        if not all(parameters.wall_directions):
            if parameters.use_depletion:
                if (parameters.depletant_radius
                        > (parameters.cutoff_factor * parameters.debye_length - 2.0 * parameters.brush_length) / 2.0):
                    raise ValueError("The depletant radius is too large for the cutoff factor and brush length when "
                                     "partial walls are included (r_d <= (cutoff_factor * lambda_D - 2 * L) / 2)")

    if parameters.use_depletion:
        assert (parameters.depletant_radius is not None and parameters.depletant_radius.value_in_unit(length_unit) > 0.0)
        for diameter in frame.particles.diameter:
            if parameters.depletant_radius.value_in_unit(length_unit) / (diameter / 2.0) > 0.1547:
                warnings.warn("Size ratio of depletant to colloid particles is too large. "
                              "Analytical computation of depletion potential may be invalid."
                              "See Dijkstra et. al., Journal of Physics: Condensed Matter, 1999, Volume 11, "
                              "pp 10079 - 10106.")

    # Explicit substrate is detected by immobile particles with mass 0.0.
    use_explicit_substrate = any(mass == 0.0 for mass in frame.particles.mass)
    if use_explicit_substrate and parameters.use_implicit_substrate:
        raise ValueError("Cannot use both explicit and implicit substrate.")
    if use_explicit_substrate or parameters.use_implicit_substrate:
        if not all(parameters.wall_directions):
            raise ValueError("A substrate can only be used if all walls are active.")


def set_up_simulation(parameters: RunParameters, frame: gsd.hoomd.Frame) -> app.Simulation:
    radii = frame.particles.diameter / 2.0 * length_unit
    surface_potentials = frame.particles.charge * electric_potential_unit

    # ----------------------------------- Set up system and parameters. ------------------------------------------------
    topology = app.topology.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("res", chain)

    atoms = []
    for type_id in frame.particles.typeid:
        atoms.append(topology.addAtom(frame.particles.types[type_id], None, residue))

    system = openmm.System()

    cell = get_cell_from_box(frame.configuration.box)
    include_walls = any(parameters.wall_directions)
    all_walls = all(parameters.wall_directions)
    if include_walls:
        box_vector_one = cell[0]
        box_vector_two = cell[1]
        box_vector_three = cell[2]
        assert (box_vector_one[1] == 0.0 and box_vector_one[2] == 0.0 and
                box_vector_two[0] == 0.0 and box_vector_two[2] == 0.0 and
                box_vector_three[0] == 0.0 and box_vector_three[1] == 0.0)
        wall_distances = (box_vector_one[0] * length_unit if parameters.wall_directions[0] else None,
                          box_vector_two[1] * length_unit if parameters.wall_directions[1] else None,
                          box_vector_three[2] * length_unit if parameters.wall_directions[2] else None)
        final_cell = cell.copy()
        if not all_walls:
            assert (not parameters.use_depletion
                    or parameters.depletant_radius
                    > (parameters.cutoff_factor * parameters.debye_length - 2.0 * parameters.brush_length) / 2.0)
        for index, wall_direction in enumerate(parameters.wall_directions):
            if wall_direction:
                # The shifted Lennard Jones walls diverge at distance r = radius - 1 from the location of the wall,
                # where radius is the radius of the particle. The minimum distance between periodic images through
                # a wall is thus 2 * radius_min - 2, where radius_min is the smallest radius in the system.
                # The maximum cutoff of the electrostatic interactions is
                # 2 * radius_max + cutoff_factor * debye_length. In order to prevent particles from interacting
                # through the walls, we thus increase the length of the periodic box vectors (not the wall) by
                # 2 * (radius_max - radius_min) + 2 + cutoff_factor * debye_length.
                final_cell[index][index] += \
                    (2.0 * (max(radii) - min(radii)) + 2.0 * length_unit
                        + parameters.cutoff_factor * parameters.debye_length).value_in_unit(length_unit)
    else:
        wall_distances = None
        final_cell = cell

    if not all_walls and if parameters.use_pbc:
        topology.setPeriodicBoxVectors(final_cell)
        system.setDefaultPeriodicBoxVectors(openmm.Vec3(*final_cell[0]), openmm.Vec3(*final_cell[1]),
                                            openmm.Vec3(*final_cell[2]))

    # Explicit substrate is detected by immobile particles with mass 0.0.
    use_substrate = any(mass == 0.0 for mass in frame.particles.mass) or parameters.use_implicit_substrate
    assert not (any(mass == 0.0 for mass in frame.particles.mass) and parameters.use_implicit_substrate)

    # TODO: Prevent printing the traceback when the platform is not existing.
    platform = openmm.Platform.getPlatformByName(parameters.platform_name)

    integrator = getattr(integrators, parameters.integrator)(**parameters.integrator_parameters)

    potentials_parameters = ColloidPotentialsParameters(
        brush_density=parameters.brush_density, brush_length=parameters.brush_length,
        debye_length=parameters.debye_length, temperature=parameters.potential_temperature,
        dielectric_constant=parameters.dielectric_constant)

    # ---------------------------------------- Create all forces. ------------------------------------------------------
    colloid_potentials = ColloidPotentialsAlgebraic(
        colloid_potentials_parameters=potentials_parameters, use_log=parameters.use_log,
        cutoff_factor=parameters.cutoff_factor, periodic_boundary_conditions=not all_walls,
        steric_radius_average=parameters.steric_radius_average,
        electrostatic_radius_average=parameters.electrostatic_radius_average)

    if include_walls:
        slj_walls = ShiftedLennardJonesWalls(wall_distances, parameters.epsilon, parameters.alpha,
                                             parameters.wall_directions, use_substrate)
    else:
        slj_walls = None

    if parameters.use_depletion:
        depletion_potential = DepletionPotential(parameters.depletion_phi, parameters.depletant_radius,
                                                 brush_length=parameters.brush_length,
                                                 temperature=parameters.potential_temperature,
                                                 periodic_boundary_conditions=not all_walls)
    else:
        depletion_potential = None

    if parameters.use_gravity:
        assert all_walls
        gravitational_potential = Gravity(parameters.gravitational_acceleration, parameters.water_density,
                                          parameters.particle_density)
    else:
        gravitational_potential = None

    if parameters.use_implicit_substrate:
        substrate_wall = ImplicitSubstrateWall(colloid_potentials_parameters=potentials_parameters,
                                               wall_distance_z=wall_distances[2],
                                               substrate_charge=parameters.substrate_wall_charge,
                                               use_log=parameters.use_log)
    else:
        substrate_wall = None

    if parameters.use_plumed:
        plumed = PlumedPotential(parameters.plumed_script)
    else:
        plumed = None

    # --------------------------- Add all particles and constraints to the system. -------------------------------------
    for mass in frame.particles.mass:
        system.addParticle(mass)

    for i in range(frame.constraints.N):
        if not len(frame.constraints.group[i]) == 2:
            raise ValueError("Every constraint must have exactly two particles.")
        system.addConstraint(frame.constraints.group[i][0], frame.constraints.group[i][1], frame.constraints.value[i])

    # ------------------------------------- Add all particles to the forces. -------------------------------------------
    # Be careful to add the particles in the same order as to the system.
    for i in range(frame.particles.N):
        is_substrate = frame.particles.mass[i] == 0.0
        colloid_potentials.add_particle(radius=radii[i], surface_potential=surface_potentials[i],
                                        substrate_flag=is_substrate)
        if include_walls and not is_substrate:
            slj_walls.add_particle(index=i, radius=radii[i])
        if parameters.use_depletion:
            depletion_potential.add_particle(radius=radii[i], substrate_flag=is_substrate)
        if parameters.use_gravity and not is_substrate:
            gravitational_potential.add_particle(index=i, radius=radii[i])
        if parameters.use_implicit_substrate:
            assert not is_substrate
            substrate_wall.add_particle(index=i, radius=radii[i], surface_potential=surface_potentials[i])
        if parameters.use_plumed:
            plumed.add_particle()

    for i in range(frame.constraints.N):
        colloid_potentials.add_exclusion(frame.constraints.group[i][0], frame.constraints.group[i][1])
        if parameters.use_depletion:
            depletion_potential.add_exclusion(frame.constraints.group[i][0], frame.constraints.group[i][1])

    # -------------------------------------- Add all forces to the system. ---------------------------------------------
    for force in colloid_potentials.yield_potentials():
        force.setForceGroup(system.getNumForces())
        system.addForce(force)

    if include_walls:
        for force in slj_walls.yield_potentials():
            force.setForceGroup(system.getNumForces())
            system.addForce(force)

    if parameters.use_depletion:
        for force in depletion_potential.yield_potentials():
            force.setForceGroup(system.getNumForces())
            system.addForce(force)

    if parameters.use_gravity:
        assert all_walls
        for force in gravitational_potential.yield_potentials():
            force.setForceGroup(system.getNumForces())
            system.addForce(force)
        assert not system.usesPeriodicBoundaryConditions()

    if parameters.use_implicit_substrate:
        assert all_walls
        for force in substrate_wall.yield_potentials():
            force.setForceGroup(system.getNumForces())
            system.addForce(force)

    if parameters.use_plumed:
        for force in plumed.yield_potentials():
            force.setForceGroup(system.getNumForces())
            system.addForce(force)

    # -------------------------------------- Set up the simulation. ----------------------------------------------------
    if parameters.platform_name == "CUDA":
        simulation = app.Simulation(topology, system, integrator, platform,
                                    platformProperties={"Precision": "mixed"})
    else:
        simulation = app.Simulation(topology, system, integrator, platform)

    return simulation


def set_up_reporters(parameters: RunParameters, simulation: app.Simulation, append_file: bool,
                     total_number_steps: int, initial_frame: gsd.hoomd.Frame) -> None:
    simulation.reporters.append(GSDReporter(parameters.trajectory_filename, parameters.trajectory_interval,
                                            initial_frame.particles.diameter / 2.0 * length_unit,
                                            initial_frame.particles.charge * electric_potential_unit, simulation,
                                            append_file=append_file,
                                            cell=get_cell_from_box(initial_frame.configuration.box) * length_unit))
    simulation.reporters.append(StatusReporter(max(1, total_number_steps // 100), total_number_steps,
                                               desc="Production"))
    simulation.reporters.append(app.StateDataReporter(parameters.state_data_filename,
                                                      parameters.state_data_interval, time=True,
                                                      kineticEnergy=True, potentialEnergy=True, temperature=True,
                                                      speed=True, append=append_file))

    if parameters.update_reporter is not None:
        update_reporter = getattr(update_reporters, parameters.update_reporter)
        try:
            simulation.reporters.append(update_reporter(simulation=simulation, append_file=append_file,
                                                        **parameters.update_reporter_parameters))
        except TypeError:
            raise TypeError(
                f"UpdateReporter does not accept the given arguments {parameters.update_reporter_parameters}. "
                f"The expected signature is {inspect.signature(update_reporter)} (the simulation and append_file "
                f"arguments should not be specified).")
    # The CheckpointReporter should always be last to ensure that all other reporters have been executed before it.
    simulation.reporters.append(app.CheckpointReporter(parameters.checkpoint_filename,
                                                       parameters.checkpoint_interval))


def colloids_run(argv: Sequence[str]) -> app.Simulation:
    parser = argparse.ArgumentParser(description="Run OpenMM for a colloids system.")
    parser.add_argument("yaml_file", help="YAML file with simulation parameters", type=str)
    parser.add_argument("-c", "--checkpoint_file", help="OpenMM checkpoint file", type=str,
                        default=None)
    parser.add_argument("--example", help="write an example YAML file and exit", action=ExampleAction)
    args = parser.parse_args(args=argv)

    if not args.yaml_file.endswith(".yaml"):
        raise ValueError("The YAML file must have the .yaml extension.")

    parameters = RunParameters.from_yaml(args.yaml_file)

    frame = read_gsd_file(parameters.initial_configuration, parameters.frame_index)

    check_frame(parameters, frame)

    simulation = set_up_simulation(parameters, frame)

    if args.checkpoint_file is not None:
        if not args.checkpoint_file.endswith(".chk"):
            raise ValueError("The checkpoint file must have the .chk extension.")

        simulation.loadCheckpoint(args.checkpoint_file)

        set_up_reporters(parameters, simulation, True, parameters.run_steps, frame)
    else:
        simulation.context.setPositions(frame.particles.position)

        if parameters.velocity_seed is not None:
            if not np.all(frame.particles.velocity == 0.0):
                warnings.warn("The initial velocities in the GSD file are ignored because a velocity seed is provided.")
            if parameters.velocity_seed < 0:
                simulation.context.setVelocitiesToTemperature(parameters.potential_temperature)
            else:
                simulation.context.setVelocitiesToTemperature(parameters.potential_temperature,
                                                              parameters.velocity_seed)
        else:
            if np.all(frame.particles.velocity == 0.0):
                warnings.warn(
                    "All initial velocities in the GSD file are zero. Set a velocity seed to assign random "
                    "values based on the temperature (use a negative seed to generate a random seed automatically).")
            simulation.context.setVelocities(frame.particles.velocity)

        if parameters.minimize_energy_initially:
            # Add reporter during minimization?
            # See https://openmm.github.io/openmm-cookbook/dev/notebooks/cookbook/report_minimization.html
            simulation.minimizeEnergy()

        if parameters.equilibration_steps > 0:
            simulation.reporters.append(StatusReporter(
                max(1, parameters.equilibration_steps // 100), parameters.equilibration_steps, desc="Equilibration"))
            simulation.step(parameters.equilibration_steps)
            simulation.reporters = []

        # Reset the current step to zero after the equilibration.
        simulation.currentStep = 0

        set_up_reporters(parameters, simulation, False, parameters.run_steps, frame)

    simulation.step(parameters.run_steps)

    if parameters.final_configuration_gsd_filename is not None:
        write_gsd_file(parameters.final_configuration_gsd_filename, simulation,
                       frame.particles.diameter / 2.0 * length_unit,
                       frame.particles.charge * electric_potential_unit,
                       get_cell_from_box(frame.configuration.box) * length_unit)

    return simulation


def main():
    colloids_run(sys.argv[1:])


if __name__ == '__main__':
    main()
