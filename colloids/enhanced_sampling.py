import csv
import os
from typing import Iterator
from abc import abstractmethod, ABC


import openmm
import openmm.app
from openmm import unit, CustomCVForce

from colloids.units import energy_unit


class UmbrellaSamplingPotential:
    """
    Abstract class for harmonic restraint force used in umbrella sampling simulations.

    :param name:
        The name of the OpenMM force object.
    :type name: str
    :param cv_force:
        The custom collective variable force in terms of which the restraint is being defined.
        This must be one of the implemented collective variable abstract classes (validated at
        simulation runtime).
    :type cv_force: openmm.CustomCVForce
    :param center:
        The center for the harmonic restraint.
    :type center: float
    :param force_constant:
        The force constant for the harmonic restraint.
    :type force_constant: unit.Quantity

    :raises TypeError:
        If the force constant is not compatible with energy units.
    """

    def __init__(self, name: str, cv_force: openmm.CustomCVForce, center: float, force_constant: unit.Quantity) -> None:
        
        if not force_constant.unit.is_compatible(energy_unit):
            raise TypeError("Force constant must have a unit compatible with kilojoules per mole.")
        
        self._name = name
        self._cv_force = cv_force
        self._center = center
        self._force_constant = force_constant
        
    def yield_potentials(self) -> "Iterator[openmm.CustomCVForce]":

        """
        Generate the restraint force for an OpenMM system.

        :return:
            An iterator yielding a restraint force. If using multiple CVs, this yields a sum of the restraint forces.
        :rtype: Iterator[openmm.CustomCVForce]
        """

        umbrella_force = CustomCVForce("0.5*force_constant*(cv-center)^2 ;")
        
        umbrella_force.setName(f"umbrella_{self._name}")
        
        umbrella_force.addGlobalParameter("force_constant", self._force_constant.value_in_unit(energy_unit))
        umbrella_force.addGlobalParameter("center", self._center)
        umbrella_force.addCollectiveVariable("cv", self._cv_force)

        return umbrella_force


class UmbrellaSamplingReporter(object):
    """Reporter for an OpenMM simulation that writes umbrella sampling CVs and bias values to a CSV file.
    
    :param filename:
        The name of the file to write to.
        The filename must end with the .csv extension.
    :type filename: str
    :param umbrellla_force: 
        The OpenMM umbrella restraint force for which the CV and bias are being output at the
        specified print interval.
    :type umbrella_force: openmm.CustomCVForce
    :param force_group:
        The force group to which the umbrella restraint force object belongs.
    :type force_group: int
    :param print_interval:
        The interval (in time steps) at which to write out CV and bias values to the csv file.
        The value must be greater than zero.
    :type print_interval: int
    :param append_file:
        If True, open an existing csv file to append to. If False, try to create a new file, and throw an error if the
        file already exists.
        Defaults to False.
    :type append_file: bool

    :raises ValueError:
        If the filename does not end with the .csv extension.
        If the print_interval is not greater than zero.
    """

    def __init__(self, filename: str, umbrella_force: openmm.CustomCVForce, force_group: int,
                print_interval: int, append_file: bool = False) -> None:
        """Constructor of the UmbrellaSamplingReporter class."""
        if not filename.endswith(".csv"):
            raise ValueError("The file must have the .csv extension.")
        if not print_interval > 0:
            raise ValueError("The print frequency must be greater than zero.")
        self._umbrella_force = umbrella_force
        self._force_group = force_group
        self._print_interval = print_interval
        self._file = open(filename, "a" if append_file else "w")
        if not append_file:
            print("timestep,cv,bias", file=self._file, flush=True)
    
    # noinspection PyPep8Naming
    def describeNextReport(self, simulation: openmm.app.Simulation) -> tuple[int, bool, bool, bool, bool, bool]:
        """Get information about the next report this reporter will generate.

        This method is called by OpenMM once this reporter is added to the list of reporters of a simulation.

        :param simulation:
            The simulation to generate a report for.
        :type simulation: openmm.app.Simulation

        :returns:
            (Number of steps until next report,
            Whether the next report requires positions (False),
            Whether the next report requires velocities (False),
            Whether the next report requires forces (False),
            Whether the next report requires energies (False),
            Whether positions should be wrapped to lie in a single periodic box (False))
        :rtype: tuple[int, bool, bool, bool, bool, bool]
        """
        steps = self._print_interval - simulation.currentStep % self._print_interval
        return steps, False, False, False, False, False

    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        
        cv_value = self._umbrella_force.getCollectiveVariableValues(simulation.context)[0]

        bias_state = simulation.context.getState(getEnergy=True, groups={self._force_group})
        bias_value = bias_state.getPotentialEnergy().value_in_unit(energy_unit)

        self.print(simulation, cv_value, bias_value)
    
    def print(self, simulation: openmm.app.Simulation, cv_value: float, bias_value: float) -> None:
        """
        Print the CV and bias values in the output csv file.

        :param simulation:
            The OpenMM simulation.
        :type simulation: openmm.app.Simulation
        :param cv_value:
            The value of the CV.
        :type cv_value: float
        :param bias_value: 
            The value of the restraint bias (dimensionless, but units correspond to the energy units
            in the simulation).
        :type bias_value: float
        """
        step = simulation.currentStep
        if step % self._print_interval == 0:
            print(f"{step},{cv_value},{bias_value}", file=self._file, flush=True)

    def __del__(self) -> None:
        """Destructor of the UmbrellaSamplingReporter class."""
        try:
            self._file.close()
        except AttributeError:
            # If another error occurred, the '_file' attribute might not exist.
            pass

class MetadynamicsReporter(object):
    """Reporter for an OpenMM simulation that writes metadynamics CVs and bias values to a CSV file.
    
    :param filename:
        The name of the file to write to.
        The filename must end with the .csv extension.
    :type filename: str
    :param metad_object: 
        The OpenMM metadynamics objects for which the CVs and biases are being output at the
        specified print interval.
    :type metad_object: openmm.app.metadynamics.Metadynamics
    :param print_interval:
        The interval (in time steps) at which to write out CV and bias values to the csv file.
        The value must be greater than zero.
    :type print_interval: int
    :param append_file:
        If True, open an existing csv file to append to. If False, try to create a new file, and throw an error if the
        file already exists.
        Defaults to False.
    :type append_file: bool

    :raises ValueError:
        If the filename does not end with the .csv extension.
        If the print_interval is not greater than zero.
    """

    def __init__(self, filename: str, metad_object: openmm.app.metadynamics.Metadynamics,
                print_interval: int, append_file: bool = False) -> None:
        """Constructor of the MetadynamicsReporter class."""
        if not filename.endswith(".csv"):
            raise ValueError("The file must have the .csv extension.")
        if not print_interval > 0:
            raise ValueError("The print frequency must be greater than zero.")
        self._metad = metad_object
        self._print_interval = print_interval
        self._file = open(filename, "a" if append_file else "w")
        if not append_file:
            cv_names = [f"cv{i + 1}" for i in range(len(self._metad.variables))]
            print(",".join(["timestep", *cv_names, "bias"]), file=self._file, flush=True)
    
    # noinspection PyPep8Naming
    def describeNextReport(self, simulation: openmm.app.Simulation) -> tuple[int, bool, bool, bool, bool, bool]:
        """Get information about the next report this reporter will generate.

        This method is called by OpenMM once this reporter is added to the list of reporters of a simulation.

        :param simulation:
            The simulation to generate a report for.
        :type simulation: openmm.app.Simulation

        :returns:
            (Number of steps until next report,
            Whether the next report requires positions (False),
            Whether the next report requires velocities (False),
            Whether the next report requires forces (False),
            Whether the next report requires energies (False),
            Whether positions should be wrapped to lie in a single periodic box (False))
        :rtype: tuple[int, bool, bool, bool, bool, bool]
        """
        steps = self._print_interval - simulation.currentStep % self._print_interval
        return steps, False, False, False, False, False

    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        
        cv_values = self._metad.getCollectiveVariables(simulation)
        
        force_group = self._metad._force.getForceGroup()
        bias_state = simulation.context.getState(getEnergy=True,groups={force_group})
        bias_energy = bias_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        
        self.print(simulation, cv_values, bias_energy)
    
    def print(self, simulation: openmm.app.Simulation, cv_values: list[float], bias_value: float) -> None:
        """
        Print the CV and bias values in the output csv file.

        :param simulation:
            The OpenMM simulation.
        :type simulation: openmm.app.Simulation
        :param cv_values:
            The values of the biased CVs.
        :type cv_values: list[float]
        :param bias_value: 
            The value of the restraint bias (dimensionless, but units correspond to the energy units
            in the simulation).
        :type bias_value: float
        """
        step = simulation.currentStep
        if step % self._print_interval == 0:
            values = [step, *cv_values, bias_value]
            print(",".join(str(value) for value in values), file=self._file, flush=True)

    def __del__(self) -> None:
        """Destructor of the MetadynamicsReporter class."""
        try:
            self._file.close()
        except AttributeError:
            # If another error occurred, the '_file' attribute might not exist.
            pass
