import csv
import os
from typing import Iterator
from abc import abstractmethod, ABC


import openmm
import openmm.app
from openmm import CustomCVForce

from colloids.abstracts import OpenMMPotentialAbstract
from colloids.helper_functions import get_cell_from_box
from colloids.units import energy_unit


class UmbrellaSamplingPotential:
    """
    Abstract class for harmonic restraint force used in umbrella sampling simulations.
    """

    def __init__(self, name, cv_force, center, force_constant) -> None:
        #super().__init__()
        self._name = name
        self._cv_force = cv_force
        self._center = center
        self._force_constant = force_constant

    '''def add_particle(self) -> None:
        """
        Add a particle to the restraint potential.

        This method has to be called for every particle in the system before the method yield_potentials is used.
        """
        super().add_particle()'''
    
    def yield_potentials(self) -> "Iterator[openmm.CustomCVForce]":

        """
        Generate the restraint force for an OpenMM system.

        This method has to be called after the method add_particle was called for every particle in the system.

        :return:
            An iterator yielding a restraint force. If using multiple CVs this yields a sum of the restraint forces.
        :rtype: Iterator[openmm.CustomCVForce]
        """
        #super().yield_potentials()
        
        umbrella_force = CustomCVForce("0.5*force_constant*(cv-center)^2 ;")
        
        umbrella_force.setName(f"umbrella_{self._name}")
        
        umbrella_force.addGlobalParameter("force_constant", self._force_constant.value_in_unit(energy_unit))
        umbrella_force.addGlobalParameter("center", self._center)
        umbrella_force.addCollectiveVariable("cv", self._cv_force)

        return umbrella_force


class UmbrellaSamplingReporter(object):
    """Write umbrella sampling CVs and bias values to a CSV file."""

    def __init__(self, filename: str, umbrella_force: openmm.CustomCVForce, force_group: int,
                print_interval: int = 1, append_file: bool = False) -> None:
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
            The value of the cv.
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