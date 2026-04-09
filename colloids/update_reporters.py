from abc import abstractmethod, ABC
import math
from typing import Optional, Sequence, Union
import warnings
import freud.cluster
import numpy as np
import openmm.app
from openmm import unit
from colloids.units import length_unit, temperature_unit


class UpdateReporterAbstract(ABC):
    """
    Abstract class for reporters for an OpenMM simulation of colloids that change the value of a parameter over
    the course of the simulation.

    The inheriting class must implement the report method. The report method can be used to specify the way the
    parameter is updated.

    This class creates a csv file that stores the current simulation step and current value of the parameter
    being updated.

    Note that the init method of the inheriting class should allow for the simulation and append_file arguments because
    they are used while setting up the reporter. All other arguments can be arbitrarily specified by the inheriting
    class and are set via the update_reporter_parameters dictionary in the RunParameters class.

    :param filename:
        The name of the file to write to.
        The filename must end with the .csv extension.
    :type filename: str
    :param parameter_name:
        The name of the parameter to be updated. This is only used for the header of the output csv file.
    :type parameter_name: str
    :param simulation:
        The OpenMM simulation that this reporter will be added to.
        This argument is only included because some inheriting classes might need it during initialization.
    :type simulation: openmm.app.Simulation
    :param update_interval:
        The interval (in time steps) at which the value of the parameter in the OpenMM simulation is updated.
        The value must be greater than zero.
        Defaults to 1.
    :type update_interval: int
    :param print_interval:
        The interval (in time steps) at which the value of the parameter in the OpenMM simulation is printed to the
        output csv file.
        The value must be greater than zero.
        Defaults to 1.
    :type print_interval: int
    :param final_update_step:
        The final step at which the value of the global parameter will be updated.
        If None, the parameter will be updated until the end of the simulation.
        If not None, the value must be greater than or equal to the update_interval.
        Defaults to None.
    :type final_update_step: Optional[int]
    :param append_file:
        If True, open an existing csv file to append to. If False, create a new file possibly overwriting an already
        existing file.
        Defaults to False.
    :type append_file: bool

    :raises ValueError:
        If the filename does not end with the .csv extension.
        If the update_interval is not greater than zero.
        If the print_interval is not greater than zero.
        If the final_update_step is not None and not greater than or equal to the update_interval.
    """

    def __init__(self, filename: str, parameter_name: str, simulation: openmm.app.Simulation, update_interval: int = 1,
                 print_interval: int = 1, final_update_step: Optional[int] = None, append_file: bool = False) -> None:
        """Constructor of the UpdateReporterAbstract class."""
        if not filename.endswith(".csv"):
            raise ValueError("The file must have the .csv extension.")
        if not update_interval > 0:
            raise ValueError("The update frequency must be greater than zero.")
        if not print_interval > 0:
            raise ValueError("The print frequency must be greater than zero.")
        if final_update_step is not None and not final_update_step >= update_interval:
            raise ValueError("The final update step must be greater than or equal to the update frequency.")
        self._parameter_name = parameter_name
        self._update_interval = update_interval
        self._print_interval = print_interval
        self._final_update_step = final_update_step
        self._file = open(filename, "a" if append_file else "w")
        if not append_file:
            print(f"timestep,{self._parameter_name}", file=self._file, flush=True)

    @abstractmethod
    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        """
        Update the value of a global parameter during the simulation.

        This function is called by OpenMM when the reporter should generate a report.

        The implementation of this method in the inheriting class should compute and set the new value of the parameter.
        Then, one should call the print method to print the value in the output csv file.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        :param state:
            The current state of the OpenMM simulation.
        :type state: openmm.State
        """
        raise NotImplementedError

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
        if self._final_update_step is not None and simulation.currentStep >= self._final_update_step:
            # 0 signals to not interrupt the simulation again.
            return 0, False, False, False, False, False
        steps = self._update_interval - simulation.currentStep % self._update_interval
        return steps, False, False, False, False, False

    def print(self, simulation: openmm.app.Simulation, new_value: float) -> None:
        """
        Print the new parameter value in the output csv file.

        :param simulation:
            The OpenMM simulation.
        :type simulation: openmm.app.Simulation
        :param new_value:
            The new value of the parameter.
        :type new_value: float
        """
        step = simulation.currentStep
        if step % self._print_interval == 0:
            print(f"{step},{new_value}", file=self._file, flush=True)

    def __del__(self) -> None:
        """Destructor of the UpdateReporterAbstract class."""
        try:
            self._file.close()
        except AttributeError:
            # If another error occurred, the '_file' attribute might not exist.
            pass


class GlobalParameterUpdateReporterAbstract(UpdateReporterAbstract, ABC):
    """
    Abstract class for reporters for an OpenMM simulation of colloids that change the value of a global parameter in a
    custom force over the course of the simulation.

    The inheriting class must implement the report method. The report method can be used to specify the way the 
    global parameter is updated. The set_and_print method can be used to set the new value of the global parameter in
    the OpenMM simulation context and print it in the output csv file.

    This class creates a csv file that stores the current simulation step and current value of the global parameter
    being updated.

    :param filename:
        The name of the file to write to.
        The filename must end with the .csv extension.
    :type filename: str
    :param parameter_name:
        The name of the global parameter to be updated.
        This must be one of the global parameters passed into any of the OpenMM CustomForce objects.
    :type parameter_name: str
    :param simulation:
        The OpenMM simulation that this reporter will be added to.
        The context of this OpenMM simulation must contain the parameter to be updated.
    :type simulation: openmm.app.Simulation
    :param start_value:
        The start value of the global parameter.
        OpenMM does not store the units of global parameters, so if using a quantity with a unit, the user must make
        sure to pass in a sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type start_value: Union[unit.Quantity, float]
    :param update_interval:
        The interval (in time steps) at which the value of the global parameter in the OpenMM simulation is updated.
        The value must be greater than zero.
        Defaults to 1.
    :type update_interval: int
    :param print_interval:
        The interval (in time steps) at which the value of the global parameter in the OpenMM simulation is printed
        to the output csv file.
        The value must be greater than zero.
        Defaults to 1.
    :type print_interval: int
    :param final_update_step:
        The final step at which the value of the global parameter will be updated.
        If None, the parameter will be updated until the end of the simulation.
        If not None, the value must be greater than or equal to the update_interval.
        Defaults to None.
    :type final_update_step: Optional[int]
    :param append_file:
        If True, open an existing csv file to append to. If False, create a new file possibly overwriting an already
        existing file.
        Defaults to False.
    :type append_file: bool

    :raises ValueError:
        If the filename does not end with the .csv extension (via the base class).
        If the update_interval is not greater than zero (via the base class).
        If the print_interval is not greater than zero (via the base class).
        If the final_update_step is not None and not greater than or equal to the update_interval (via the base class).
        If the parameter_name is not in the simulation context.
    """

    def __init__(self, filename: str, parameter_name: str, simulation: openmm.app.Simulation,
                 start_value: Union[unit.Quantity, float], update_interval: int = 1, print_interval: int = 1,
                 final_update_step: Optional[int] = None, append_file: bool = False) -> None:
        """Constructor of the GlobalParameterUpdateReporterAbstract class."""
        super().__init__(filename=filename, parameter_name=parameter_name, simulation=simulation,
                         update_interval=update_interval, print_interval=print_interval,
                         final_update_step=final_update_step, append_file=append_file)
        if self._parameter_name not in simulation.context.getParameters():
            raise ValueError(f"The global parameter {self._parameter_name} is not in the simulation context.")
        if isinstance(start_value, unit.Quantity):
            self._start_value = start_value.value_in_unit_system(unit.md_unit_system)
        else:
            self._start_value = start_value
        # Check if the start value of the global parameter matches the value in the OpenMM simulation.
        # If the file is being appended to, this check is not necessary since the simulation was resumed in which case
        # the start value is not necessarily the same as the value in the OpenMM simulation.
        if (not append_file
                and abs(self._start_value - simulation.context.getParameters()[self._parameter_name]) > 1.0e-12):
            warnings.warn("The start value of the global parameter does not match the value in the OpenMM simulation.")
            simulation.context.setParameter(self._parameter_name, self._start_value)
        if not append_file:
            print(f"0,{self._start_value}", file=self._file)

    def set_and_print(self, simulation: openmm.app.Simulation, new_value: float) -> None:
        """
        Update the value of the global parameter in the OpenMM simulation context and print the new parameter value in
        the output csv file.

        :param simulation:
            The OpenMM simulation.
        :type simulation: openmm.app.Simulation
        :param new_value:
            The new value of the global parameter.
        :type new_value: float
        """
        simulation.context.setParameter(self._parameter_name, new_value)
        self.print(simulation, new_value)


class RampUpdateReporter(GlobalParameterUpdateReporterAbstract):
    """
    This class sets up a reporter to linearly change the value of a custom-force-related global parameter in a ramp over
    the course of an OpenMM simulation.

    Both the start and end values of the global parameter are specified on initialization.

    This class creates a csv file that stores the current simulation step and current value of the global parameter
    being updated.

    :param filename:
        The name of the file to write to.
        The filename must end with the .csv extension.
    :type filename: str
    :param parameter_name:
        The name of the global parameter to be updated.
        This must be one of the global parameters passed into any of the OpenMM CustomForce objects.
    :type parameter_name: str
    :param simulation:
        The OpenMM simulation that this reporter will be added to.
        The context of this OpenMM simulation must contain the parameter to be updated.
    :type simulation: openmm.app.Simulation
    :param start_value:
        The start value of the global parameter.
        OpenMM does not store the units of global parameters, so if using a quantity with a unit, the user must make
        sure to pass in a sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type start_value: Union[unit.Quantity, float]
    :param end_value:
        The end value of the global parameter.
        OpenMM does not store the units of global parameters, so if using a quantity with a unit, the user must make
        sure to pass in a sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type end_value: Union[unit.Quantity, float]
    :param final_update_step:
        The final step at which the value of the global parameter will be updated.
        The value must be greater than or equal to the update_interval.
    :type final_update_step: int
    :param update_interval:
        The interval (in time steps) at which the value of the global parameter in the OpenMM simulation is updated.
        The value must be greater than zero.
        Defaults to 1.
    :type update_interval: int
    :param print_interval:
        The interval (in time steps) at which the value of the global parameter in the OpenMM simulation is printed
        to the output csv file.
        The value must be greater than zero.
        Defaults to 1.
    :type print_interval: int
    :param append_file:
        If True, open an existing csv file to append to. If False, create a new file possibly overwriting an already
        existing file.
        Defaults to False.
    :type append_file: bool

    :raises ValueError:
        If the filename does not end with the .csv extension (via the base class).
        If the update_interval is not greater than zero (via the base class).
        If the print_interval is not greater than zero (via the base class).
        If the final_update_step is not greater than or equal to the update_interval (via the base class).
        If the parameter_name is not in the simulation context (via the base class).
        If the start and end values have incompatible units.
    """

    def __init__(self, filename: str, parameter_name: str, simulation: openmm.app.Simulation,
                 start_value: Union[unit.Quantity, float], end_value: Union[unit.Quantity, float],
                 final_update_step: int, update_interval: int = 1, print_interval: int = 1,
                 append_file: bool = False) -> None:
        """Constructor of the RampUpdateReporter class."""
        super().__init__(filename=filename, parameter_name=parameter_name, simulation=simulation,
                         start_value=start_value, update_interval=update_interval, print_interval=print_interval,
                         final_update_step=final_update_step, append_file=append_file)
        # Base class allows final_update_step to be None, but here it must be specified.
        if final_update_step is None:
            raise ValueError("The final update step must be specified for the RampUpdateReporter.")
        if isinstance(end_value, unit.Quantity):
            if not isinstance(start_value, unit.Quantity):
                raise ValueError(f"The start and end values have incompatible units.")
            if not start_value.unit.is_compatible(end_value.unit):
                raise ValueError(f"The start and end values have incompatible units.")
            self._end_value = end_value.value_in_unit_system(unit.md_unit_system)
        else:
            if not isinstance(end_value, type(start_value)):
                raise ValueError(f"The start and end values have incompatible units.")
            self._end_value = end_value

    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        """
        Linearly change the value of a global parameter during the simulation.

        This function is called by OpenMM when the reporter should generate a report.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        :param state:
            The current state of the OpenMM simulation.
        :type state: openmm.State
        """
        old_value = simulation.context.getParameter(self._parameter_name)
        new_value = old_value + (self._end_value - self._start_value) * self._update_interval / self._final_update_step
        self.set_and_print(simulation, new_value)


class TriangleUpdateReporter(GlobalParameterUpdateReporterAbstract):
    """
    This class sets up a reporter to change the value of a custom-force-related global parameter following a triangular
    wave over the course of an OpenMM simulation.

    Both the start and end values of the global parameter during a single increasing or decreasing ramp of the
    triangular wave are specified on initialization. If the end value is greater than the start value, the global
    parameter value increases until the switch step, then decreases back to the start value. Otherwise, the global
    parameter value decreases until the switch step, then increases back to the start value. This is repeated until the
    final update step is reached.

    This class creates a csv file that stores the current simulation step and current value of the global parameter
    being updated.

    :param filename:
        The name of the file to write to.
        The filename must end with the .csv extension.
    :type filename: str
    :param parameter_name:
        The name of the global parameter to be updated.
        This must be one of the global parameters passed into any of the OpenMM CustomForce objects.
    :type parameter_name: str
    :param simulation:
        The OpenMM simulation that this reporter will be added to.
        The context of this OpenMM simulation must contain the parameter to be updated.
    :type simulation: openmm.app.Simulation
    :param start_value:
        The start value of the global parameter.
        OpenMM does not store the units of global parameters, so if using a quantity with a unit, the user must make
        sure to pass in a sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type start_value: Union[unit.Quantity, float]
    :param end_value:
        The end value of the global parameter.
        OpenMM does not store the units of global parameters, so if using a quantity with a unit, the user must make
        sure to pass in a sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type end_value: Union[unit.Quantity, float]
    :param switch_step:
        The number of steps after which this reporter switches from increasing to decreasing (or decreasing to
        increasing) the value of the global parameter.
        The value must be a multiple of the update_interval, and less than or equal to the final_update_step.
    :type switch_step: int
    :param update_interval:
        The interval (in time steps) at which the value of the global parameter in the OpenMM simulation is updated.
        The value must be greater than zero.
        Defaults to 1.
    :type update_interval: int
    :param print_interval:
        The interval (in time steps) at which the value of the global parameter in the OpenMM simulation is printed
        to the output csv file.
        The value must be greater than zero.
        Defaults to 1.
    :type print_interval: int
    :param final_update_step:
        The final step at which the value of the global parameter will be updated.
        If None, the parameter will be updated until the end of the simulation.
        If not None, the value must be greater than or equal to the update_interval.
        Defaults to None.
    :type final_update_step: Optional[int]
    :param append_file:
        If True, open an existing csv file to append to. If False, create a new file possibly overwriting an already
        existing file.
        Defaults to False.
    :type append_file: bool

    :raises ValueError:
        If the filename does not end with the .csv extension (via the base class).
        If the update_interval is not greater than zero (via the base class).
        If the print_interval is not greater than zero (via the base class).
        If the final_update_step is not None and not greater than or equal to the update_interval (via the base class).
        If the parameter_name is not in the simulation context (via the base class).
        If the start and end values have incompatible units.
        If the switch step is not a multiple of the update frequency.
        If the switch step is not less than or equal to the final update step.
    """

    def __init__(self, filename: str, parameter_name: str, simulation: openmm.app.Simulation,
                 start_value: Union[unit.Quantity, float], end_value: Union[unit.Quantity, float], switch_step: int,
                 update_interval: int = 1, print_interval: int = 1, final_update_step: Optional[int] = None,
                 append_file: bool = False):
        super().__init__(filename=filename, parameter_name=parameter_name, simulation=simulation,
                         start_value=start_value, update_interval=update_interval, print_interval=print_interval,
                         final_update_step=final_update_step, append_file=append_file)
        if isinstance(end_value, unit.Quantity):
            if not isinstance(start_value, unit.Quantity):
                raise ValueError(f"The start and end values have incompatible units.")
            if not start_value.unit.is_compatible(end_value.unit):
                raise ValueError(f"The start and end values have incompatible units.")
            self._end_value = end_value.value_in_unit_system(unit.md_unit_system)
        else:
            if not isinstance(end_value, type(start_value)):
                raise ValueError(f"The start and end values have incompatible units.")
            self._end_value = end_value
        if not final_update_step >= switch_step >= update_interval:
            raise ValueError("The switch step must be greater than or equal to the update frequency,"
                             "and less than or equal to the final update step.")
        if not switch_step % update_interval == 0:
            raise ValueError("The switch step must be a multiple of the update frequency.")
        self._switch_step = switch_step

    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        """
        Change the value of a global parameter during the simulation according to a triangular wave.

        This function is called by OpenMM when the reporter should generate a report.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        :param state:
            The current state of the OpenMM simulation.
        :type state: openmm.State
        """
        current_step = simulation.currentStep
        old_value = simulation.context.getParameter(self._parameter_name)
        assert current_step - self._update_interval >= 0
        last_update_remainder = (current_step - self._update_interval) % (2.0 * self._switch_step)
        if last_update_remainder < self._switch_step:
            new_value = old_value + (self._end_value - self._start_value) * self._update_interval / self._switch_step
        else:
            new_value = old_value - (self._end_value - self._start_value) * self._update_interval / self._switch_step
        self.set_and_print(simulation, new_value)


class SquaredSinusoidalUpdateReporter(GlobalParameterUpdateReporterAbstract):
    """
    This class sets up a reporter to change the value of a custom-force-related global parameter following a squared
    sinusoidal wave over the course of an OpenMM simulation.

    Both the start and end values of the global parameter during a single increasing or decreasing part of the squared
    sinusoidal wave are specified on initialization. If the end value is greater than the start value, the global
    parameter value increases until the switch step, then decreases back to the start value. Otherwise, the global
    parameter value decreases until the switch step, then increases back to the start value. This is repeated until the
    final update step is reached.

    This class creates a csv file that stores the current simulation step and current value of the global parameter
    being updated.

    :param filename:
        The name of the file to write to.
        The filename must end with the .csv extension.
    :type filename: str
    :param parameter_name:
        The name of the global parameter to be updated.
        This must be one of the global parameters passed into any of the OpenMM CustomForce objects.
    :type parameter_name: str
    :param simulation:
        The OpenMM simulation that this reporter will be added to.
        The context of this OpenMM simulation must contain the parameter to be updated.
    :type simulation: openmm.app.Simulation
    :param start_value:
        The start value of the global parameter.
        OpenMM does not store the units of global parameters, so if using a quantity with a unit, the user must make
        sure to pass in a sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type start_value: Union[unit.Quantity, float]
    :param end_value:
        The end value of the global parameter.
        OpenMM does not store the units of global parameters, so if using a quantity with a unit, the user must make
        sure to pass in a sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type end_value: Union[unit.Quantity, float]
    :param switch_step:
        The number of steps after which this reporter switches from increasing to decreasing (or decreasing to
        increasing) the value of the global parameter.
        The value must be a multiple of the update_interval, and less than or equal to the final_update_step.
    :type switch_step: int
    :param update_interval:
        The interval (in time steps) at which the value of the global parameter in the OpenMM simulation is updated.
        The value must be greater than zero.
        Defaults to 1.
    :type update_interval: int
    :param print_interval:
        The interval (in time steps) at which the value of the global parameter in the OpenMM simulation is printed
        to the output csv file.
        The value must be greater than zero.
        Defaults to 1.
    :type print_interval: int
    :param final_update_step:
        The final step at which the value of the global parameter will be updated.
        If None, the parameter will be updated until the end of the simulation.
        If not None, the value must be greater than or equal to the update_interval.
        Defaults to None.
    :type final_update_step: Optional[int]
    :param append_file:
        If True, open an existing csv file to append to. If False, create a new file possibly overwriting an already
        existing file.
        Defaults to False.
    :type append_file: bool

    :raises ValueError:
        If the filename does not end with the .csv extension (via the base class).
        If the update_interval is not greater than zero (via the base class).
        If the print_interval is not greater than zero (via the base class).
        If the final_update_step is not None and not greater than or equal to the update_interval (via the base class).
        If the parameter_name is not in the simulation context (via the base class).
        If the start and end values have incompatible units.
        If the switch step is not a multiple of the update frequency.
        If the switch step is not less than or equal to the final update step.
    """

    def __init__(self, filename: str, parameter_name: str, simulation: openmm.app.Simulation,
                 start_value: Union[unit.Quantity, float], end_value: Union[unit.Quantity, float], switch_step: int,
                 update_interval: int = 1, print_interval: int = 1, final_update_step: Optional[int] = None,
                 append_file: bool = False):
        super().__init__(filename=filename, parameter_name=parameter_name, simulation=simulation,
                         start_value=start_value, update_interval=update_interval, print_interval=print_interval,
                         final_update_step=final_update_step, append_file=append_file)
        if isinstance(end_value, unit.Quantity):
            if not isinstance(start_value, unit.Quantity):
                raise ValueError(f"The start and end values have incompatible units.")
            if not start_value.unit.is_compatible(end_value.unit):
                raise ValueError(f"The start value and amplitude have incompatible units.")
            end_value_float = end_value.value_in_unit_system(unit.md_unit_system)
        else:
            if not isinstance(end_value, type(start_value)):
                raise ValueError(f"The start value and amplitude have incompatible units.")
            end_value_float = end_value
        self._amplitude = end_value_float - self._start_value
        if not final_update_step >= switch_step >= update_interval:
            raise ValueError("The switch step must be greater than or equal to the update frequency,"
                             "and less than or equal to the final update step.")
        if not switch_step % update_interval == 0:
            raise ValueError("The switch step must be a multiple of the update frequency.")
        self._period = math.pi / (2.0 * switch_step)

    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        """
        Change the value of a global parameter during the simulation according to a squared sinusoidal wave.

        This function is called by OpenMM when the reporter should generate a report.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        :param state:
            The current state of the OpenMM simulation.
        :type state: openmm.State
        """
        step = simulation.currentStep
        current_value = self._amplitude * (math.sin(self._period * step) ** 2) + self._start_value
        self.set_and_print(simulation, current_value)


class RampTemperatureUpdateReporter(UpdateReporterAbstract):
    """
    This class sets up a reporter to linearly change the value of the temperature in a ramp over the course of an OpenMM
    simulation.

    This update reporter only changes the temperature of the integrator.

    Both the start and end values of the temperature are specified on initialization.

    This class creates a csv file that stores the current simulation step and current value of the temperature.

    :param filename:
        The name of the file to write to.
        The filename must end with the .csv extension.
    :type filename: str
    :param simulation:
        The OpenMM simulation that this reporter will be added to.
        The context of this OpenMM simulation must contain the parameter to be updated.
    :type simulation: openmm.app.Simulation
    :param start_value:
        The start value of the temperature.
        The unit must be compatible with Kelvin and the value must be bigger than zero.
    :type start_value: unit.Quantity
    :param end_value:
        The end value of the temperature.
        The unit must be compatible with Kelvin and the value must be bigger than zero.
    :type end_value: unit.Quantity
    :param final_update_step:
        The final step at which the value of the temperature will be updated.
        The value must be greater than or equal to the update_interval.
    :type final_update_step: int
    :param update_interval:
        The interval (in time steps) at which the value of the temperature in the OpenMM simulation is updated.
        The value must be greater than zero.
        Defaults to 1.
    :type update_interval: int
    :param print_interval:
        The interval (in time steps) at which the value of the temperature in the OpenMM simulation is printed
        to the output csv file.
        The value must be greater than zero.
        Defaults to 1.
    :type print_interval: int
    :param append_file:
        If True, open an existing csv file to append to. If False, create a new file possibly overwriting an already
        existing file.
        Defaults to False.
    :type append_file: bool

    :raises ValueError:
        If the filename does not end with the .csv extension (via the base class).
        If the update_interval is not greater than zero (via the base class).
        If the print_interval is not greater than zero (via the base class).
        If the final_update_step is not greater than or equal to the update_interval (via the base class).
        If the start and end values do not have units compatible with Kelvin.
        If the start and end values are not greater than zero.
    """

    def __init__(self, filename: str, simulation: openmm.app.Simulation, start_value: unit.Quantity,
                 end_value: unit.Quantity, final_update_step: int, update_interval: int = 1, print_interval: int = 1,
                 append_file: bool = False) -> None:
        """Constructor of the RampTemperatureUpdateReporter class."""
        super().__init__(filename=filename, parameter_name="temperature", simulation=simulation,
                         update_interval=update_interval, print_interval=print_interval,
                         final_update_step=final_update_step, append_file=append_file)
        # Base class allows final_update_step to be None, but here it must be specified.
        if final_update_step is None:
            raise ValueError("The final update step must be specified for the RampTemperatureUpdateReporter.")
        if not start_value.unit.is_compatible(temperature_unit):
            raise TypeError("The start value must have a unit that is compatible with Kelvin.")
        if not end_value.unit.is_compatible(temperature_unit):
            raise TypeError("The end value must have a unit that is compatible with Kelvin.")
        if not start_value.value_in_unit(temperature_unit) > 0.0:
            raise ValueError("The start value must be greater than zero.")
        if not end_value.value_in_unit(temperature_unit) > 0.0:
            raise ValueError("The end value must be greater than zero.")
        self._start_value = start_value.value_in_unit(temperature_unit)
        self._end_value = end_value.value_in_unit(temperature_unit)
        # Check if the start value of the temperature matches the value in the OpenMM simulation.
        # If the file is being appended to, this check is not necessary since the simulation was resumed in which case
        # the start value is not necessarily the same as the value in the OpenMM simulation.
        if (not append_file
                and abs(self._start_value
                        - simulation.integrator.getTemperature().value_in_unit(temperature_unit)) > 1.0e-12):
            warnings.warn("The start value of the temperature does not match the value in the OpenMM integrator.")
            simulation.integrator.setTemperature(self._start_value)
        if not append_file:
            print(f"0,{self._start_value}", file=self._file)

    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        """
        Linearly change the value of a global parameter during the simulation.

        This function is called by OpenMM when the reporter should generate a report.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        :param state:
            The current state of the OpenMM simulation.
        :type state: openmm.State
        """
        old_value = simulation.integrator.getTemperature().value_in_unit(temperature_unit)
        new_value = old_value + (self._end_value - self._start_value) * self._update_interval / self._final_update_step
        step = simulation.currentStep
        simulation.integrator.setTemperature(new_value)
        if step % self._print_interval == 0:
            print(f"{step},{new_value}", file=self._file, flush=True)


class RampUpdateReporterUntilCluster(GlobalParameterUpdateReporterAbstract):
    """
    This class sets up a reporter to linearly change the value of a custom-force-related global parameter in a ramp over
    the course of an OpenMM simulation until a cluster of particle formed.

    This update reporter can be used, e.g., to increase the Debye length until the interaction is large enough to
    form a cluster of particles. The reporter will then stop updating the Debye length.

    The cluster formation is checked every check_interval time steps. A cluster is defined as a group of particles
    where each particle in the cluster is within the cutoff distance of at least one other particle in the cluster.
    The size of the largest cluster is compared to the specified cluster_size. One can optionally ignore certain
    atom types when checking for cluster formation (as, e.g., immobile substrate atoms).

    Both the start and end values of the global parameter are specified on initialization. They specify the slope of
    the ramp. If a cluster of particles is formed before the end value is reached, the reporter will stop updating
    the global parameter.

    This class creates a csv file that stores the current simulation step and current value of the global parameter
    being updated.

    :param filename:
        The name of the file to write to.
        The filename must end with the .csv extension.
    :type filename: str
    :param parameter_name:
        The name of the global parameter to be updated.
        This must be one of the global parameters passed into any of the OpenMM CustomForce objects.
    :type parameter_name: str
    :param simulation:
        The OpenMM simulation that this reporter will be added to.
        The context of this OpenMM simulation must contain the parameter to be updated.
    :type simulation: openmm.app.Simulation
    :param start_value:
        The start value of the global parameter.
        OpenMM does not store the units of global parameters, so if using a quantity with a unit, the user must make
        sure to pass in a sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type start_value: Union[unit.Quantity, float]
    :param end_value:
        The end value of the global parameter.
        OpenMM does not store the units of global parameters, so if using a quantity with a unit, the user must make
        sure to pass in a sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type end_value: Union[unit.Quantity, float]
    :param final_update_step:
        The final step at which the value of the global parameter will be updated.
        The value must be greater than or equal to the update_interval.
    :type final_update_step: int
    :param cluster_size:
        The number of particles that must be in the largest cluster to stop updating the global parameter.
        The value must be greater than zero.
    :type cluster_size: int
    :param cutoff_distance:
        The cutoff distance to use when determining if two particles are part of the same cluster.
        The unit must be compatible with nanometers and the value must be bigger than zero.
    :type cutoff_distance: unit.Quantity
    :param update_interval:
        The interval (in time steps) at which the value of the global parameter in the OpenMM simulation is updated.
        The value must be greater than zero.
        Defaults to 1.
    :type update_interval: int
    :param print_interval:
        The interval (in time steps) at which the value of the global parameter in the OpenMM simulation is printed
        to the output csv file.
        The value must be greater than zero.
        Defaults to 1.
    :type print_interval: int
    :param check_interval:
        The interval (in time steps) at which the simulation is checked for the formation of a
        cluster of the specified size.
        The value must be greater than zero.
        Defaults to 1.
    :type check_interval: int
    :param append_file:
        If True, open an existing csv file to append to. If False, create a new file possibly overwriting an already
        existing file.
        Defaults to False.
    :type append_file: bool
    :param ignore_types:
        A list of atom types to ignore when checking for cluster formation.
        If None, all atom types are considered when checking for cluster formation.
        Defaults to None.
    :type ignore_types: Optional[Sequence[str]]

    :raises ValueError:
        If the filename does not end with the .csv extension (via the base class).
        If the update_interval is not greater than zero (via the base class).
        If the print_interval is not greater than zero (via the base class).
        If the final_update_step is not greater than or equal to the update_interval (via the base class).
        If the parameter_name is not in the simulation context (via the base class).
        If the start and end values have incompatible units.
        If the check_interval is not greater than zero.
        If the cluster_size is not greater than zero.
        If the cutoff_distance does not have a unit compatible with nanometers.
        If the cutoff_distance is not greater than zero.
        If any of the ignore_types are not in the simulation.
    """

    def __init__(self, filename: str, parameter_name: str, simulation: openmm.app.Simulation,
                 start_value: unit.Quantity, end_value: unit.Quantity, final_update_step: int,
                 cluster_size: int, cutoff_distance: unit.Quantity, update_interval: int = 1, print_interval: int = 1,
                 check_interval: int = 1, append_file: bool = False, ignore_types: Optional[Sequence[str]] = None):
        """Constructor of the RampUpdateReporterUntilCluster class."""
        super().__init__(filename=filename, parameter_name=parameter_name, simulation=simulation,
                         start_value=start_value, update_interval=update_interval, print_interval=print_interval,
                         final_update_step=final_update_step, append_file=append_file)
        # Base class allows final_update_step to be None, but here it must be specified.
        if final_update_step is None:
            raise ValueError("The final update step must be specified for the RampUpdateReporter.")
        if isinstance(end_value, unit.Quantity):
            if not isinstance(start_value, unit.Quantity):
                raise ValueError(f"The start and end values have incompatible units.")
            if not start_value.unit.is_compatible(end_value.unit):
                raise ValueError(f"The start and end values have incompatible units.")
            self._end_value = end_value.value_in_unit_system(unit.md_unit_system)
        else:
            if not isinstance(end_value, type(start_value)):
                raise ValueError(f"The start and end values have incompatible units.")
            self._end_value = end_value
        if not check_interval > 0:
            raise ValueError("The check frequency must be greater than zero.")
        if not cluster_size > 0:
            raise ValueError("The cluster size must be greater than zero.")
        if not cutoff_distance.unit.is_compatible(length_unit):
            raise TypeError("The cutoff distance must have a unit that is compatible with nanometers.")
        if not cutoff_distance.value_in_unit(length_unit) > 0.0:
            raise ValueError("The cutoff distance must be greater than zero.")
        self._check_interval = check_interval
        self._cluster_size = cluster_size
        self._cutoff_distance = cutoff_distance.value_in_unit(length_unit)
        self._cluster_reached = False
        self._ignore_types = ignore_types
        self._mask = np.ones(simulation.topology.getNumAtoms(), dtype=bool)
        if ignore_types is not None:
            atom_names = set(atom.name for atom in simulation.topology.atoms())
            for t in ignore_types:
                if t not in atom_names:
                    raise ValueError(f"The atom type {t} is not in the simulation.")
            for index, atom in enumerate(simulation.topology.atoms()):
                if atom.name in ignore_types:
                    self._mask[index] = False
        self._freud_cluster = freud.cluster.Cluster()
        if simulation.system.usesPeriodicBoundaryConditions():
            box_vectors = simulation.system.getDefaultPeriodicBoxVectors()
            box_matrix = [list(bv.value_in_unit(length_unit)) for bv in box_vectors]
            self._freud_box = freud.box.Box.from_matrix(box_matrix)
            self._freud_box.periodic = True
        else:
            # Without periodic boundary conditions, the box dimensions do not matter for cluster computation.
            self._freud_box = freud.box.Box(Lx=1.0, Ly=1.0, Lz=1.0)
            self._freud_box.periodic = False

    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        """
        Linearly change the value of a global parameter during the simulation until a cluster is formed.

        This function is called by OpenMM when the reporter should generate a report.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        :param state:
            The current state of the OpenMM simulation.
        :type state: openmm.State
        """
        step = simulation.currentStep
        if step % self._check_interval == 0 and not self._cluster_reached:
            state = simulation.context.getState(getPositions=True)
            positions = state.getPositions(asNumpy=True).value_in_unit(length_unit)[self._mask]
            if self._freud_box.periodic[0] or self._freud_box.periodic[1] or self._freud_box.periodic[2]:
                positions = freud.box.Box.from_box(self._freud_box).wrap(positions)
            self._freud_cluster.compute((self._freud_box, positions),
                                        neighbors={'r_max': self._cutoff_distance, "exclude_ii": True})
            # noinspection PyTypeChecker
            unique, counts = np.unique(self._freud_cluster.cluster_idx, return_counts=True)
            biggest_cluster_size = counts.max()
            if biggest_cluster_size >= self._cluster_size:
                self._cluster_reached = True

        old_value = simulation.context.getParameter(self._parameter_name)
        if not self._cluster_reached:
            new_value = (old_value
                         + (self._end_value - self._start_value) * self._update_interval / self._final_update_step)
            self.set_and_print(simulation, new_value)
        else:
            if step % self._print_interval == 0:
                print(f"{step},{old_value}", file=self._file, flush=True)
