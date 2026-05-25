from typing import Optional
import warnings
import openmm
from openmm import unit


# noinspection PyPep8Naming
def BrownianIntegrator(temperature: unit.Quantity, frictionCoeff: unit.Quantity, stepSize: unit.Quantity,
                       randomNumberSeed: Optional[int] = None) -> openmm.Integrator:
    """
    Function to return the OpenMM Brownian integrator that defines the keyword arguments (in contrast to OpenMM).

    The following is the OpenMM documentation for the Brownian integrator (see
    https://docs.openmm.org/latest/api-python/generated/openmm.openmm.BrownianIntegrator.html).

    This is an Integrator which simulates a System using Brownian dynamics.

    :param temperature:
        The temperature of the heat bath (in Kelvin).
    :type temperature: unit.Quantity
    :param frictionCoeff:
        The friction coefficient which couples the system to the heat bath, measured in 1/ps.
    :type frictionCoeff: unit.Quantity
    :param stepSize:
        The step size with which to integrate the system (in picoseconds).
    :type stepSize: unit.Quantity
    :param randomNumberSeed:
        Set the random number seed. The precise meaning of this parameter is undefined, and is left up to each Platform
        to interpret in an appropriate way. It is guaranteed that if two simulations are run with different random
        number seeds, the sequence of random forces will be different. On the other hand, no guarantees are made about
        the behavior of simulations that use the same seed. In particular, Platforms are permitted to use
        non-deterministic algorithms which produce different results on successive runs, even if those runs were
        initialized identically.
        If seed is set to 0 or None, a unique seed is chosen when a Context is created from this Integrator. This is
        done to ensure that each Context receives unique random seeds without you needing to set them explicitly.
        Defaults to None.
    :type randomNumberSeed: Optional[int]

    :return:
        The Brownian integrator.
    :rtype: openmm.Integrator
    """
    # Checks of units and values are done within OpenMM.
    integrator = openmm.BrownianIntegrator(temperature, frictionCoeff, stepSize)
    if randomNumberSeed is not None:
        if randomNumberSeed == 0:
            warnings.warn(f"The random number seed for the Brownian integrator is set to 0 which, possibly unexpected, "
                          f"results in a unique seed being chosen when a Context is created from this integrator.")
        integrator.setRandomNumberSeed(randomNumberSeed)
    return integrator


# noinspection PyPep8Naming
def LangevinIntegrator(temperature: unit.Quantity, frictionCoeff: unit.Quantity, stepSize: unit.Quantity,
                       randomNumberSeed: Optional[int] = None) -> openmm.Integrator:
    """
    Function to return the OpenMM Langevin integrator that defines the keyword arguments (in contrast to OpenMM).

    The following is the OpenMM documentation for the Langevin integrator (see
    https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LangevinIntegrator.html).

    This is an Integrator which simulates a System using Langevin dynamics.

    :param temperature:
        The temperature of the heat bath (in Kelvin).
    :type temperature: unit.Quantity
    :param frictionCoeff:
        The friction coefficient which couples the system to the heat bath (in inverse picoseconds).
    :type frictionCoeff: unit.Quantity
    :param stepSize:
        The step size with which to integrate the system (in picoseconds).
    :type stepSize: unit.Quantity
    :param randomNumberSeed:
        Set the random number seed. The precise meaning of this parameter is undefined, and is left up to each Platform
        to interpret in an appropriate way. It is guaranteed that if two simulations are run with different random
        number seeds, the sequence of random forces will be different. On the other hand, no guarantees are made about
        the behavior of simulations that use the same seed. In particular, Platforms are permitted to use
        non-deterministic algorithms which produce different results on successive runs, even if those runs were
        initialized identically.
        If seed is set to 0 or None, a unique seed is chosen when a Context is created from this Integrator. This is
        done to ensure that each Context receives unique random seeds without you needing to set them explicitly.
        Defaults to None.
    :type randomNumberSeed: Optional[int]

    :return:
        The Langevin integrator.
    :rtype: openmm.Integrator
    """
    # Checks of units and values are done within OpenMM.
    integrator = openmm.LangevinIntegrator(temperature, frictionCoeff, stepSize)
    if randomNumberSeed is not None:
        if randomNumberSeed == 0:
            warnings.warn(f"The random number seed for the Langevin integrator is set to 0 which, possibly unexpected, "
                          f"results in a unique seed being chosen when a Context is created from this integrator.")
        integrator.setRandomNumberSeed(randomNumberSeed)
    return integrator


# noinspection PyPep8Naming
def LangevinMiddleIntegrator(temperature: unit.Quantity, frictionCoeff: unit.Quantity, stepSize: unit.Quantity,
                             randomNumberSeed: Optional[int] = None) -> openmm.Integrator:
    """
    Function to return the OpenMM Langevin middle integrator that defines the keyword arguments (in contrast to OpenMM).

    The following is the OpenMM documentation for the Langevin middle integrator (see
    https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LangevinMiddleIntegrator.html).

    This is an Integrator which simulates a System using Langevin dynamics, with the LFMiddle discretization
    (J. Phys. Chem. A 2019, 123, 28, 6056-6079). This method tend to produce more accurate configurational sampling
    than other discretizations, such as the one used in LangevinIntegrator.

    The algorithm is closely related to the BAOAB discretization (Proc. R. Soc. A. 472: 20160138). Both methods produce
    identical trajectories, but LFMiddle returns half step (leapfrog) velocities, while BAOAB returns on-step
    velocities. The former provide a much more accurate sampling of the thermal ensemble.

    :param temperature:
        The temperature of the heat bath (in Kelvin).
    :type temperature: unit.Quantity
    :param frictionCoeff:
        The friction coefficient which couples the system to the heat bath (in inverse picoseconds).
    :type frictionCoeff: unit.Quantity
    :param stepSize:
        The step size with which to integrate the system (in picoseconds).
    :type stepSize: unit.Quantity
    :param randomNumberSeed:
        Set the random number seed. The precise meaning of this parameter is undefined, and is left up to each Platform
        to interpret in an appropriate way. It is guaranteed that if two simulations are run with different random
        number seeds, the sequence of random forces will be different. On the other hand, no guarantees are made about
        the behavior of simulations that use the same seed. In particular, Platforms are permitted to use
        non-deterministic algorithms which produce different results on successive runs, even if those runs were
        initialized identically.
        If seed is set to 0 or None, a unique seed is chosen when a Context is created from this Integrator. This is
        done to ensure that each Context receives unique random seeds without you needing to set them explicitly.
        Defaults to None.
    :type randomNumberSeed: Optional[int]

    :return:
        The Langevin middle integrator.
    :rtype: openmm.Integrator
    """
    # Checks of units and values are done within OpenMM.
    integrator = openmm.LangevinMiddleIntegrator(temperature, frictionCoeff, stepSize)
    if randomNumberSeed is not None:
        if randomNumberSeed == 0:
            warnings.warn(f"The random number seed for the Langevin middle integrator is set to 0 which, possibly "
                          f"unexpected, results in a unique seed being chosen when a Context is created from this "
                          f"integrator.")
        integrator.setRandomNumberSeed(randomNumberSeed)
    return integrator


# noinspection PyPep8Naming
def NoseHooverIntegrator(temperature: unit.Quantity, collisionFrequency: unit.Quantity, stepSize: unit.Quantity,
                         chainLength: int = 3, numMTS: int = 3, numYoshidaSuzuki: int = 7) -> openmm.Integrator:
    """
    Function to return the OpenMM Nose-Hoover integrator that defines the keyword arguments (in contrast to OpenMM).

    The following is the OpenMM documentation for the Nose-Hoover integrator (see
    https://docs.openmm.org/latest/api-python/generated/openmm.openmm.NoseHooverIntegrator.html).

    This is an Integrator which simulates a System using one or more Nose Hoover chain thermostats, using the "middle"
    leapfrog propagation algorithm described in J. Phys. Chem. A 2019, 123, 6056-6079.

    :param temperature:
        The target temperature for the system (in Kelvin).
    :type temperature: unit.Quantity
    :param collisionFrequency:
        The frequency of the interaction with the heat bath (in inverse picoseconds).
    :type collisionFrequency: unit.Quantity
    :param stepSize:
        The step size with which to integrate the system (in picoseconds).
    :type stepSize: unit.Quantity
    :param chainLength:
        The number of beads in the Nose-Hoover chain.
        Defaults to 3.
    :type chainLength: int
    :param numMTS:
        The number of step in the multiple time step chain propagation algorithm.
        Defaults to 3.
    :type numMTS: int
    :param numYoshidaSuzuki:
        The number of terms in the Yoshida-Suzuki multi time step decomposition used in the chain propagation algorithm
        (must be 1, 3, 5, or 7).
        Defaults to 7.
    :type numYoshidaSuzuki: int

    :return:
        The Nose-Hoover integrator.
    :rtype: openmm.Integrator
    """
    # Checks of units and values are done within OpenMM.
    return openmm.NoseHooverIntegrator(temperature, collisionFrequency, stepSize, chainLength, numMTS,
                                       numYoshidaSuzuki)


# noinspection PyPep8Naming
def VariableLangevinIntegrator(temperature: unit.Quantity, frictionCoeff: unit.Quantity, errorTol: float,
                               maximumStepSize: Optional[unit.Quantity] = None,
                               randomNumberSeed: Optional[int] = None) -> openmm.Integrator:
    """
    Function to return the OpenMM variable Langevin integrator that defines the keyword arguments (in contrast to
    OpenMM).

    The following is the OpenMM documentation for the variable Langevin integrator (see
    https://docs.openmm.org/latest/api-python/generated/openmm.openmm.VariableLangevinIntegrator.html).

    This is an error controlled, variable time step Integrator that simulates a System using Langevin dynamics. It
    compares the result of the Langevin integrator to that of an explicit Euler integrator, takes the difference between
    the two as a measure of the integration error in each time step, and continuously adjusts the step size to keep the
    error below a specified tolerance. This both improves the stability of the integrator and allows it to take larger
    steps on average, while still maintaining comparable accuracy to a fixed step size integrator.

    It is best not to think of the error tolerance as having any absolute meaning. It is just an adjustable parameter
    that affects the step size and integration accuracy. You should try different values to find the largest one that
    produces a trajectory sufficiently accurate for your purposes. 0.001 is often a good starting point.

    You can optionally set a maximum step size it will ever use. This is useful to prevent it from taking excessively
    large steps in usual situations, such as when the system is right at a local energy minimum.

    :param temperature:
        The temperature of the heat bath (in Kelvin).
    :type temperature: unit.Quantity
    :param frictionCoeff:
        The friction coefficient which couples the system to the heat bath (in inverse picoseconds).
    :type frictionCoeff: unit.Quantity
    :param errorTol:
        The error tolerance.
    :type errorTol: float
    :param maximumStepSize:
        The maximum step size the integrator will ever use, in ps.
        If None, the integrator will not have a maximum step size.
        Defaults to None.
    :type maximumStepSize: Optional[unit.Quantity]
    :param randomNumberSeed:
        Set the random number seed. The precise meaning of this parameter is undefined, and is left up to each Platform
        to interpret in an appropriate way. It is guaranteed that if two simulations are run with different random
        number seeds, the sequence of random forces will be different. On the other hand, no guarantees are made about
        the behavior of simulations that use the same seed. In particular, Platforms are permitted to use
        non-deterministic algorithms which produce different results on successive runs, even if those runs were
        initialized identically.
        If seed is set to 0 or None, a unique seed is chosen when a Context is created from this Integrator. This is
        done to ensure that each Context receives unique random seeds without you needing to set them explicitly.
        Defaults to None.
    :type randomNumberSeed: Optional[int]

    :return:
        The variable Langevin integrator.
    :rtype: openmm.Integrator
    """
    # Checks of units and values are done within OpenMM.
    integrator = openmm.VariableLangevinIntegrator(temperature, frictionCoeff, errorTol)
    if maximumStepSize is not None:
        integrator.setMaximumStepSize(maximumStepSize)
    if randomNumberSeed is not None:
        if randomNumberSeed == 0:
            warnings.warn(f"The random number seed for the variable Langevin integrator is set to 0 which, possibly "
                          f"unexpected, results in a unique seed being chosen when a Context is created from this "
                          f"integrator.")
        integrator.setRandomNumberSeed(randomNumberSeed)
    return integrator


# noinspection PyPep8Naming
def VariableVerletIntegrator(errorTol: float, maximumStepSize: Optional[unit.Quantity] = None) -> openmm.Integrator:
    """
    Function to return the OpenMM variable Verlet integrator that defines the keyword arguments (in contrast to OpenMM).

    The following is the OpenMM documentation for the variable Verlet integrator (see
    https://docs.openmm.org/latest/api-python/generated/openmm.openmm.VariableVerletIntegrator.html).

    This is an error controlled, variable time step Integrator that simulates a System using the leap-frog Verlet
    algorithm. It compares the result of the Verlet integrator to that of an explicit Euler integrator, takes the
    difference between the two as a measure of the integration error in each time step, and continuously adjusts the
    step size to keep the error below a specified tolerance. This both improves the stability of the integrator and
    allows it to take larger steps on average, while still maintaining comparable accuracy to a fixed step size
    integrator.

    It is best not to think of the error tolerance as having any absolute meaning. It is just an adjustable parameter
    that affects the step size and integration accuracy. You should try different values to find the largest one that
    produces a trajectory sufficiently accurate for your purposes. 0.001 is often a good starting point.

    Unlike a fixed step size Verlet integrator, variable step size Verlet is not symplectic. This means that at a given
    accuracy level, energy is not as precisely conserved over long time periods. This makes it most appropriate for
    constant temperate simulations. In constant energy simulations where precise energy conservation over long time
    periods is important, a fixed step size Verlet integrator may be more appropriate.

    You can optionally set a maximum step size it will ever use. This is useful to prevent it from taking excessively
    large steps in usual situations, such as when the system is right at a local energy minimum.

    :param errorTol:
        The error tolerance.
    :type errorTol: float
    :param maximumStepSize:
        The maximum step size the integrator will ever use, in ps.
        If None, the integrator will not have a maximum step size.
        Defaults to None.
    :type maximumStepSize: Optional[unit.Quantity]

    :return:
        The variable Verlet integrator.
    :rtype: openmm.Integrator
    """
    # Checks of units and values are done within OpenMM.
    integrator = openmm.VariableVerletIntegrator(errorTol)
    if maximumStepSize is not None:
        integrator.setMaximumStepSize(maximumStepSize)
    return integrator


# noinspection PyPep8Naming
def MonteCarloBarostat(temperature: unit.Quantity, pressure: unit.Quantity, frequency: int = 25) -> openmm.Force:
    """
    Function to return the OpenMM Monte Carlo barostat that defines the keyword arguments (in contrast to OpenMM).

    The following is the OpenMM documentation for the Monte Carlo barostat (see
    https://docs.openmm.org/latest/api-python/generated/openmm.openmm.MonteCarloBarostat.html).

    :param temperature:
        The temperature of the heat bath (in Kelvin).
    :type temperature: unit.Quantity
    :param pressure:
        The target pressure for the Monte Carlo barostat.
    :type pressure: unit.Quantity
    :param frequency:
        The number of integration steps between Monte Carlo barostat attempts.
    :type frequency: int

    :return:
        The Monte Carlo barostat.
    :rtype: openmm.Force
    """
    # Checks of units and values are done within OpenMM.
    return openmm.MonteCarloBarostat(pressure, temperature, frequency)


# noinspection PyPep8Naming
def MonteCarloAnisotropicBarostat(temperature: unit.Quantity, pressure_x: unit.Quantity, pressure_y: unit.Quantity,
                                  pressure_z: unit.Quantity, scale_x: bool = True, scale_y: bool = True,
                                  scale_z: bool = True, frequency: int = 25) -> openmm.Force:
    """
    Function to return the OpenMM Monte Carlo anisotropic barostat that defines the keyword arguments.

    The following is the OpenMM documentation for the Monte Carlo anisotropic barostat (see
    https://docs.openmm.org/latest/api-python/generated/openmm.openmm.MonteCarloAnisotropicBarostat.html).

    :param temperature:
        The temperature of the heat bath (in Kelvin).
    :type temperature: unit.Quantity
    :param pressure_x:
        The pressure applied to the x axis.
    :type pressure_x: unit.Quantity
    :param pressure_y:
        The pressure applied to the y axis.
    :type pressure_y: unit.Quantity
    :param pressure_z:
        The pressure applied to the z axis.
    :type pressure_z: unit.Quantity
    :param scale_x:
        Whether the x dimension of the periodic box may change size.
    :type scale_x: bool
    :param scale_y:
        Whether the y dimension of the periodic box may change size.
    :type scale_y: bool
    :param scale_z:
        Whether the z dimension of the periodic box may change size.
    :type scale_z: bool
    :param frequency:
        The number of integration steps between Monte Carlo barostat attempts.
    :type frequency: int

    :return:
        The Monte Carlo anisotropic barostat.
    :rtype: openmm.Force
    """
    # Checks of units and values are done within OpenMM.
    pressure = openmm.Vec3(pressure_x.value_in_unit(unit.bar), pressure_y.value_in_unit(unit.bar),
                           pressure_z.value_in_unit(unit.bar))
    return openmm.MonteCarloAnisotropicBarostat(pressure, temperature, scale_x, scale_y, scale_z, frequency)


# noinspection PyPep8Naming
def VerletIntegrator(stepSize: unit.Quantity) -> openmm.Integrator:
    """
    Function to return the OpenMM Verlet integrator that defines the keyword arguments (in contrast to OpenMM).

    The following is the OpenMM documentation for the Verlet integrator (see
    https://docs.openmm.org/latest/api-python/generated/openmm.openmm.VerletIntegrator.html).

    This is an Integrator which simulates a System using the leap-frog Verlet algorithm.

    :param stepSize:
        The step size with which to integrate the system (in picoseconds).
    :type stepSize: unit.Quantity

    :return:
        The Verlet integrator.
    :rtype: openmm.Integrator
    """
    # Checks of units and values are done within OpenMM.
    return openmm.VerletIntegrator(stepSize)


INTEGRATORS = {
    "BrownianIntegrator": BrownianIntegrator,
    "LangevinIntegrator": LangevinIntegrator,
    "LangevinMiddleIntegrator": LangevinMiddleIntegrator,
    "MonteCarloAnisotropicBarostat": MonteCarloAnisotropicBarostat,
    "MonteCarloBarostat": MonteCarloBarostat,
    "NoseHooverIntegrator": NoseHooverIntegrator,
    "VariableLangevinIntegrator": VariableLangevinIntegrator,
    "VariableVerletIntegrator": VariableVerletIntegrator,
    "VerletIntegrator": VerletIntegrator,
}
