import os
import pytest
import numpy as np
import openmm
import openmm.app as app
from openmm import unit, LangevinIntegrator
import gsd.hoomd as gsd


from colloids.colloids_run import colloids_run

from colloids.enhanced_sampling import UmbrellaSamplingPotential
from colloids.units import energy_unit



class TestUmbrellaSamplingParameters(object):
    @pytest.fixture
    def force_constant(self):
        return 100.0 * unit.kilojoule_per_mole
    
    @pytest.fixture
    def center(self):
        return 1.0
    
    @pytest.fixture
    def cv_force(self):
        cv_force = openmm.CustomExternalForce("x")
        cv_force.addParticle(0, [])
        return cv_force
    
    @pytest.fixture
    def box_length(self):
        return 1000.0 * (unit.nano * unit.meter)
    
    @pytest.fixture
    def temperature(self):
        return 298.0 * unit.kelvin

    @pytest.fixture
    def openmm_system(self, box_length):
        system = openmm.System()
        system.setDefaultPeriodicBoxVectors(openmm.Vec3(box_length, 0.0, 0.0),
                                            openmm.Vec3(0.0, box_length, 0.0),
                                            openmm.Vec3(0.0, 0.0, box_length))
        
        system.addParticle(1.0)
        #system.addParticle(1.0)
        return system
    
    @pytest.fixture
    def openmm_topology(self):
        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("colloid", chain)
        topology.addAtom("1", None, residue)
        return topology

    @pytest.fixture
    def openmm_dummy_integrator(self):
        return LangevinIntegrator(0.0, 0.0, 0.0)


class TestUmbrellaExceptions(TestUmbrellaSamplingParameters):

    def test_exception_force_constant_unit(self, cv_force, center):
        with pytest.raises(TypeError):
             UmbrellaSamplingPotential(name="test_force", cv_force=cv_force, center=center, 
                                       force_constant=100.0 * unit.nanometer)


class TestUmbrellaSampling(TestUmbrellaSamplingParameters):
    @pytest.fixture(autouse=True)
    def change_test_dir(self, request, monkeypatch):
        # Change the working directory to the directory of the test file.
        # See https://stackoverflow.com/questions/62044541/change-pytest-working-directory-to-test-case-directory
        monkeypatch.chdir(request.fspath.dirname)

    @staticmethod
    def compute_expected_harmonic_restraint(x, center, force_constant):
        return 0.5*force_constant*(x-center)**2


    def test_harmonic_restraint(self, openmm_system, openmm_topology, cv_force, center, 
                                force_constant, openmm_dummy_integrator): #, yaml_file):
        

        umbrella = UmbrellaSamplingPotential(name="x_position", cv_force=cv_force,
            center=center,force_constant=force_constant)

        umbrella_force = umbrella.yield_potentials()
        umbrella_force.setForceGroup(1)
        openmm_system.addForce(umbrella_force)

        simulation = app.Simulation(openmm_topology, openmm_system, openmm_dummy_integrator)

        x_position = 0.35
        simulation.context.setPositions([[x_position, 0.0, 0.0]] * unit.nanometer)

        state = simulation.context.getState(getEnergy=True,groups={1},)

        actual_energy = state.getPotentialEnergy().value_in_unit(energy_unit)
        expected_energy = self.compute_expected_harmonic_restraint(x_position, center, force_constant.value_in_unit(energy_unit))
        
        assert actual_energy == pytest.approx(expected_energy,rel=1.0e-7,abs=1.0e-7,)

    def test_umbrella_force_metadata(self):
        cv_force = openmm.CustomExternalForce("x")
        cv_force.addParticle(0, [])

        umbrella = UmbrellaSamplingPotential(name="x_position", cv_force=cv_force,
            center=0.25,force_constant=100.0 * energy_unit)

        umbrella_force = umbrella.yield_potentials()

        assert umbrella_force.getName() == "umbrella_x_position"
        assert umbrella_force.getNumCollectiveVariables() == 1
        assert umbrella_force.getCollectiveVariableName(0) == "cv"

if __name__ == '__main__':
    pytest.main([__file__])

