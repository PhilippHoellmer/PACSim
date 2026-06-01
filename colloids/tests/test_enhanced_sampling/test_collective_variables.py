import os
import pytest
import numpy as np
import openmm
import openmm.app as app
from openmm import unit
import gsd.hoomd as gsd

from colloids.colloids_run import colloids_run
from colloids.collective_variables import HighCoordCompositionCV


class TestCollectiveVariableParameters(object):
    @pytest.fixture
    def target_particle_type(self):
        return "2"
    
    @pytest.fixture
    def coordination_d0(self):
        return 300.0 * (unit.nano * unit.meter)
    
    @pytest.fixture
    def coordination_r0(self):
        return 50.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def coordination_dmax(self):
        return 400.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def highcoord_threshold(self):
        return 12.0
    
    @pytest.fixture
    def box_length(self):
        return 11200.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def openmm_system(self, box_length):
        system = openmm.System()
        system.setDefaultPeriodicBoxVectors(openmm.Vec3(box_length, 0.0, 0.0),
                                            openmm.Vec3(0.0, box_length, 0.0),
                                            openmm.Vec3(0.0, 0.0, box_length))
        
        system.addParticle(1.0)
        system.addParticle(1.0)
        return system
    
    @pytest.fixture
    def openmm_topology(self):
        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("colloid", chain)
        topology.addAtom("1", None, residue)
        topology.addAtom("2", None, residue)
        return topology
    
    @pytest.fixture
    def highcoordcomp_cv_force(self, openmm_topology, openmm_system, target_particle_type, coordination_d0, 
                               coordination_r0, coordination_dmax, highcoord_threshold):
        return HighCoordCompositionCV(openmm_topology, openmm_system, target_particle_type, coordination_d0, 
                               coordination_r0, coordination_dmax, highcoord_threshold)

class TestCollectiveVariablesExceptions(TestCollectiveVariableParameters):

    def test_exception_target_particle_type(self, openmm_topology, openmm_system):
        with pytest.raises(ValueError):
             HighCoordCompositionCV(topology=openmm_topology, system=openmm_system,
                target_particle_type="fake_particle",coordination_d0=300.0 * unit.nanometer,
                coordination_r0=50.0 * unit.nanometer,coordination_dmax=400.0 * unit.nanometer,
                highcoord_threshold=12.0,
            )

    def test_exception_highcoord_threshold(self, openmm_topology, openmm_system):
        with pytest.raises(ValueError):
             HighCoordCompositionCV(topology=openmm_topology, system=openmm_system,
                target_particle_type="2",coordination_d0=300.0 * unit.nanometer,
                coordination_r0=50.0 * unit.nanometer,coordination_dmax=400.0 * unit.nanometer,
                highcoord_threshold=0.0,
            )

    def test_exception_coordination_d0_wrong_unit(self, openmm_topology, openmm_system):
        with pytest.raises(TypeError):
             HighCoordCompositionCV(topology=openmm_topology, system=openmm_system,
                target_particle_type="2",coordination_d0=300.0 * unit.kilojoule_per_mole,
                coordination_r0=50.0 * unit.nanometer,coordination_dmax=400.0 * unit.nanometer,
                highcoord_threshold=12.0,
            )

    def test_exception_coordination_r0_wrong_unit(self, openmm_topology, openmm_system):
        with pytest.raises(TypeError):
             HighCoordCompositionCV(topology=openmm_topology, system=openmm_system,
                target_particle_type="2",coordination_d0=300.0 * unit.nanometer,
                coordination_r0=50.0 * unit.kilojoule_per_mole,coordination_dmax=400.0 * unit.nanometer,
                highcoord_threshold=12.0,
            )

    def test_exception_coordination_dmax_wrong_unit(self, openmm_topology, openmm_system):
        with pytest.raises(TypeError):
             HighCoordCompositionCV(topology=openmm_topology, system=openmm_system,
                target_particle_type="2",coordination_d0=300.0 * unit.nanometer,
                coordination_r0=50.0 * unit.nanometer,coordination_dmax=400.0 * unit.kilojoule_per_mole,
                highcoord_threshold=12.0,
            )

    def test_exception_coordination_d0_nonpositive(self, openmm_topology, openmm_system):
        with pytest.raises(ValueError):
             HighCoordCompositionCV(topology=openmm_topology, system=openmm_system,
                target_particle_type="2",coordination_d0=-300.0 * unit.nanometer,
                coordination_r0=50.0 * unit.nanometer,coordination_dmax=400.0 * unit.nanometer,
                highcoord_threshold=12.0,
            )

    def test_exception_coordination_r0_nonpositive(self, openmm_topology, openmm_system):
        with pytest.raises(ValueError):
             HighCoordCompositionCV(topology=openmm_topology, system=openmm_system,
                target_particle_type="2",coordination_d0=300.0 * unit.nanometer,
                coordination_r0=-50.0 * unit.nanometer,coordination_dmax=400.0 * unit.nanometer,
                highcoord_threshold=12.0,
            )
    
    def test_exception_coordination_dmax_nonpositive(self, openmm_topology, openmm_system):
        with pytest.raises(ValueError):
             HighCoordCompositionCV(topology=openmm_topology, system=openmm_system,
                target_particle_type="2",coordination_d0=300.0 * unit.nanometer,
                coordination_r0=50.0 * unit.nanometer,coordination_dmax=-400.0 * unit.nanometer,
                highcoord_threshold=12.0,
            )


class TestCollectiveVariables(object):
    @pytest.fixture(autouse=True)
    def change_test_dir(self, request, monkeypatch):
        # Change the working directory to the directory of the test file.
        # See https://stackoverflow.com/questions/62044541/change-pytest-working-directory-to-test-case-directory
        monkeypatch.chdir(request.fspath.dirname)

    @pytest.fixture(autouse=True)
    def tear_down(self):
        yield
        assert os.path.isfile("final_frame.gsd")
        assert os.path.isfile("state_data.csv")
        assert os.path.isfile("trajectory.gsd")
        assert os.path.isfile("colvars_xP.csv")

        os.remove("final_frame.gsd")
        os.remove("state_data.csv")
        os.remove("trajectory.gsd")
        os.remove("colvars_xP.csv")

    @staticmethod
    def compute_expected_highcoord_composition(positions, particle_type_names, target_particle_type, box_lengths,
                                                coordination_d0, coordination_r0, coordination_dmax, highcoord_threshold,
                                                nn, ignore_types) -> float:
        _EPSILON = 1e-12
        positions = np.asarray(positions, dtype=float)
        particle_type_names = np.asarray(particle_type_names)

        include_particles = np.ones(len(particle_type_names), dtype=float)
        if ignore_types is not None:
            include_particles = ~np.isin(particle_type_names, ignore_types)
            include_particles = include_particles.astype(float)

        is_target = (particle_type_names == target_particle_type).astype(float)

        displacements = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

        if box_lengths is not None:
            box_lengths = np.asarray(box_lengths, dtype=float)
            displacements -= box_lengths * np.rint(displacements / box_lengths)

        distances = np.linalg.norm(displacements, axis=-1)
        np.fill_diagonal(distances, np.inf)

        switch = (
            (distances <= coordination_dmax).astype(float)
            / (1.0 + np.exp((distances - coordination_d0) / coordination_r0))
        )

        coord = (switch * include_particles[np.newaxis, :]).sum(axis=1)

        highcoord = (
            coord**nn
            / (highcoord_threshold**nn + coord**nn + _EPSILON)
        )

        target_highcoord = (include_particles * is_target * highcoord).sum()
        all_highcoord = (include_particles * highcoord).sum()

        return target_highcoord / (all_highcoord + _EPSILON)

    @pytest.mark.filterwarnings(
    "ignore:The initial velocities in the GSD file are ignored because a velocity seed is provided.*:UserWarning")
    @pytest.mark.parametrize("yaml_file", ["run_highcoordcomp_umbrella.yaml"])
    def test_highcoord_comp_cv(self, yaml_file):
        colloids_run([yaml_file])

        f= np.loadtxt('colvars_xP.csv', delimiter=",", dtype=float, skiprows=1)
        reported_steps = f[:, 0].astype(int)
        actual_cv_values = f[:, 1]

        expected_cv_values = []

        with gsd.open("trajectory.gsd", "r") as trajectory:
            frames_by_step = {
                int(frame.configuration.step): frame
                for frame in trajectory}

            for step in reported_steps:
                frame = frames_by_step[step]
                particle_type_names = np.array(frame.particles.types)[frame.particles.typeid]

                expected = self.compute_expected_highcoord_composition(
                    positions=frame.particles.position,
                    particle_type_names=particle_type_names,
                    target_particle_type="2",
                    box_lengths=np.array(frame.configuration.box[:3]),
                    coordination_d0=300.0,
                    coordination_r0=50.0,
                    coordination_dmax=400.0,
                    highcoord_threshold=12.0,
                    nn=12,
                    ignore_types=None,
                )

                expected_cv_values.append(expected)

        assert actual_cv_values == pytest.approx(expected_cv_values,rel=1.0e-7,abs=1.0e-7,)

if __name__ == '__main__':
    pytest.main([__file__])

