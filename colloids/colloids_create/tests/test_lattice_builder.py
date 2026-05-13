import os
import subprocess
import gsd.hoomd
import numpy as np
import pytest


class TestClusterGenerator(object):
    @pytest.fixture(autouse=True)
    def change_test_dir(self, request, monkeypatch):
        # Change the working directory to the directory of the test file.
        # See https://stackoverflow.com/questions/62044541/change-pytest-working-directory-to-test-case-directory
        monkeypatch.chdir(request.fspath.dirname)

    @pytest.fixture
    def initial_configuration_filename(self):
        return "first_frame_lattice_builder.gsd"

    def test_cubic_lattice_with_satellites_generator(self, initial_configuration_filename):
        # Comparison to reference configurations in gsd format created by a tested version.
        configuration_parameters_file = "configuration_lattice_builder_test.yaml"
        reference_configuration_filename = "reference_configuration_lattice_builder.gsd"

        subprocess.run(f"pacsim-create {configuration_parameters_file} {initial_configuration_filename}",
                       shell=True, check=True)
        assert os.path.isfile(initial_configuration_filename)
        with (gsd.hoomd.open(initial_configuration_filename, "r") as f_new,
              gsd.hoomd.open(reference_configuration_filename, "r") as f_ref):
            assert len(f_new) == 1
            assert len(f_ref) == 1
            frame_new = f_new[0]
            frame_ref = f_ref[0]
            assert frame_new.particles.N == frame_ref.particles.N
            assert len(frame_new.particles.types) == len(frame_ref.particles.types)
            assert all(frame_new.particles.types[nt] == frame_ref.particles.types[ot]
                       for nt, ot in zip(frame_new.particles.typeid, frame_ref.particles.typeid))
            assert np.allclose(frame_new.configuration.box, frame_ref.configuration.box)
            assert np.allclose(frame_new.particles.position, frame_ref.particles.position)

            assert frame_new.constraints.N == frame_ref.constraints.N == 0

            assert np.allclose(frame_new.particles.mass, frame_ref.particles.mass)
            assert np.allclose(frame_new.particles.charge, frame_ref.particles.charge)
            assert np.allclose(frame_new.particles.diameter, frame_ref.particles.diameter)
        os.remove(initial_configuration_filename)


if __name__ == '__main__':
    pytest.main([__file__])
