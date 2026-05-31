from openmm import Context, Platform, System, VerletIntegrator, unit, Vec3
import pytest
from colloids import MagneticField


class TestMagneticFieldParameters(object):
    @pytest.fixture
    def amplitude_x(self):
        return 1.0 * (unit.kilojoule_per_mole / unit.nanometer)

    @pytest.fixture
    def amplitude_y(self):
        return 2.0 * (unit.kilojoule_per_mole / unit.nanometer)

    @pytest.fixture
    def frequency_x(self):
        return 1.0 / unit.picosecond

    @pytest.fixture
    def frequency_y(self):
        return 1.0 / unit.picosecond

    @pytest.fixture
    def field_dc(self, amplitude_x, amplitude_y):
        return MagneticField("DC", amplitude_x, amplitude_y, typeids=[1])

    @pytest.fixture
    def field_ac(self, amplitude_x, amplitude_y, frequency_x, frequency_y):
        return MagneticField("AC", amplitude_x, amplitude_y, typeids=[0], frequency_x=frequency_x,
                             frequency_y=frequency_y, phase_x=0.0, phase_y=0.0)

    @pytest.fixture
    def field_dc_pbc(self, amplitude_x, amplitude_y):
        box_length = 10.0 * unit.nanometer
        return MagneticField("DC", amplitude_x, amplitude_y, typeids=[0], use_pbc=True,
                             box_lengths=(box_length, box_length))


class TestMagneticFieldExceptions(TestMagneticFieldParameters):
    def test_typeids_must_be_integer(self, amplitude_x, amplitude_y):
        with pytest.raises(TypeError):
            MagneticField("DC", amplitude_x, amplitude_y, typeids=[0.5])

    def test_dc_must_not_accept_time_parameters(self, amplitude_x, amplitude_y, frequency_x, frequency_y):
        with pytest.raises(ValueError):
            MagneticField("DC", amplitude_x, amplitude_y, typeids=[0], frequency_x=frequency_x,
                          frequency_y=frequency_y)


class TestMagneticField(TestMagneticFieldParameters):
    @pytest.fixture
    def system(self):
        system = System()
        system.setDefaultPeriodicBoxVectors(Vec3(10.0, 0.0, 0.0), Vec3(0.0, 10.0, 0.0), Vec3(0.0, 0.0, 10.0))
        system.addParticle(0.0)
        system.addParticle(0.0)
        return system

    @pytest.fixture
    def single_particle_system(self):
        system = System()
        system.setDefaultPeriodicBoxVectors(Vec3(10.0, 0.0, 0.0), Vec3(0.0, 10.0, 0.0), Vec3(0.0, 0.0, 10.0))
        system.addParticle(0.0)
        return system

    @pytest.fixture
    def platform(self):
        return Platform.getPlatformByName("Reference")

    @pytest.fixture
    def integrator(self):
        return VerletIntegrator(0.25 * unit.picosecond)

    def test_dc_field(self, system, field_dc, platform):
        field_dc.add_particle(index=0, typeid=0)
        field_dc.add_particle(index=1, typeid=1)
        for potential in field_dc.yield_potentials():
            system.addForce(potential)

        context = Context(system, VerletIntegrator(0.001 * unit.picosecond), platform)
        context.setPositions([Vec3(1.0, 2.0, 0.0), Vec3(3.0, 4.0, 0.0)])

        energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        assert energy == pytest.approx(1.0 * 3.0 + 2.0 * 4.0)

    def test_ac_field(self, system, field_ac, platform):
        field_ac.add_particle(index=0, typeid=0)
        field_ac.add_particle(index=1, typeid=1)
        for potential in field_ac.yield_potentials():
            system.addForce(potential)

        context = Context(system, VerletIntegrator(0.001 * unit.picosecond), platform)
        context.setPositions([Vec3(1.0, 2.0, 0.0), Vec3(3.0, 4.0, 0.0)])

        initial_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        assert initial_energy == pytest.approx(1.0 + 2.0 * 2.0)

        context.setParameter("magnetic_field_time", 0.25)
        stepped_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        assert stepped_energy == pytest.approx(0.0, abs=1.0e-12)

    def test_dc_field_with_pbc_wraps_coordinates(self, single_particle_system, field_dc_pbc, platform):
        field_dc_pbc.add_particle(index=0, typeid=0)
        for potential in field_dc_pbc.yield_potentials():
            single_particle_system.addForce(potential)

        context = Context(single_particle_system, VerletIntegrator(0.001 * unit.picosecond), platform)

        context.setPositions([Vec3(6.0, 0.0, 0.0)])
        wrapped_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        context.setPositions([Vec3(-4.0, 0.0, 0.0)])
        reference_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        assert wrapped_energy == pytest.approx(reference_energy)