from openmm import unit
import openmm
import pytest

import colloids.integrators as integrators
from colloids.colloids_run import initialize_barostat, initialize_integrators
from colloids.run_parameters import RunParameters


class TestIntegrators(object):
    def test_monte_carlo_barostat_returns_barostat(self):
        barostat = integrators.MonteCarloBarostat(
            temperature=300.0 * unit.kelvin,
            pressure=1.5 * unit.bar,
            frequency=17,
        )

        assert isinstance(barostat, openmm.MonteCarloBarostat)
        assert barostat.getDefaultPressure().value_in_unit(unit.bar) == pytest.approx(1.5)
        assert barostat.getDefaultTemperature().value_in_unit(unit.kelvin) == pytest.approx(300.0)
        assert barostat.getFrequency() == 17

    def test_monte_carlo_anisotropic_barostat_returns_barostat(self):
        barostat = integrators.MonteCarloAnisotropicBarostat(
            temperature=301.0 * unit.kelvin,
            pressure_x=1.0 * unit.bar,
            pressure_y=2.0 * unit.bar,
            pressure_z=3.0 * unit.bar,
            scale_x=False,
            scale_y=True,
            scale_z=False,
            frequency=19,
        )

        assert isinstance(barostat, openmm.MonteCarloAnisotropicBarostat)
        default_pressure = barostat.getDefaultPressure()
        assert default_pressure.x == pytest.approx(1.0)
        assert default_pressure.y == pytest.approx(2.0)
        assert default_pressure.z == pytest.approx(3.0)
        assert barostat.getDefaultTemperature().value_in_unit(unit.kelvin) == pytest.approx(301.0)
        assert barostat.getFrequency() == 19

    def test_initialize_integrators_builds_all_configured_objects(self, monkeypatch):
        calls = []

        def make_fake_integrator(name):
            def fake_integrator(**kwargs):
                calls.append((name, kwargs))
                return name

            return fake_integrator

        monkeypatch.setitem(integrators.INTEGRATORS, "AlphaIntegrator", make_fake_integrator("AlphaIntegrator"))
        monkeypatch.setitem(integrators.INTEGRATORS, "BetaIntegrator", make_fake_integrator("BetaIntegrator"))

        parameters = RunParameters(
            initial_configuration="first_frame.gsd",
            integrators={
                "AlphaIntegrator": {"value": 1},
                "BetaIntegrator": {"value": 2},
            },
        )

        calls.clear()
        integrator_objects = initialize_integrators(parameters)

        assert list(integrator_objects) == ["AlphaIntegrator", "BetaIntegrator"]
        assert integrator_objects == {"AlphaIntegrator": "AlphaIntegrator", "BetaIntegrator": "BetaIntegrator"}
        assert calls == [("AlphaIntegrator", {"value": 1}), ("BetaIntegrator", {"value": 2})]

    def test_initialize_barostat_builds_optional_barostat(self, monkeypatch):
        calls = []

        def fake_barostat(**kwargs):
            calls.append(kwargs)
            return "barostat"

        monkeypatch.setitem(integrators.INTEGRATORS, "MonteCarloBarostat", fake_barostat)

        parameters = RunParameters(
            initial_configuration="first_frame.gsd",
            barostat="MonteCarloBarostat",
            barostat_parameters={"temperature": 300.0 * unit.kelvin, "pressure": 1.0 * unit.bar,
                                 "frequency": 25},
        )

        calls.clear()
        barostat = initialize_barostat(parameters)

        assert barostat == "barostat"
        assert calls == [{"temperature": 300.0 * unit.kelvin, "pressure": 1.0 * unit.bar, "frequency": 25}]


if __name__ == '__main__':
    pytest.main([__file__])
