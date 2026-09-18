import csv
from datetime import datetime, timedelta

import numpy as np


def test_import() -> None:
    from pytides2 import astro, constituent, nodal_corrections, tide


def test_extended_constituents() -> None:
    from pytides2 import constituent
    from pytides2.astro import astro

    extended = constituent.extended
    # The extended set reproduces the bundled 67-constituent table.
    assert len(extended) == 67
    assert len({c.name for c in extended}) == 67

    # pytides spells out Greek letters in lower case where the table uses an
    # all-upper-case abbreviation.
    alias = {
        "ALP1": "alpha1",
        "SIG1": "sigma1",
        "BET1": "beta1",
        "THE1": "theta1",
        "UPS1": "upsilon1",
        "EPS2": "epsilon2",
        "LDA2": "lambda2",
        "MSM": "MSm",
    }
    by_name = {c.name.upper(): c for c in extended}
    a = astro(datetime(2010, 6, 15, 7))
    with open("test/data/Tidal_constituents_67.csv", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        next(reader)
        rows = [(name.strip(), float(freq)) for name, freq in reader]

    assert len(rows) == 67
    for name, frequency in rows:
        c = by_name[alias.get(name, name).upper()]
        # speed() is in degrees/hour; the table lists cycles/hour.
        assert abs(c.speed(a) / 360.0 - frequency) < 1e-6, name


def test_Tide_noaa() -> None:
    from pytides2 import constituent
    from pytides2.tide import Tide

    by_name = {c.name: c for c in constituent.noaa}
    amplitudes = {"M2": 1.5, "S2": 0.5, "K1": 0.4, "O1": 0.3, "N2": 0.2}
    phases = {"M2": 45.0, "S2": 90.0, "K1": 120.0, "O1": 200.0, "N2": 300.0}

    synthetic = Tide(
        constituents=[by_name[n] for n in amplitudes],
        amplitudes=[amplitudes[n] for n in amplitudes],
        phases=[phases[n] for n in amplitudes],
    )
    t0 = datetime(2015, 1, 1)
    times = [t0 + timedelta(hours=h) for h in range(24 * 120)]
    heights = synthetic.at(times)

    # Fitting with the NOAA set should recover the input constituents.
    tide, _ = Tide.decompose(heights, np.array(times), constituents=constituent.noaa)
    fitted = {m["constituent"].name: m for m in tide.model}
    for name in amplitudes:
        assert abs(fitted[name]["amplitude"] - amplitudes[name]) < 1e-3
        assert abs(fitted[name]["phase"] - phases[name]) < 0.5


def test_Tide_extended() -> None:
    from pytides2 import constituent
    from pytides2.tide import Tide

    by_name = {c.name: c for c in constituent.extended}
    amplitudes = {"M2": 1.5, "S2": 0.5, "K1": 0.4, "O1": 0.3, "N2": 0.2}
    phases = {"M2": 45.0, "S2": 90.0, "K1": 120.0, "O1": 200.0, "N2": 300.0}

    synthetic = Tide(
        constituents=[by_name[n] for n in amplitudes],
        amplitudes=[amplitudes[n] for n in amplitudes],
        phases=[phases[n] for n in amplitudes],
    )
    t0 = datetime(2015, 1, 1)
    times = [t0 + timedelta(hours=h) for h in range(24 * 120)]
    heights = synthetic.at(times)

    # Fitting with the full extended set should recover the input constituents.
    tide, _ = Tide.decompose(
        heights, np.array(times), constituents=constituent.extended
    )
    fitted = {m["constituent"].name: m for m in tide.model}
    for name in amplitudes:
        assert abs(fitted[name]["amplitude"] - amplitudes[name]) < 1e-3
        assert abs(fitted[name]["phase"] - phases[name]) < 0.5


def test_Tide() -> None:
    from pytides2.tide import Tide

    data = np.genfromtxt(
        "test/data/water_level.csv",
        dtype=np.dtype([("datetime", datetime), ("water_level", float)]),
        delimiter=",",
    )
    water_datetimes = []
    water_levels = []
    for d in data:
        water_datetimes.append(
            datetime.strptime(d["datetime"].decode(), "%d/%m/%Y %H:%M:%S")
        )
        water_levels.append(d["water_level"])

    t, least_square = Tide.decompose(water_levels, water_datetimes)

    assert least_square is None
    assert t.model[0]["amplitude"] == 178.94736842105263
    assert t.model[0][2] == 0.0
    assert len(t.model) == 33
    # The exact value drifts slightly with the numpy/scipy/BLAS version (the
    # least-squares solver does not fully converge on this short record), so
    # compare against the known results with a tolerance rather than exactly.
    assert any(
        abs(t.formzahl - v) < 1e-3
        for v in (0.3691950696609845, 0.38101230038937306, 0.38376753628038296)
    )
    assert t.type == "mixed (semidiurnal)"
