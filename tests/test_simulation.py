from pantheon import Simulation


def test_determinism():
    assert Simulation(20, seed=7).run(3) == Simulation(20, seed=7).run(3)


def test_decisions():
    r = Simulation(100, seed=4).run(5)
    assert (r.agents, r.ticks, r.decisions) == (100, 5, 500)


def test_reject_invalid_input():
    try:
        Simulation(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative population must fail")
