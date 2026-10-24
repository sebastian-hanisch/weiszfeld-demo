"""Szenario: der blockweise Strom ist der Strom der kopierten Klasse, Layouts und Gewichte haben ihre dokumentierten Eigenschaften."""

import math

import numpy as np
import pytest

import wsz_constants as C
from wsz_rng import SplitMix64
from wsz_scenario import Draws, big_customer_weight, layout_batch, make_instance, threshold_weight, weights_batch


@pytest.mark.parametrize("seed", [0, 1, 7, 2**63 + 5, 999999])
def test_block_stream_equals_the_copied_splitmix64(seed):
    rng = SplitMix64(seed)
    expected = [rng.next() for _ in range(50)]
    assert Draws(seed).raw(50).tolist() == expected


def test_blocks_continue_the_stream_where_the_last_block_ended():
    d = Draws(5)
    first, second = d.raw(7), d.raw(9)
    assert np.concatenate([first, second]).tolist() == Draws(5).raw(16).tolist()
    assert Draws(5).raw(0).size == 0


def test_uniform_is_the_top_53_bits_of_the_stream():
    rng = SplitMix64(3)
    expected = [(rng.next() >> 11) / 2**53 for _ in range(10)]
    assert Draws(3).uniform(10).tolist() == expected
    u = Draws(4).uniform((1000, 3))
    assert u.min() >= 0 and u.max() < 1 and abs(u.mean() - 0.5) < 0.03


def test_normal_by_hand_from_two_uniforms():
    """Box-Muller: z0 = √(−2 ln(1 − u1))·cos(2π u2), z1 = … sin(2π u2); bei Größe 2 steht z0 zuerst."""
    u = Draws(9).uniform(2)
    r = math.sqrt(-2 * math.log(1 - u[0]))
    z = Draws(9).normal(2)
    assert z.tolist() == pytest.approx([r * math.cos(2 * math.pi * u[1]), r * math.sin(2 * math.pi * u[1])])


def test_normal_has_mean_zero_and_variance_one():
    z = Draws(10).normal(200000)
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01
    assert Draws(10).normal(3).shape == (3,)           # ungerade Größe
    assert Draws(10).normal((4, 2, 2)).shape == (4, 2, 2)


def test_layouts():
    d = Draws(1)
    q = layout_batch(d, "Quadrat", 50, 20)
    assert q.shape == (50, 20, 2) and q.min() >= -1 and q.max() < 1
    o = layout_batch(Draws(1), "Kern + Außenposten", 5, 8)
    assert (o[:, 0] == [6.0, 6.0]).all()
    assert layout_batch(Draws(1), "Normal", 3, 4).shape == (3, 4, 2)
    with pytest.raises(ValueError):
        layout_batch(Draws(1), "Kreis", 1, 3)


def test_each_layout_is_different_from_the_others():
    pts = [layout_batch(Draws(2), name, 1, 10)[0] for name in C.LAYOUTS]
    assert not np.allclose(pts[0], pts[1]) and not np.allclose(pts[0], pts[2]) and not np.allclose(pts[1], pts[2])


def test_big_customer_weight_gives_the_requested_share():
    assert big_customer_weight(3, 0.5) == pytest.approx(2.0)                    # 2 / (2 + 1 + 1) = 50 %
    w = weights_batch(Draws(1), "Großkunde", 1, 20, 0.3)[0]
    assert w[0] / w.sum() == pytest.approx(0.3) and (w[1:] == 1).all()
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            big_customer_weight(5, bad)
    with pytest.raises(ValueError):
        big_customer_weight(1, 0.5)


def test_weight_kinds():
    assert (weights_batch(Draws(1), "gleich", 2, 5) == 1).all()
    w1 = weights_batch(Draws(1), "lognormal 1", 400, 50)
    w05 = weights_batch(Draws(1), "lognormal 0,5", 400, 50)
    assert (w1 > 0).all() and (w05 > 0).all()
    assert np.log(w1).std() == pytest.approx(1.0, abs=0.05) and np.log(w05).std() == pytest.approx(0.5, abs=0.03)
    with pytest.raises(ValueError):
        weights_batch(Draws(1), "unbekannt", 1, 3)


def test_threshold_weight_by_hand():
    """A = (0, 0), B = (4, 0), C = (0, 3): die Zugkraft von B und C auf A ist (1, 1), Norm √2."""
    p = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
    assert threshold_weight(p, np.ones(2)) == pytest.approx(math.sqrt(2))
    assert threshold_weight(p, np.array([2.0, 1.0])) == pytest.approx(math.sqrt(5))     # (2, 1)


def test_make_instance_is_deterministic_and_reacts_to_every_regler():
    base = make_instance("Normal", 10, "gleich", 40, 1.0, 7)
    again = make_instance("Normal", 10, "gleich", 40, 1.0, 7)
    assert np.array_equal(base.points, again.points) and np.array_equal(base.weights, again.weights) and base.n == 10
    assert not np.allclose(base.points, make_instance("Normal", 10, "gleich", 40, 1.0, 8).points)
    assert not np.allclose(base.points, make_instance("Quadrat", 10, "gleich", 40, 1.0, 7).points)
    assert not np.allclose(base.weights, make_instance("Normal", 10, "lognormal 1", 40, 1.0, 7).weights)
    big40 = make_instance("Normal", 10, "Großkunde", 40, 1.0, 7)
    big60 = make_instance("Normal", 10, "Großkunde", 60, 1.0, 7)
    assert big40.weights[0] / big40.weights.sum() == pytest.approx(0.4) and big60.weights[0] / big60.weights.sum() == pytest.approx(0.6)


def test_threshold_instance_has_pull_equal_to_f_times_weight():
    from wsz_geomed import pull_norms
    for f in (0.5, 1.0, 2.0):
        inst = make_instance("Normal", 6, "Großkunde an der Schwelle", 40, f, 3)
        norms, own = pull_norms(inst.points, inst.weights)
        assert norms[0] / own[0] == pytest.approx(1.0 / f)
