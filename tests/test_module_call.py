import os
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pyneapples.rep_sampler_2d import GeomReduction

@pytest.fixture
def geom_reduction_instance():
    np.random.seed(42)
    gr = GeomReduction(500, 3, 20, 100, 1, 1, weighted=False, pdfcomp="KLdiv", intweights=0, verbose=False)
    excit_e = 2 + (np.random.rand(500, 3) - 0.5) * 4
    excit_e = np.clip(excit_e, 0, 4)
    trans_dip_mom_x = 10 + np.random.rand(500, 3) * 90
    trans_dip_mom_y = 10 + np.random.rand(500, 3) * 90
    trans_dip_mom_z = 10 + np.random.rand(500, 3) * 90
    gr._trans_dip_mom_x = trans_dip_mom_x
    gr._trans_dip_mom_y = trans_dip_mom_y
    gr._trans_dip_mom_z = trans_dip_mom_z
    gr.read_data_direct(excit_e, trans_dip_mom_x, trans_dip_mom_y, trans_dip_mom_z)
    return gr

def test_reduction_count(geom_reduction_instance):
    gr = geom_reduction_instance
    gr.reduce_geoms()
    reduced = gr.subsamples
    expected_count = 20
    assert len(reduced) == expected_count, f"Expected {expected_count} reduced geometries, got {len(reduced)} "

def test_transition_dipole_components_range(geom_reduction_instance):
    gr = geom_reduction_instance
    for comp, label in zip([gr._trans_dip_mom_x, gr._trans_dip_mom_y, gr._trans_dip_mom_z], ['x', 'y', 'z']):
        min_val = comp.min()
        max_val = comp.max()
        assert min_val >= 10, f"Transition dipole moment component '{label}' has a minimum value {min_val} which is less than 10."
        assert max_val <= 100, f"Transition dipole moment component '{label}' has a maximum value {max_val} which is greater than 100."

def test_read_data_direct_calculations():
    ns = 2
    nstates = 2
    gr = GeomReduction(ns, nstates, subset=1, cycles=1, ncores=1, njobs=1, weighted=False, pdfcomp="KLdiv", intweights=0, verbose=False)
    excitation = np.array([[1.0, 2.0], [3.0, 4.0]])
    dip_x = np.array([[1.0, 0.0], [0.0, 1.0]])
    dip_y = np.array([[0.0, 1.0], [1.0, 0.0]])
    dip_z = np.array([[1.0, 1.0], [1.0, 1.0]])
    gr.read_data_direct(excitation, dip_x, dip_y, dip_z)
    expected_trans = np.array([[2.0, 2.0], [2.0, 2.0]])
    np.testing.assert_allclose(gr.exc, excitation, err_msg="Excitation energies not set correctly.")
    np.testing.assert_allclose(gr.trans, expected_trans, err_msg="Transition dipole moment processing failed.")
    expected_weights = excitation * expected_trans
    np.testing.assert_allclose(gr.weights, expected_weights, err_msg="Computed weights do not match expected values.")
    expected_wnorms = np.sum(expected_weights, axis=0) / np.sum(expected_weights)
    np.testing.assert_allclose(gr.wnorms, expected_wnorms, err_msg="Normalized weights are incorrect.")

def test_reduce_geoms(monkeypatch):
    ns = 20
    nstates = 1
    subset = 5
    cycles = 1
    gr = GeomReduction(ns, nstates, subset, cycles, ncores=1, njobs=1, weighted=False, pdfcomp="KLdiv", intweights=0, verbose=False)
    np.random.seed(0)
    excitation = np.random.rand(ns, nstates) * 4
    dip_x = np.random.rand(ns, nstates) * 90 + 10
    dip_y = np.random.rand(ns, nstates) * 90 + 10
    dip_z = np.random.rand(ns, nstates) * 90 + 10
    gr.read_data_direct(excitation, dip_x, dip_y, dip_z)
    monkeypatch.setattr(os, "mkdir", lambda x: None)
    monkeypatch.setattr(os, "chdir", lambda x: None)
    monkeypatch.setattr(np, "savetxt", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)
    gr.reduce_geoms()
    assert len(gr.subsamples) == subset, f"Expected {subset} reduced geometries, got {len(gr.subsamples)}."

def test_read_data_direct_osc():
    gr = GeomReduction(2, 1, subset=1, cycles=1, ncores=1, njobs=1,
                       weighted=False, pdfcomp="KLdiv", intweights=0, verbose=False)
    excitation = np.array([2.0, 4.0])
    osc_strengths = np.array([0.1, 0.2])
    gr.read_data_direct_osc(excitation, osc_strengths)
    expected_trans = abs((3 * osc_strengths) / (2 * (excitation / 27.211396)))
    expected_weights = excitation * expected_trans
    expected_wnorms = np.sum(expected_weights, axis=0) / np.sum(expected_weights)
    assert gr.infile == "Test_Filename"
    np.testing.assert_allclose(gr.exc, excitation, err_msg="Excitation energies do not match")
    np.testing.assert_allclose(gr.trans, expected_trans, err_msg="Transition dipole values do not match")
    np.testing.assert_allclose(gr.weights, expected_weights, err_msg="Weights do not match")
    np.testing.assert_allclose(gr.wnorms, expected_wnorms, err_msg="Normalized weights do not match")
    from datetime import datetime
    assert isinstance(gr.time, datetime), "gr.time is not a datetime instance"
