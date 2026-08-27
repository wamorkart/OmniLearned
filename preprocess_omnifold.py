#!/usr/bin/env python
"""Preprocess the energyflow Z+jets Pythia/Herwig Delphes dataset for OmniFold
unfolding. Direct port of ViniciusMikuni/OmniLearn's
preprocessing/preprocess_omnifold.py (same feature engineering, same output
schema: 13 particle features, 4 jet features, 6 substructure observables) so
that its output is a drop-in match for the train_pythia.h5/test_pythia.h5
already produced under /pscratch/sd/t/twamorka/unfolding/ by that reference
script -- running this for --sample herwig fills in the missing
train_herwig.h5/test_herwig.h5 counterpart without touching the existing
pythia files.

Run with the `energyflow` conda env (has energyflow/h5py/numpy; the shared
omnilearned-clean training env intentionally does NOT have energyflow --
this is a one-off data-prep step, kept isolated from live training/eval jobs):

    conda activate energyflow
    python preprocess_omnifold.py --folder /pscratch/sd/t/twamorka/unfolding --sample herwig
"""

import h5py
import os
import numpy as np
from optparse import OptionParser
import energyflow as ef


def convert_pid(pid, mask):
    pid_dict = {
        0.0: 22,
        0.1: 211,
        0.2: -211,
        0.3: 130,
        0.4: 11,
        0.5: -11,
        0.6: 13,
        0.7: -13,
        0.8: 321,
        0.9: -321,
        1.0: 2212,
        1.1: -2212,
        1.2: 2112,
        1.3: -2112,
    }
    for value in pid_dict:
        pid[pid == value] = pid_dict[value]

    return pid * mask


def get_substructure_obs(dataset):
    feature_names = ["widths", "mults", "sdms", "zgs", "tau2s"]
    gen_features = [dataset["gen_jets"][:, 3]]
    sim_features = [dataset["sim_jets"][:, 3]]
    for feature in feature_names:
        gen_features.append(dataset["gen_" + feature])
        sim_features.append(dataset["sim_" + feature])

    gen_features = np.stack(gen_features, -1)
    sim_features = np.stack(sim_features, -1)
    # ln rho
    gen_features[:, 3] = 2 * np.ma.log(
        np.ma.divide(gen_features[:, 3], dataset["gen_jets"][:, 0] + 10**-100).filled(0)
    ).filled(0)
    sim_features[:, 3] = 2 * np.ma.log(
        np.ma.divide(sim_features[:, 3], dataset["sim_jets"][:, 0] + 10**-100).filled(0)
    ).filled(0)
    # tau21
    gen_features[:, 5] = gen_features[:, 5] / (10**-50 + gen_features[:, 1])
    sim_features[:, 5] = sim_features[:, 5] / (10**-50 + sim_features[:, 1])

    return sim_features, gen_features


# Preprocessing for the Z+jets Delphes dataset (same feature engineering as
# OmniLearn's top-tagging/JetClass preprocessing -- see preprocess_omnifold.py
# in the reference repo).
def preprocess(p, jets, nparts=101):
    particles = p[:, :, :-1]
    mask = particles[:, :, 0] > 0
    particles[:, :, 0] = 100.0 * particles[:, :, 0]
    particles[:, :, 1] = particles[:, :, 1] + jets[:, None, 1]
    particles[:, :, 2] = particles[:, :, 2] + jets[:, None, 2]
    pid = p[:, :, -1]
    pid = convert_pid(pid, mask)
    p4 = ef.p4s_from_ptyphipids(np.concatenate([particles, pid[:, :, None]], -1))

    jets = np.sum(p4, axis=1)

    jets_e = jets[:, 0]
    eta = ef.etas_from_p4s(jets)
    jets = ef.ptyphims_from_p4s(jets)
    jets[:, 1] = eta

    p_e = particles[:, :, 0] * np.cosh(particles[:, :, 1]) * mask
    particles_full = np.concatenate([particles, p_e[:, :, None]], -1)
    # The reference script hardcodes nparts=101 assuming a full-scale load
    # (num_data=-1) always reaches that many particles in its busiest event --
    # true for the full ~1.6M-event Pythia/Herwig samples (confirmed by the
    # existing train_pythia.h5 on disk, shape (1600000, 101, 13)), but a
    # smaller/subset load can have fewer. Use whatever's actually available
    # and zero-pad up to nparts, so output is always shape-consistent
    # regardless of how many events were loaded -- a strict superset of the
    # reference behavior (identical output at full scale, where n_use==nparts
    # makes the padding branch below a no-op).
    n_use = min(particles_full.shape[1], nparts)
    particles = particles_full[:, :n_use]
    pid = pid[:, :n_use]

    NFEAT = 13
    points_use = np.zeros((particles.shape[0], n_use, NFEAT))

    delta_phi = particles[:, :, 2] - jets[:, None, 2]
    delta_phi[delta_phi > np.pi] -= 2 * np.pi
    delta_phi[delta_phi <= -np.pi] += 2 * np.pi

    points_use[:, :, 0] = particles[:, :, 1] - jets[:, None, 1]
    points_use[:, :, 1] = delta_phi
    points_use[:, :, 2] = np.ma.log(1.0 - particles[:, :, 0] / jets[:, None, 0]).filled(0)
    points_use[:, :, 3] = np.ma.log(particles[:, :, 0]).filled(0)
    points_use[:, :, 4] = np.ma.log(1.0 - particles[:, :, 3] / jets_e[:, None]).filled(0)
    points_use[:, :, 5] = np.ma.log(particles[:, :, 3]).filled(0)
    points_use[:, :, 6] = np.hypot(points_use[:, :, 0], points_use[:, :, 1])
    points_use[:, :, 7] = (np.sign(pid)) * (pid != 22) * (pid != 130)
    points_use[:, :, 8] = (np.abs(pid) == 211) | (np.abs(pid) == 321) | (np.abs(pid) == 2212)
    points_use[:, :, 9] = (np.abs(pid) == 130) | (np.abs(pid) == 2112) | (pid == 0)
    points_use[:, :, 10] = np.abs(pid) == 22
    points_use[:, :, 11] = np.abs(pid) == 11
    points_use[:, :, 12] = np.abs(pid) == 13

    mult = np.sum(mask, -1)
    points_use *= mask[:, :n_use, None]

    if n_use < nparts:
        points = np.zeros((particles.shape[0], nparts, NFEAT))
        points[:, :n_use] = points_use
    else:
        points = points_use

    # delete phi
    jets = np.delete(jets, 2, axis=1)
    jets = np.concatenate([jets, mult[:, None]], -1)

    return points, jets


def write_split(fname, reco_parts, gen_parts, reco_jets, gen_jets, reco_subs, gen_subs, sl):
    with h5py.File(fname, "w") as fh5:
        fh5.create_dataset("reco", data=reco_parts[sl])
        fh5.create_dataset("gen", data=gen_parts[sl])
        fh5.create_dataset("reco_jets", data=reco_jets[sl])
        fh5.create_dataset("gen_jets", data=gen_jets[sl])
        fh5.create_dataset("reco_subs", data=reco_subs[sl])
        fh5.create_dataset("gen_subs", data=gen_subs[sl])
    print(f"Wrote {fname}: {reco_parts[sl].shape[0]} events")


if __name__ == "__main__":
    parser = OptionParser(usage="%prog [opt]")
    parser.add_option("--npoints", type=int, default=101, help="Number of particles per event")
    parser.add_option(
        "--folder",
        type="string",
        default="/pscratch/sd/t/twamorka/unfolding",
        help="Folder containing/receiving the dataset (raw energyflow cache goes in "
        "<folder>/datasets/, output h5 files go directly in <folder>/)",
    )
    parser.add_option(
        "--sample", type="string", default="herwig", help="sample to use: pythia or herwig"
    )
    parser.add_option(
        "--num-data", type=int, default=-1, help="Number of events to load (-1 = all cached)"
    )

    (flags, args) = parser.parse_args()

    # energyflow's own load() appends "datasets/ZjetsDelphes" under whatever
    # cache_dir it's given -- pass flags.folder directly (matching the
    # reference script) so this lands at <folder>/datasets/ZjetsDelphes/,
    # the exact path the existing Pythia21 cache already lives at.
    samples_path = flags.folder
    NPARTS = flags.npoints

    generator = {"pythia": "Pythia21", "herwig": "Herwig"}[flags.sample]
    dataset = ef.zjets_delphes.load(
        generator,
        num_data=flags.num_data,
        pad=True,
        cache_dir=samples_path,
        source="zenodo",
        which="all",
        include_keys=None,
        exclude_keys=None,
    )

    reco_subs, gen_subs = get_substructure_obs(dataset)
    reco_parts, reco_jets = preprocess(
        dataset["sim_particles"], dataset["sim_jets"], NPARTS
    )
    gen_parts, gen_jets = preprocess(dataset["gen_particles"], dataset["gen_jets"], NPARTS)

    n = reco_parts.shape[0]
    n_train = min(1600000, n)
    train_sl = slice(0, n_train)
    # Reference uses a fixed 1,200,000 test-start offset regardless of n_train,
    # matching the exact train_pythia.h5/test_pythia.h5 split already on disk.
    test_sl = slice(1200000, n)

    write_split(
        os.path.join(flags.folder, f"train_{flags.sample}.h5"),
        reco_parts, gen_parts, reco_jets, gen_jets, reco_subs, gen_subs, train_sl,
    )
    write_split(
        os.path.join(flags.folder, f"test_{flags.sample}.h5"),
        reco_parts, gen_parts, reco_jets, gen_jets, reco_subs, gen_subs, test_sl,
    )
