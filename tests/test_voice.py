"""Tests for the creature's voice (sonification)."""
import numpy as np
import torch

from nca.model import DiatomNCA
from nca.voice import describe_trajectory, frame_descriptor, save_wav, sing


def test_frame_descriptor_sane():
    m = DiatomNCA(num_channels=16, hidden_dim=16)
    x = m.seed(1, 24)
    d = frame_descriptor(x[0])
    assert 0 <= d["mass"] <= 1
    assert 0 <= d["pan"] <= 1
    assert 0 <= d["symmetry"] <= 1
    assert len(d["rgb"]) == 3


def test_song_is_stereo_music():
    # Seeded: an unlucky random rule can grow dead centre, and a centred body pans to both ears equally.
    torch.manual_seed(1)
    m = DiatomNCA(num_channels=16, hidden_dim=16)
    with torch.no_grad():
        traj = m.grow(40, size=24)
    song = sing(traj[0], frame_dur=0.05)
    assert song.ndim == 2 and song.shape[1] == 2  # stereo
    assert np.isfinite(song).all()
    assert np.abs(song).max() > 0.01, "song must be audible"
    assert np.abs(song).max() <= 1.0
    assert (song[:, 0] != song[:, 1]).any(), "pan must separate channels"


def test_save_wav_roundtrip(tmp_path):
    m = DiatomNCA(num_channels=16, hidden_dim=16)
    with torch.no_grad():
        traj = m.grow(40, size=24)
    song = sing(traj[0], frame_dur=0.05)
    p = str(tmp_path / "voice.wav")
    save_wav(p, song)
    from scipy.io import wavfile

    sr, data = wavfile.read(p)
    assert sr == 22050 and data.shape[1] == 2


def test_describe_trajectory_length():
    m = DiatomNCA(num_channels=16, hidden_dim=16)
    with torch.no_grad():
        traj = m.grow(10, size=16)
    assert len(describe_trajectory(traj[0])) == 11  # includes t=0
