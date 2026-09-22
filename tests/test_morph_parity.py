"""The Python pose and the Electron pose are one function."""
import json
import os
import subprocess

from nca.instinct import InstinctState
from nca.morph import pose_from_state


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _node(cases):
    proc = subprocess.run(
        ["node", "electron/renderer/morph.js"],
        cwd=ROOT,
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def test_pose_and_drives_match_javascript():
    poses = [
        {"op": "pose", "rid": [0, 0, 0, 0, 0, 0], "utterance": "", "user_text": "", "genome_fold": 8, "damage_pulse": 0},
        {
            "op": "pose",
            "rid": [0.5, -0.4, 0.8, 0.2, 0.7, -0.1],
            "utterance": "Something tore me.",
            "user_text": "bloom now",
            "genome_fold": 12,
            "damage_pulse": 0.4,
        },
        {
            "op": "pose",
            "rid": [-1, 1, -1, 1, 1, 1],
            "utterance": "rest sleep quiet",
            "user_text": "sing the tide",
            "genome_fold": 3,
            "damage_pulse": 2,
        },
        {
            "op": "pose",
            "rid": [1, 1, 1, 1, 1, 1],
            "utterance": "More! Bloom now, ecstatic!",
            "user_text": "pulse beat drum",
            "genome_fold": 20,
            "damage_pulse": 0,
        },
    ]
    drives = [
        {
            "op": "drives",
            "rid": [0, 0, 0, 0, 0, 0],
            "desc": {"mass": 0.3, "rgb": [0.5, 0.5, 0.5], "spread": 0.5, "pan": 0.5, "symmetry": 0.9},
            "growth": -0.05,
            "audio_energy": 0.2,
            "beat": 0.1,
            "rel_drop": 0.4,
            "momentum": 0.0,
        },
        {
            "op": "drives",
            "rid": [0.2, -0.3, 0.5, 0.1, -0.2, 0.4],
            "desc": {"mass": 0.1, "rgb": [0.2, 0.3, 0.1], "spread": 0.2, "pan": 0.5, "symmetry": 0.4},
            "growth": 0.0,
            "audio_energy": 0.8,
            "beat": 0.5,
            "rel_drop": 0.0,
            "momentum": 0.6,
        },
    ]
    expected = []
    for case in poses:
        expected.append(pose_from_state(
            case["rid"], case["utterance"], case["user_text"], case["genome_fold"], case["damage_pulse"],
        ))
    for case in drives:
        state = InstinctState(
            arousal=case["rid"][0], valence=case["rid"][1], pain=case["rid"][2],
            hunger=case["rid"][3], rhythm=case["rid"][4], calm=case["rid"][5],
            momentum=case["momentum"],
        )
        state.update(case["desc"], case["growth"], case["audio_energy"], case["beat"], case["rel_drop"])
        expected.append(state.rid)

    got = _node(poses + drives)
    assert len(got) == len(expected)
    for want, have in zip(expected, got):
        if isinstance(want, dict):
            assert set(want) == set(have)
            for key, value in want.items():
                if isinstance(value, float):
                    assert abs(value - have[key]) < 1e-9
                else:
                    assert value == have[key]
        else:
            assert len(want) == len(have) == 6
            for a, b in zip(want, have):
                assert abs(a - b) < 1e-9


def test_node_voice_selfcheck():
    proc = subprocess.run(
        ["node", "electron/renderer/voice.js"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "voice ok" in proc.stdout
