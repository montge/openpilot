# Stone Soup Adapter Layer Design

**Date**: 2026-01-24
**Status**: Approved
**OpenSpec**: add-stonesoup-comparison, Section 1

## Overview

Adapter layer for bidirectional conversion between openpilot and Stone Soup types, enabling algorithm comparison without modifying production code.

## Decisions

- **Stone Soup version**: 1.8 (latest stable)
- **Timestamps**: Relative to arbitrary epoch (datetime(2000, 1, 1)), converting openpilot monotonic seconds to Python datetime
- **Radar state vector**: 3D `[dRel, yRel, vRel]`
- **Pose state vector**: 6D `[x, y, yaw, vx, vy, yaw_rate]`

## Directory Structure

```
tools/stonesoup/
├── __init__.py
├── adapters.py          # Type conversions
├── requirements.txt     # stonesoup>=1.8
└── tests/
    └── test_adapters.py
```

## Implementation

### adapters.py

```python
from datetime import datetime, timedelta
import numpy as np
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector


class OpenpilotAdapter:
    """Bidirectional conversion between openpilot and Stone Soup types."""

    def __init__(self, epoch: datetime | None = None):
        # Reference epoch for relative timestamp conversion
        self.epoch = epoch or datetime(2000, 1, 1)

    def mono_time_to_datetime(self, mono_time: float) -> datetime:
        """Convert openpilot monotonic time (seconds) to datetime."""
        return self.epoch + timedelta(seconds=mono_time)

    def radar_point_to_detection(
        self,
        d_rel: float,
        y_rel: float,
        v_rel: float,
        timestamp: float,
        track_id: int | None = None,
        measured: bool = True
    ) -> Detection:
        """Convert radar point to Stone Soup Detection.

        State vector: [dRel, yRel, vRel] (longitudinal dist, lateral dist, relative velocity)
        """
        return Detection(
            state_vector=StateVector([d_rel, y_rel, v_rel]),
            timestamp=self.mono_time_to_datetime(timestamp),
            metadata={
                'track_id': track_id,
                'measured': measured,
            }
        )

    def gaussian_state_to_lead_dict(
        self,
        state,  # GaussianState or similar
        v_ego: float = 0.0,
        model_prob: float = 0.0,
        track_id: int = -1,
    ) -> dict:
        """Convert Stone Soup state estimate to LeadData dict.

        Expects state vector: [dRel, yRel, vRel] or [dRel, yRel, vRel, aRel]
        """
        sv = state.state_vector
        d_rel = float(sv[0])
        y_rel = float(sv[1])
        v_rel = float(sv[2])
        a_rel = float(sv[3]) if len(sv) > 3 else 0.0

        v_lead = v_ego + v_rel

        return {
            'dRel': d_rel,
            'yRel': y_rel,
            'vRel': v_rel,
            'vLead': v_lead,
            'vLeadK': v_lead,      # Kalman-filtered velocity
            'aLeadK': a_rel,       # Kalman-filtered acceleration
            'aLeadTau': 1.5,       # Default decay constant
            'status': True,
            'fcw': False,
            'modelProb': model_prob,
            'radar': True,
            'radarTrackId': track_id,
        }

    def pose_to_gaussian_state(
        self,
        x: float, y: float, yaw: float,
        vx: float, vy: float, yaw_rate: float,
        timestamp: float,
        covariance: np.ndarray | None = None,
    ) -> GaussianState:
        """Convert openpilot pose to Stone Soup GaussianState.

        State vector: [x, y, yaw, vx, vy, yaw_rate]
        """
        state_vector = StateVector([x, y, yaw, vx, vy, yaw_rate])

        if covariance is None:
            # Default uncertainty
            covariance = np.diag([1.0, 1.0, 0.01, 0.5, 0.5, 0.01])

        return GaussianState(
            state_vector=state_vector,
            covar=covariance,
            timestamp=self.mono_time_to_datetime(timestamp),
        )

    def gaussian_state_to_pose_dict(self, state) -> dict:
        """Convert Stone Soup state back to pose dict."""
        sv = state.state_vector
        return {
            'x': float(sv[0]),
            'y': float(sv[1]),
            'yaw': float(sv[2]),
            'vx': float(sv[3]),
            'vy': float(sv[4]),
            'yaw_rate': float(sv[5]),
        }
```

### tests/test_adapters.py

```python
import pytest
import numpy as np
from datetime import datetime, timedelta

from openpilot.tools.stonesoup.adapters import OpenpilotAdapter


class TestOpenpilotAdapter:
    def test_mono_time_to_datetime(self):
        adapter = OpenpilotAdapter()
        dt = adapter.mono_time_to_datetime(3600.0)  # 1 hour
        expected = adapter.epoch + timedelta(hours=1)
        assert dt == expected

    def test_radar_point_to_detection(self):
        adapter = OpenpilotAdapter()
        det = adapter.radar_point_to_detection(
            d_rel=50.0, y_rel=-1.5, v_rel=-5.0,
            timestamp=100.0, track_id=42
        )
        assert det.state_vector[0] == 50.0   # dRel
        assert det.state_vector[1] == -1.5   # yRel
        assert det.state_vector[2] == -5.0   # vRel
        assert det.metadata['track_id'] == 42

    def test_gaussian_state_to_lead_dict(self):
        from stonesoup.types.state import GaussianState
        from stonesoup.types.array import StateVector

        adapter = OpenpilotAdapter()
        state = GaussianState(
            state_vector=StateVector([50.0, -1.5, -5.0]),
            covar=np.eye(3),
            timestamp=adapter.mono_time_to_datetime(100.0),
        )
        lead = adapter.gaussian_state_to_lead_dict(state, v_ego=20.0)

        assert lead['dRel'] == 50.0
        assert lead['vLead'] == 15.0  # v_ego + v_rel
        assert lead['status'] is True

    def test_pose_roundtrip(self):
        adapter = OpenpilotAdapter()
        gs = adapter.pose_to_gaussian_state(
            x=100.0, y=5.0, yaw=0.1,
            vx=20.0, vy=0.5, yaw_rate=0.01,
            timestamp=50.0
        )
        pose = adapter.gaussian_state_to_pose_dict(gs)

        assert pose['x'] == 100.0
        assert pose['vx'] == 20.0
```

## Dependencies

Add to `tools/stonesoup/requirements.txt`:
```
stonesoup>=1.8
```
