"""
Tests for Sub-Task 1: Project Scaffold & Shared Models.

Covers:
- missionmind package imports cleanly.
- config.py values exist and have correct types.
- models/mission.py: Waypoint, MissionStatus, MissionPlan.
- models/events.py: EventType, MissionEvent.
- schemas/outputs.py: ScienceAnalysis, ResourceBudget, RiskAssessment, MissionPlanOutput.
- Pydantic validation errors on bad inputs.
- JSON serialisation round-trips.
"""

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------

def test_package_imports():
    """The top-level missionmind package must import without error."""
    import missionmind  # noqa: F401
    assert missionmind.__version__ == "0.1.0"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_config_imports(self):
        from missionmind import config
        assert hasattr(config, "LLM_MODEL_NAME")
        assert hasattr(config, "MIN_RETURN_BATTERY_PCT")
        assert hasattr(config, "CRITICAL_BATTERY_PCT")
        assert hasattr(config, "MAX_MISSION_DURATION_HOURS")
        assert hasattr(config, "COMM_TIMEOUT_SECONDS")
        assert hasattr(config, "MAX_TERRAIN_RISK_SCORE")
        assert hasattr(config, "MAX_PLANNING_RETRIES")

    def test_config_types(self):
        from missionmind import config
        assert isinstance(config.LLM_MODEL_NAME, str)
        assert isinstance(config.MIN_RETURN_BATTERY_PCT, float)
        assert isinstance(config.CRITICAL_BATTERY_PCT, float)
        assert isinstance(config.MAX_MISSION_DURATION_HOURS, float)
        assert isinstance(config.COMM_TIMEOUT_SECONDS, int)
        assert isinstance(config.MAX_TERRAIN_RISK_SCORE, float)
        assert isinstance(config.MAX_PLANNING_RETRIES, int)

    def test_battery_thresholds_are_fractions(self):
        from missionmind import config
        assert 0.0 < config.MIN_RETURN_BATTERY_PCT < 1.0
        assert 0.0 < config.CRITICAL_BATTERY_PCT < 1.0
        # Critical must be less than minimum return threshold
        assert config.CRITICAL_BATTERY_PCT < config.MIN_RETURN_BATTERY_PCT

    def test_max_terrain_risk_is_fraction(self):
        from missionmind import config
        assert 0.0 < config.MAX_TERRAIN_RISK_SCORE <= 1.0

    def test_config_env_override(self, monkeypatch):
        """Environment variables must override default values."""
        monkeypatch.setenv("MAX_TERRAIN_RISK_SCORE", "0.55")
        # Re-import the module to pick up the env var.
        import importlib
        import missionmind.config as cfg_module
        importlib.reload(cfg_module)
        assert cfg_module.MAX_TERRAIN_RISK_SCORE == pytest.approx(0.55)
        # Restore by reloading without the env var.
        monkeypatch.delenv("MAX_TERRAIN_RISK_SCORE")
        importlib.reload(cfg_module)


# ---------------------------------------------------------------------------
# MissionStatus
# ---------------------------------------------------------------------------

class TestMissionStatus:
    def test_all_statuses_exist(self):
        from missionmind.models.mission import MissionStatus
        assert set(MissionStatus) == {
            MissionStatus.PENDING,
            MissionStatus.ACTIVE,
            MissionStatus.REPLANNING,
            MissionStatus.ABORTED,
            MissionStatus.COMPLETE,
        }

    def test_status_is_string_enum(self):
        from missionmind.models.mission import MissionStatus
        assert MissionStatus.ACTIVE == "ACTIVE"


# ---------------------------------------------------------------------------
# Waypoint
# ---------------------------------------------------------------------------

class TestWaypoint:
    def test_minimal_construction(self):
        from missionmind.models.mission import Waypoint
        wp = Waypoint(x=100.0, y=200.0)
        assert wp.x == 100.0
        assert wp.y == 200.0
        # Defaults
        assert wp.scientific_value == 0.0
        assert wp.terrain_risk == 0.0
        assert wp.is_base is False
        assert wp.label == ""
        assert isinstance(wp.id, str) and len(wp.id) > 0

    def test_full_construction(self):
        from missionmind.models.mission import Waypoint
        wp = Waypoint(
            id="wp-001",
            x=50.0,
            y=-30.0,
            scientific_value=0.85,
            terrain_risk=0.2,
            estimated_travel_time_minutes=12.5,
            estimated_energy_wh=18.3,
            is_base=False,
            label="crater-rim-A",
        )
        assert wp.id == "wp-001"
        assert wp.scientific_value == pytest.approx(0.85)
        assert wp.terrain_risk == pytest.approx(0.2)
        assert wp.label == "crater-rim-A"

    def test_base_waypoint(self):
        from missionmind.models.mission import Waypoint
        base = Waypoint(x=0.0, y=0.0, is_base=True, label="BASE")
        assert base.is_base is True

    def test_scientific_value_out_of_range_rejected(self):
        from missionmind.models.mission import Waypoint
        with pytest.raises(ValidationError):
            Waypoint(x=0.0, y=0.0, scientific_value=1.5)

    def test_terrain_risk_negative_rejected(self):
        from missionmind.models.mission import Waypoint
        with pytest.raises(ValidationError):
            Waypoint(x=0.0, y=0.0, terrain_risk=-0.1)

    def test_negative_energy_rejected(self):
        from missionmind.models.mission import Waypoint
        with pytest.raises(ValidationError):
            Waypoint(x=0.0, y=0.0, estimated_energy_wh=-5.0)

    def test_waypoint_json_serialisation(self):
        from missionmind.models.mission import Waypoint
        wp = Waypoint(x=1.0, y=2.0, scientific_value=0.5)
        data = json.loads(wp.model_dump_json())
        assert data["x"] == 1.0
        assert data["scientific_value"] == pytest.approx(0.5)

    def test_unique_ids_auto_generated(self):
        from missionmind.models.mission import Waypoint
        wp1 = Waypoint(x=0.0, y=0.0)
        wp2 = Waypoint(x=1.0, y=1.0)
        assert wp1.id != wp2.id


# ---------------------------------------------------------------------------
# MissionPlan
# ---------------------------------------------------------------------------

class TestMissionPlan:
    def _make_plan(self):
        from missionmind.models.mission import MissionPlan, MissionStatus, Waypoint
        science_wp = Waypoint(
            id="wp-sci",
            x=100.0,
            y=50.0,
            scientific_value=0.9,
            terrain_risk=0.1,
            estimated_travel_time_minutes=20.0,
            estimated_energy_wh=30.0,
        )
        base_wp = Waypoint(id="wp-base", x=0.0, y=0.0, is_base=True, label="BASE")
        return MissionPlan(
            waypoints=[science_wp, base_wp],
            total_energy_wh=30.0,
            total_time_minutes=20.0,
            status=MissionStatus.PENDING,
            reasoning="Test plan",
            confidence=0.75,
        )

    def test_construction(self):
        plan = self._make_plan()
        from missionmind.models.mission import MissionStatus
        assert plan.status == MissionStatus.PENDING
        assert len(plan.waypoints) == 2
        assert isinstance(plan.plan_id, str)
        assert isinstance(plan.created_at, datetime)

    def test_default_status_is_pending(self):
        from missionmind.models.mission import MissionPlan, MissionStatus
        plan = MissionPlan()
        assert plan.status == MissionStatus.PENDING

    def test_has_return_waypoint_true(self):
        plan = self._make_plan()
        assert plan.has_return_waypoint() is True

    def test_has_return_waypoint_false_when_no_base(self):
        from missionmind.models.mission import MissionPlan, Waypoint
        wp = Waypoint(x=1.0, y=2.0)
        plan = MissionPlan(waypoints=[wp])
        assert plan.has_return_waypoint() is False

    def test_science_waypoints_excludes_base(self):
        plan = self._make_plan()
        science_wps = plan.science_waypoints()
        assert len(science_wps) == 1
        assert science_wps[0].id == "wp-sci"

    def test_confidence_out_of_range_rejected(self):
        from missionmind.models.mission import MissionPlan
        with pytest.raises(ValidationError):
            MissionPlan(confidence=1.5)

    def test_negative_energy_rejected(self):
        from missionmind.models.mission import MissionPlan
        with pytest.raises(ValidationError):
            MissionPlan(total_energy_wh=-10.0)

    def test_json_serialisation(self):
        plan = self._make_plan()
        data = json.loads(plan.model_dump_json())
        assert data["status"] == "PENDING"
        assert len(data["waypoints"]) == 2

    def test_unique_plan_ids(self):
        from missionmind.models.mission import MissionPlan
        p1 = MissionPlan()
        p2 = MissionPlan()
        assert p1.plan_id != p2.plan_id

    def test_empty_plan_no_return_waypoint(self):
        from missionmind.models.mission import MissionPlan
        plan = MissionPlan()
        assert plan.has_return_waypoint() is False


# ---------------------------------------------------------------------------
# EventType
# ---------------------------------------------------------------------------

class TestEventType:
    def test_all_event_types_exist(self):
        from missionmind.models.events import EventType
        assert set(EventType) == {
            EventType.BATTERY_FAILURE,
            EventType.COMM_LOSS,
            EventType.TERRAIN_HAZARD,
            EventType.NEW_DISCOVERY,
            EventType.RETURN_TO_BASE,
        }

    def test_event_type_is_string_enum(self):
        from missionmind.models.events import EventType
        assert EventType.COMM_LOSS == "COMM_LOSS"


# ---------------------------------------------------------------------------
# MissionEvent
# ---------------------------------------------------------------------------

class TestMissionEvent:
    def test_minimal_construction(self):
        from missionmind.models.events import EventType, MissionEvent
        evt = MissionEvent(event_type=EventType.TERRAIN_HAZARD, severity=0.6)
        assert evt.event_type == EventType.TERRAIN_HAZARD
        assert evt.severity == pytest.approx(0.6)
        assert evt.payload == {}
        assert isinstance(evt.timestamp, datetime)

    def test_payload_stored(self):
        from missionmind.models.events import EventType, MissionEvent
        payload = {"waypoint_id": "wp-001", "obstacle": "boulder"}
        evt = MissionEvent(
            event_type=EventType.TERRAIN_HAZARD,
            severity=0.8,
            payload=payload,
        )
        assert evt.payload["waypoint_id"] == "wp-001"

    def test_severity_above_one_rejected(self):
        from missionmind.models.events import EventType, MissionEvent
        with pytest.raises(ValueError, match="severity"):
            MissionEvent(event_type=EventType.BATTERY_FAILURE, severity=1.5)

    def test_severity_below_zero_rejected(self):
        from missionmind.models.events import EventType, MissionEvent
        with pytest.raises(ValueError, match="severity"):
            MissionEvent(event_type=EventType.BATTERY_FAILURE, severity=-0.1)

    def test_timestamp_defaults_to_utc(self):
        from missionmind.models.events import EventType, MissionEvent
        evt = MissionEvent(event_type=EventType.COMM_LOSS, severity=0.5)
        assert evt.timestamp.tzinfo is not None

    def test_timestamp_can_be_overridden(self):
        from missionmind.models.events import EventType, MissionEvent
        ts = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        evt = MissionEvent(
            event_type=EventType.NEW_DISCOVERY,
            severity=0.3,
            timestamp=ts,
        )
        assert evt.timestamp == ts


# ---------------------------------------------------------------------------
# ScienceAnalysis
# ---------------------------------------------------------------------------

class TestScienceAnalysis:
    def test_empty_construction(self):
        from missionmind.schemas.outputs import ScienceAnalysis
        sa = ScienceAnalysis()
        assert sa.scored_targets == []
        assert sa.priority_order == []
        assert sa.reasoning == ""

    def test_full_construction(self):
        from missionmind.schemas.outputs import ScienceAnalysis, ScoredTarget
        sa = ScienceAnalysis(
            scored_targets=[
                ScoredTarget(waypoint_id="wp-1", scientific_value=0.9, justification="iron-rich"),
                ScoredTarget(waypoint_id="wp-2", scientific_value=0.4, justification="sandy"),
            ],
            priority_order=["wp-1", "wp-2"],
            reasoning="Prioritised by iron content",
        )
        assert len(sa.scored_targets) == 2
        assert sa.priority_order[0] == "wp-1"

    def test_scientific_value_clamped_above_one(self):
        from missionmind.schemas.outputs import ScoredTarget
        with pytest.raises(ValidationError):
            ScoredTarget(waypoint_id="wp-1", scientific_value=1.1)

    def test_scientific_value_clamped_below_zero(self):
        from missionmind.schemas.outputs import ScoredTarget
        with pytest.raises(ValidationError):
            ScoredTarget(waypoint_id="wp-1", scientific_value=-0.1)

    def test_json_serialisation(self):
        from missionmind.schemas.outputs import ScienceAnalysis, ScoredTarget
        sa = ScienceAnalysis(
            scored_targets=[ScoredTarget(waypoint_id="wp-1", scientific_value=0.7)],
            priority_order=["wp-1"],
            reasoning="test",
        )
        data = json.loads(sa.model_dump_json())
        assert data["scored_targets"][0]["waypoint_id"] == "wp-1"


# ---------------------------------------------------------------------------
# ResourceBudget
# ---------------------------------------------------------------------------

class TestResourceBudget:
    def test_construction(self):
        from missionmind.schemas.outputs import ResourceBudget
        rb = ResourceBudget(
            available_energy_wh=120.0,
            available_time_minutes=360.0,
            recommended_waypoints=["wp-1", "wp-2"],
            energy_per_waypoint={"wp-1": 30.0, "wp-2": 45.0},
            reasoning="Fits within budget",
        )
        assert rb.available_energy_wh == pytest.approx(120.0)
        assert len(rb.recommended_waypoints) == 2

    def test_negative_energy_rejected(self):
        from missionmind.schemas.outputs import ResourceBudget
        with pytest.raises(ValidationError):
            ResourceBudget(available_energy_wh=-5.0, available_time_minutes=100.0)

    def test_json_serialisation(self):
        from missionmind.schemas.outputs import ResourceBudget
        rb = ResourceBudget(available_energy_wh=50.0, available_time_minutes=120.0)
        data = json.loads(rb.model_dump_json())
        assert data["available_energy_wh"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# RiskAssessment
# ---------------------------------------------------------------------------

class TestRiskAssessment:
    def test_default_risk_level_is_low(self):
        from missionmind.schemas.outputs import RiskAssessment, RiskLevel
        ra = RiskAssessment()
        assert ra.overall_risk_level == RiskLevel.LOW

    def test_high_risk_construction(self):
        from missionmind.schemas.outputs import RiskAssessment, RiskLevel, WaypointRisk
        ra = RiskAssessment(
            waypoint_risks=[
                WaypointRisk(waypoint_id="wp-1", risk_score=0.9, factors=["steep slope"]),
            ],
            overall_risk_level=RiskLevel.HIGH,
            recommended_exclusions=["wp-1"],
            reasoning="Too dangerous",
        )
        assert ra.overall_risk_level == RiskLevel.HIGH
        assert "wp-1" in ra.recommended_exclusions

    def test_risk_score_out_of_range_rejected(self):
        from missionmind.schemas.outputs import WaypointRisk
        with pytest.raises(ValidationError):
            WaypointRisk(waypoint_id="wp-1", risk_score=1.5)

    def test_risk_level_enum_values(self):
        from missionmind.schemas.outputs import RiskLevel
        assert set(RiskLevel) == {RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH}

    def test_json_serialisation(self):
        from missionmind.schemas.outputs import RiskAssessment, RiskLevel
        ra = RiskAssessment(overall_risk_level=RiskLevel.MEDIUM)
        data = json.loads(ra.model_dump_json())
        assert data["overall_risk_level"] == "MEDIUM"


# ---------------------------------------------------------------------------
# MissionPlanOutput
# ---------------------------------------------------------------------------

class TestMissionPlanOutput:
    def test_empty_construction(self):
        from missionmind.schemas.outputs import MissionPlanOutput
        mpo = MissionPlanOutput()
        assert mpo.planned_waypoints == []
        assert mpo.confidence == 0.0

    def test_full_construction(self):
        from missionmind.schemas.outputs import MissionPlanOutput, PlannedWaypointEntry
        mpo = MissionPlanOutput(
            planned_waypoints=[
                PlannedWaypointEntry(
                    waypoint_id="wp-1",
                    visit_order=1,
                    expected_science_value=0.8,
                    expected_energy_wh=25.0,
                ),
                PlannedWaypointEntry(
                    waypoint_id="wp-base",
                    visit_order=2,
                    expected_science_value=0.0,
                    expected_energy_wh=10.0,
                ),
            ],
            total_estimated_energy_wh=35.0,
            total_estimated_time_minutes=60.0,
            confidence=0.9,
            reasoning="Balanced plan",
        )
        assert len(mpo.planned_waypoints) == 2
        assert mpo.confidence == pytest.approx(0.9)

    def test_visit_order_below_one_rejected(self):
        from missionmind.schemas.outputs import PlannedWaypointEntry
        with pytest.raises(ValidationError):
            PlannedWaypointEntry(waypoint_id="wp-1", visit_order=0)

    def test_confidence_out_of_range_rejected(self):
        from missionmind.schemas.outputs import MissionPlanOutput
        with pytest.raises(ValidationError):
            MissionPlanOutput(confidence=2.0)

    def test_json_serialisation(self):
        from missionmind.schemas.outputs import MissionPlanOutput, PlannedWaypointEntry
        mpo = MissionPlanOutput(
            planned_waypoints=[
                PlannedWaypointEntry(waypoint_id="wp-1", visit_order=1)
            ],
            confidence=0.5,
        )
        data = json.loads(mpo.model_dump_json())
        assert data["confidence"] == pytest.approx(0.5)
        assert data["planned_waypoints"][0]["visit_order"] == 1


# ---------------------------------------------------------------------------
# models package re-export
# ---------------------------------------------------------------------------

class TestModelsPackageExports:
    def test_top_level_models_export(self):
        from missionmind.models import (
            EventType,
            MissionEvent,
            MissionPlan,
            MissionStatus,
            Waypoint,
        )
        assert MissionStatus.ACTIVE == "ACTIVE"
        assert EventType.COMM_LOSS == "COMM_LOSS"
        wp = Waypoint(x=0.0, y=0.0)
        assert wp is not None
        plan = MissionPlan()
        assert plan is not None
        evt = MissionEvent(event_type=EventType.BATTERY_FAILURE, severity=0.5)
        assert evt is not None


# ---------------------------------------------------------------------------
# schemas package re-export
# ---------------------------------------------------------------------------

class TestSchemasPackageExports:
    def test_top_level_schemas_export(self):
        from missionmind.schemas import (
            MissionPlanOutput,
            ResourceBudget,
            RiskAssessment,
            RiskLevel,
            ScienceAnalysis,
            ScoredTarget,
        )
        assert RiskLevel.HIGH == "HIGH"
        sa = ScienceAnalysis()
        rb = ResourceBudget(available_energy_wh=10.0, available_time_minutes=60.0)
        ra = RiskAssessment()
        mpo = MissionPlanOutput()
        st = ScoredTarget(waypoint_id="wp-1", scientific_value=0.5)
        assert sa is not None
        assert rb is not None
        assert ra is not None
        assert mpo is not None
        assert st is not None
