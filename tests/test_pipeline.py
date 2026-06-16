"""
Unit tests for the MLOps Pipeline.
Tests data ingestion, feature engineering, and API logic.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.generate_synthetic_data import generate_sensor_data
from src.ingestion.preprocess import engineer_features, get_feature_cols


class TestDataGeneration:
    def test_shape(self):
        df = generate_sensor_data(n=100)
        assert len(df) == 100

    def test_columns(self):
        df = generate_sensor_data(n=100)
        expected = {"timestamp", "temperature_c", "vibration_mms", "pressure_bar",
                    "rpm", "oil_level", "machine_id", "failure"}
        assert expected.issubset(set(df.columns))

    def test_no_nulls(self):
        df = generate_sensor_data(n=100)
        assert df.isnull().sum().sum() == 0

    def test_failure_binary(self):
        df = generate_sensor_data(n=500)
        assert set(df["failure"].unique()).issubset({0, 1})

    def test_oil_level_range(self):
        df = generate_sensor_data(n=500)
        assert df["oil_level"].between(0, 1).all()

    def test_machine_ids(self):
        df = generate_sensor_data(n=500)
        assert all(m.startswith("M") for m in df["machine_id"].unique())


class TestFeatureEngineering:
    @pytest.fixture
    def sample_df(self):
        return generate_sensor_data(n=200)

    def test_rolling_features_created(self, sample_df):
        result = engineer_features(sample_df)
        assert "temperature_c_roll_mean_6h" in result.columns
        assert "vibration_mms_roll_std_6h" in result.columns

    def test_ratio_features_created(self, sample_df):
        result = engineer_features(sample_df)
        assert "temp_vibration_ratio" in result.columns
        assert "pressure_rpm_ratio" in result.columns

    def test_no_new_nulls_in_numerics(self, sample_df):
        result = engineer_features(sample_df)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert result[numeric_cols].isnull().sum().sum() == 0

    def test_feature_count(self, sample_df):
        result = engineer_features(sample_df)
        feature_cols = get_feature_cols(result)
        assert len(feature_cols) >= 15


class TestMonitoring:
    def test_log_prediction_batch(self, tmp_path):
        from src.monitoring.monitor import log_prediction_batch
        import src.monitoring.monitor as mon_module
        original = mon_module.MONITOR_LOG
        mon_module.MONITOR_LOG = str(tmp_path / "test_log.jsonl")

        preds = [0.1, 0.3, 0.7, 0.9, 0.2]
        result = log_prediction_batch(preds, model_version="test-v1")

        assert "avg_failure_probability" in result
        assert result["n_predictions"] == 5
        assert "alert" in result

        mon_module.MONITOR_LOG = original

    def test_psi_zero_for_identical(self):
        from src.monitoring.monitor import population_stability_index
        arr = np.random.uniform(0, 1, 500)
        psi = population_stability_index(arr, arr)
        assert psi < 0.01
