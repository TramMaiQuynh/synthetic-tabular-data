"""
Unit tests for src.cli.main — argument parsing, validation, and dispatch logic.

These tests are purely unit-level: they mock all heavy dependencies
(ModelTrainer, SyntheticSampler, EvaluationSuite, etc.) so that no I/O,
GPU, or network access is required.

Run with:
    py -3.11 -m pytest tests/test_cli.py -v
"""

from __future__ import annotations

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cli.main import _build_parser, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(args: list[str]):
    """Parse args without calling the command handler."""
    parser = _build_parser()
    return parser.parse_args(args)


# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------

class TestTopLevelParser:
    def test_no_subcommand_exits(self):
        with pytest.raises(SystemExit):
            _parse([])

    def test_verbose_flag(self):
        ns = _parse(["train", "--dataset", "telco_customer_churn",
                      "--data-path", "data/x.csv"])
        # verbose not set by default
        assert not ns.verbose

    def test_verbose_flag_set(self):
        ns = _parse(["-v", "train", "--dataset", "telco_customer_churn",
                      "--data-path", "data/x.csv"])
        assert ns.verbose is True

    def test_artifacts_root_default(self):
        ns = _parse(["train", "--dataset", "telco_customer_churn",
                      "--data-path", "data/x.csv"])
        assert ns.artifacts_root is None  # Default resolved at runtime


# ---------------------------------------------------------------------------
# train subcommand parsing
# ---------------------------------------------------------------------------

class TestTrainParser:
    BASE = ["train", "--dataset", "telco_customer_churn", "--data-path", "data/x.csv"]

    def test_required_dataset(self):
        with pytest.raises(SystemExit):
            _parse(["train", "--data-path", "data/x.csv"])

    def test_required_data_path(self):
        with pytest.raises(SystemExit):
            _parse(["train", "--dataset", "telco_customer_churn"])

    def test_defaults_are_none(self):
        ns = _parse(self.BASE)
        assert ns.model_type is None
        assert ns.epochs is None
        assert ns.batch_size is None
        assert ns.lr is None
        assert ns.epsilon is None
        assert ns.delta is None

    def test_model_type_choices(self):
        for m in ("ctgan", "ctvae", "diffusion"):
            ns = _parse(self.BASE + ["--model-type", m])
            assert ns.model_type == m

    def test_invalid_model_type(self):
        with pytest.raises(SystemExit):
            _parse(self.BASE + ["--model-type", "invalid_model"])

    def test_enable_dp_flag(self):
        ns = _parse(self.BASE + ["--enable-dp"])
        assert ns.enable_dp is True

    def test_dp_defaults_none(self):
        ns = _parse(self.BASE)
        assert not ns.enable_dp
        assert ns.epsilon is None
        assert ns.delta is None
        assert ns.max_grad_norm is None

    def test_epsilon_delta(self):
        ns = _parse(self.BASE + ["--epsilon", "3.0", "--delta", "1e-5"])
        assert ns.epsilon == pytest.approx(3.0)
        assert ns.delta == pytest.approx(1e-5)

    def test_single_constraint(self):
        ns = _parse(self.BASE + ["--constraint", "tenure >= 0"])
        assert ns.constraints == ["tenure >= 0"]

    def test_multiple_constraints(self):
        ns = _parse(self.BASE + ["-c", "A > 0", "-c", "B < 100"])
        assert ns.constraints == ["A > 0", "B < 100"]

    def test_architecture_overrides(self):
        ns = _parse(self.BASE + [
            "--hidden-dim", "256", "--n-layers", "3",
            "--latent-dim", "64", "--noise-dim", "128"
        ])
        assert ns.hidden_dim == 256
        assert ns.n_layers == 3
        assert ns.latent_dim == 64
        assert ns.noise_dim == 128

    def test_sample_after_train(self):
        ns = _parse(self.BASE + ["--sample-after-train", "--sample-rows", "500"])
        assert ns.sample_after_train is True
        assert ns.sample_rows == 500

    def test_early_stopping(self):
        ns = _parse(self.BASE + ["--early-stopping-patience", "10"])
        assert ns.early_stopping_patience == 10


# ---------------------------------------------------------------------------
# sample subcommand parsing
# ---------------------------------------------------------------------------

class TestSampleParser:
    BASE = ["sample", "--dataset", "telco_customer_churn", "--n-rows", "1000"]

    def test_required_n_rows(self):
        with pytest.raises(SystemExit):
            _parse(["sample", "--dataset", "telco_customer_churn"])

    def test_required_dataset(self):
        with pytest.raises(SystemExit):
            _parse(["sample", "--n-rows", "100"])

    def test_defaults(self):
        ns = _parse(self.BASE)
        assert ns.n_rows == 1000
        assert ns.output is None
        assert ns.parquet is False
        assert ns.condition_col is None
        assert ns.condition_val is None
        assert ns.constraints is None
        assert ns.max_retries == 5
        assert ns.device is None

    def test_output_path(self):
        ns = _parse(self.BASE + ["--output", "out/data.csv"])
        assert ns.output == "out/data.csv"

    def test_parquet_flag(self):
        ns = _parse(self.BASE + ["--parquet"])
        assert ns.parquet is True

    def test_conditional_generation(self):
        ns = _parse(self.BASE + ["--condition-col", "Churn", "--condition-val", "Yes"])
        assert ns.condition_col == "Churn"
        assert ns.condition_val == "Yes"

    def test_device_override(self):
        ns = _parse(self.BASE + ["--device", "cuda:0"])
        assert ns.device == "cuda:0"

    def test_batch_size(self):
        ns = _parse(self.BASE + ["--batch-size", "4096"])
        assert ns.batch_size == 4096

    def test_max_retries(self):
        ns = _parse(self.BASE + ["--max-retries", "10"])
        assert ns.max_retries == 10

    def test_checkpoint_override(self):
        ns = _parse(self.BASE + ["--checkpoint", "custom/model.pt"])
        assert ns.checkpoint == "custom/model.pt"


# ---------------------------------------------------------------------------
# evaluate subcommand parsing
# ---------------------------------------------------------------------------

class TestEvaluateParser:
    BASE = [
        "evaluate",
        "--dataset", "telco_customer_churn",
        "--real-csv", "data/real.csv",
        "--synth-csv", "data/synth.csv",
    ]

    def test_required_args(self):
        # Missing --real-csv
        with pytest.raises(SystemExit):
            _parse(["evaluate", "--dataset", "d", "--synth-csv", "s.csv"])

    def test_defaults(self):
        ns = _parse(self.BASE)
        assert ns.target_col is None
        assert ns.sensitive_col is None
        assert ns.real_train_csv is None
        assert ns.real_test_csv is None

    def test_target_and_sensitive(self):
        ns = _parse(self.BASE + ["--target-col", "Churn", "--sensitive-col", "gender"])
        assert ns.target_col == "Churn"
        assert ns.sensitive_col == "gender"

    def test_explicit_split_files(self):
        ns = _parse(self.BASE + [
            "--real-train-csv", "train.csv",
            "--real-test-csv", "test.csv"
        ])
        assert ns.real_train_csv == "train.csv"
        assert ns.real_test_csv == "test.csv"


# ---------------------------------------------------------------------------
# hpo subcommand parsing
# ---------------------------------------------------------------------------

class TestHPOParser:
    BASE = [
        "hpo",
        "--dataset", "telco_customer_churn",
        "--data-path", "data/x.csv",
    ]

    def test_required_args(self):
        with pytest.raises(SystemExit):
            _parse(["hpo", "--dataset", "d"])

    def test_defaults(self):
        ns = _parse(self.BASE)
        assert ns.n_trials == 20
        assert ns.timeout is None
        assert ns.n_jobs == 1
        assert ns.fidelity_weight == pytest.approx(0.7)
        assert ns.hpo_budget_fraction == pytest.approx(0.2)
        assert not ns.train_after_hpo
        assert not ns.enable_dp

    def test_n_trials(self):
        ns = _parse(self.BASE + ["--n-trials", "50"])
        assert ns.n_trials == 50

    def test_train_after_hpo(self):
        ns = _parse(self.BASE + ["--train-after-hpo"])
        assert ns.train_after_hpo is True

    def test_dp_flags(self):
        ns = _parse(self.BASE + ["--enable-dp", "--epsilon", "2.0", "--delta", "1e-6"])
        assert ns.enable_dp is True
        assert ns.epsilon == pytest.approx(2.0)
        assert ns.delta == pytest.approx(1e-6)

    def test_fidelity_weight(self):
        ns = _parse(self.BASE + ["--fidelity-weight", "0.5"])
        assert ns.fidelity_weight == pytest.approx(0.5)

    def test_constraints(self):
        ns = _parse(self.BASE + ["-c", "age >= 18", "-c", "income > 0"])
        assert ns.constraints == ["age >= 18", "income > 0"]


# ---------------------------------------------------------------------------
# cmd_sample: validation of --condition-col / --condition-val pairing
# ---------------------------------------------------------------------------

class TestCmdSampleValidation:
    """Test cmd_sample runtime validation (requires mock of SyntheticSampler)."""

    def _run_sample(self, extra_args: list[str]) -> int:
        """Run the sample subcommand via main() with mocked dependencies."""
        args = [
            "sample",
            "--dataset", "telco_customer_churn",
            "--n-rows", "100",
        ] + extra_args

        mock_config = MagicMock()
        mock_config.model.model_type = "ctgan"
        mock_config.model.epochs = 100
        mock_config.model.batch_size = 256
        mock_config.model.learning_rate = 2e-4
        mock_config.privacy.enable_differential_privacy = False
        mock_config.privacy.target_epsilon = 1.0
        mock_config.privacy.target_delta = 1e-5
        mock_config.privacy.max_grad_norm = 1.0

        mock_df = MagicMock()
        mock_df.shape = (100, 10)
        mock_df.isnull.return_value.sum.return_value.sum.return_value = 0
        mock_df.dtypes.value_counts.return_value = {}
        mock_df.to_csv = MagicMock()
        mock_df.to_parquet = MagicMock()

        mock_sampler = MagicMock()
        mock_sampler.load.return_value = mock_sampler
        mock_sampler.generate.return_value = mock_df

        with patch("src.cli.main._load_config", return_value=mock_config), \
             patch("src.cli.main._resolve_artifacts_root", return_value="/tmp/arts"):
            return main(args)

    def test_condition_col_without_val_returns_1(self):
        result = self._run_sample(["--condition-col", "Churn"])
        assert result == 1

    def test_condition_val_without_col_returns_1(self):
        result = self._run_sample(["--condition-val", "Yes"])
        assert result == 1


# ---------------------------------------------------------------------------
# cmd_train: missing data file returns error
# ---------------------------------------------------------------------------

class TestCmdTrainDataFileValidation:
    def test_missing_data_file_returns_1(self):
        mock_config = MagicMock()
        mock_config.model.model_type = "ctvae"
        mock_config.model.epochs = 100
        mock_config.model.batch_size = 256
        mock_config.model.learning_rate = 2e-4
        mock_config.privacy.enable_differential_privacy = False
        mock_config.privacy.target_epsilon = 1.0
        mock_config.privacy.target_delta = 1e-5
        mock_config.privacy.max_grad_norm = 1.0

        args = [
            "train",
            "--dataset", "telco_customer_churn",
            "--data-path", "/nonexistent/path/data.csv",
        ]
        with patch("src.cli.main._load_config", return_value=mock_config):
            result = main(args)
        assert result == 1


# ---------------------------------------------------------------------------
# cmd_evaluate: missing csv files return error
# ---------------------------------------------------------------------------

class TestCmdEvaluateFileValidation:
    def test_missing_real_csv_returns_1(self):
        mock_config = MagicMock()
        mock_config.model.model_type = "ctvae"

        args = [
            "evaluate",
            "--dataset", "telco_customer_churn",
            "--real-csv", "/nonexistent/real.csv",
            "--synth-csv", "/nonexistent/synth.csv",
        ]
        with patch("src.cli.main._load_config", return_value=mock_config):
            result = main(args)
        assert result == 1
