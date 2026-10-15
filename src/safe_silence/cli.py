"""Command-line entry point for reproducible project operations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import yaml

from safe_silence.config import ConfigurationError, load_config
from safe_silence.paths import resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="safe-silence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("validate-config", "validate all YAML files and scientific invariants"),
        ("show-config", "print the fully merged configuration"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
        child.add_argument("--json", action="store_true", help="emit JSON instead of YAML")
    audit = subparsers.add_parser("audit-ooi", help="audit downloaded OOI ERDDAP chunks")
    audit.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    audit.add_argument("--raw-root", type=Path, help="override YAML raw OOI directory")
    audit.add_argument("--output-root", type=Path, help="override YAML audit output directory")
    preprocess = subparsers.add_parser(
        "preprocess-ooi", help="run Task G harmonization and immutable split construction"
    )
    preprocess.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    preprocess.add_argument("--raw-root", type=Path, help="override YAML raw OOI directory")
    preprocess.add_argument("--audit-root", type=Path, help="override Task F audit directory")
    preprocess.add_argument("--output-root", type=Path, help="override Task G output directory")
    label = subparsers.add_parser(
        "label-events", help="run Task H leakage-safe critical-event labeling"
    )
    label.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    label.add_argument("--input-root", type=Path, help="override Task G input directory")
    label.add_argument("--output-root", type=Path, help="override Task H output directory")
    predictor = subparsers.add_parser(
        "train-predictor", help="run Task I causal prediction and uncertainty calibration"
    )
    predictor.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    predictor.add_argument("--input-root", type=Path, help="override Task H input directory")
    predictor.add_argument(
        "--artifact-root", type=Path, help="override Task I model artifact directory"
    )
    predictor.add_argument("--result-root", type=Path, help="override Task I result directory")
    semantic = subparsers.add_parser(
        "build-semantic-evidence",
        help="run Task J causal semantic innovation and bounded-validity decisions",
    )
    semantic.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    semantic.add_argument("--prediction-root", type=Path, help="override Task I result directory")
    semantic.add_argument(
        "--prediction-artifact-root", type=Path, help="override Task I artifact directory"
    )
    semantic.add_argument("--event-root", type=Path, help="override Task H directory")
    semantic.add_argument("--output-root", type=Path, help="override Task J result directory")
    network = subparsers.add_parser(
        "simulate-network",
        help="run Task K matched acoustic-network realizations and transport evaluation",
    )
    network.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    network.add_argument("--input-root", type=Path, help="override Task J result directory")
    network.add_argument("--output-root", type=Path, help="override Task K result directory")
    receiver = subparsers.add_parser(
        "interpret-silence",
        help="run Task L receiver dual-belief and bounded-silence interpretation",
    )
    receiver.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    receiver.add_argument("--input-root", type=Path, help="override Task K result directory")
    receiver.add_argument("--output-root", type=Path, help="override Task L result directory")
    cause = subparsers.add_parser(
        "infer-causes",
        help="run Task M receiver-observable coarse ambiguity-cause inference",
    )
    cause.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    cause.add_argument("--input-root", type=Path, help="override Task L result directory")
    cause.add_argument("--output-root", type=Path, help="override Task M result directory")
    recovery = subparsers.add_parser(
        "recover-bounded",
        help="run Task N receiver-observable bounded recovery and matched references",
    )
    recovery.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    recovery.add_argument("--input-root", type=Path, help="override Task M result directory")
    recovery.add_argument("--network-root", type=Path, help="override Task K result directory")
    recovery.add_argument("--output-root", type=Path, help="override Task N result directory")
    fusion = subparsers.add_parser(
        "fuse-relay",
        help="run Task O criticality-preserving associative relay fusion",
    )
    fusion.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    fusion.add_argument("--input-root", type=Path, help="override Task J result directory")
    fusion.add_argument("--output-root", type=Path, help="override Task O result directory")
    scheduler = subparsers.add_parser(
        "schedule-risk",
        help="run Task P causal tail-risk-reduction-per-joule scheduling",
    )
    scheduler.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    scheduler.add_argument("--fusion-root", type=Path, help="override Task O result directory")
    scheduler.add_argument("--network-root", type=Path, help="override Task K result directory")
    scheduler.add_argument("--output-root", type=Path, help="override Task P result directory")
    integration = subparsers.add_parser(
        "evaluate-end-to-end",
        help="run Task Q matched-realization relay delivery and baseline evaluation",
    )
    integration.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    integration.add_argument("--scheduler-root", type=Path, help="override Task P directory")
    integration.add_argument("--fusion-root", type=Path, help="override Task O directory")
    integration.add_argument("--network-root", type=Path, help="override Task K directory")
    integration.add_argument("--output-root", type=Path, help="override Task Q directory")
    reintegration = subparsers.add_parser(
        "reintegrate-system",
        help="run Task R relay-receiver-recovery reintegration and paired evaluation",
    )
    reintegration.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    reintegration.add_argument("--integration-root", type=Path, help="override Task Q directory")
    reintegration.add_argument("--scheduler-root", type=Path, help="override Task P directory")
    reintegration.add_argument("--fusion-root", type=Path, help="override Task O directory")
    reintegration.add_argument("--network-root", type=Path, help="override Task K directory")
    reintegration.add_argument("--output-root", type=Path, help="override Task R directory")
    confirmation = subparsers.add_parser(
        "confirm-independent",
        help="run Task S independently seeded confirmatory replication",
    )
    confirmation.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    confirmation.add_argument("--semantic-root", type=Path, help="override Task J directory")
    confirmation.add_argument("--fusion-root", type=Path, help="override Task O directory")
    confirmation.add_argument(
        "--prior-reintegration-root", type=Path, help="override frozen Task R directory"
    )
    confirmation.add_argument("--output-root", type=Path, help="override Task S directory")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, yaml.YAMLError, ConfigurationError) as exc:
        raise SystemExit(f"configuration error: {exc}") from exc

    if args.command == "validate-config":
        print(f"VALID: {args.config.resolve()}")
        return
    if args.command == "audit-ooi":
        from safe_silence.data.audit import AuditError, audit_ooi

        raw_root = (
            args.raw_root.resolve()
            if args.raw_root
            else resolve_project_path(Path(config["project"]["paths"]["raw_data"]) / "ooi")
        )
        output_root = (
            args.output_root.resolve()
            if args.output_root
            else resolve_project_path(config["data"]["audit"]["output_directory"])
        )
        try:
            result = audit_ooi(raw_root, output_root, config["data"]["audit"])
        except AuditError as exc:
            raise SystemExit(f"audit error: {exc}") from exc
        print(json.dumps(result, indent=2))
        return
    if args.command == "preprocess-ooi":
        from safe_silence.data.preprocess import PreprocessingError, preprocess_ooi

        raw_root = (
            args.raw_root.resolve()
            if args.raw_root
            else resolve_project_path(Path(config["project"]["paths"]["raw_data"]) / "ooi")
        )
        audit_root = (
            args.audit_root.resolve()
            if args.audit_root
            else resolve_project_path(config["preprocessing"]["audit_directory"])
        )
        output_root = (
            args.output_root.resolve()
            if args.output_root
            else resolve_project_path(config["preprocessing"]["output_directory"])
        )
        try:
            result = preprocess_ooi(raw_root, audit_root, output_root, config["preprocessing"])
        except PreprocessingError as exc:
            raise SystemExit(f"preprocessing error: {exc}") from exc
        print(json.dumps(result, indent=2))
        return
    if args.command == "label-events":
        from safe_silence.data.events import EventLabelingError, label_events

        input_root = (
            args.input_root.resolve()
            if args.input_root
            else resolve_project_path(config["events"]["input_directory"])
        )
        output_root = (
            args.output_root.resolve()
            if args.output_root
            else resolve_project_path(config["events"]["output_directory"])
        )
        try:
            result = label_events(
                input_root, output_root, config["events"], config["preprocessing"]
            )
        except EventLabelingError as exc:
            raise SystemExit(f"event-labeling error: {exc}") from exc
        print(json.dumps(result, indent=2))
        return
    if args.command == "train-predictor":
        from safe_silence.models.predictor import PredictorError, train_predictor

        input_root = (
            args.input_root.resolve()
            if args.input_root
            else resolve_project_path(config["model"]["input_directory"])
        )
        artifact_root = (
            args.artifact_root.resolve()
            if args.artifact_root
            else resolve_project_path(config["model"]["artifact_directory"])
        )
        result_root = (
            args.result_root.resolve()
            if args.result_root
            else resolve_project_path(config["model"]["result_directory"])
        )
        try:
            result = train_predictor(
                input_root,
                artifact_root,
                result_root,
                config["model"],
                config["preprocessing"],
                config["project"],
            )
        except PredictorError as exc:
            raise SystemExit(f"predictor error: {exc}") from exc
        print(json.dumps(result, indent=2))
        return
    if args.command == "build-semantic-evidence":
        from safe_silence.protocol.semantic import SemanticEvidenceError, build_semantic_evidence

        semantic_config = config["semantic"]
        prediction_root = (
            args.prediction_root.resolve()
            if args.prediction_root
            else resolve_project_path(semantic_config["prediction_directory"])
        )
        prediction_artifact_root = (
            args.prediction_artifact_root.resolve()
            if args.prediction_artifact_root
            else resolve_project_path(semantic_config["prediction_artifact_directory"])
        )
        event_root = (
            args.event_root.resolve()
            if args.event_root
            else resolve_project_path(semantic_config["event_directory"])
        )
        output_root = (
            args.output_root.resolve()
            if args.output_root
            else resolve_project_path(semantic_config["output_directory"])
        )
        try:
            result = build_semantic_evidence(
                prediction_root,
                prediction_artifact_root,
                event_root,
                output_root,
                semantic_config,
                config["preprocessing"],
            )
        except SemanticEvidenceError as exc:
            raise SystemExit(f"semantic-evidence error: {exc}") from exc
        print(json.dumps(result, indent=2))
        return
    if args.command == "simulate-network":
        from safe_silence.simulation.network import NetworkSimulationError, simulate_network

        network_config = config["network"]
        input_root = (
            args.input_root.resolve()
            if args.input_root
            else resolve_project_path(network_config["input_directory"])
        )
        output_root = (
            args.output_root.resolve()
            if args.output_root
            else resolve_project_path(network_config["output_directory"])
        )
        try:
            result = simulate_network(input_root, output_root, network_config)
        except NetworkSimulationError as exc:
            raise SystemExit(f"network-simulation error: {exc}") from exc
        print(json.dumps(result, indent=2))
        return
    if args.command == "interpret-silence":
        from safe_silence.protocol.receiver import (
            ReceiverInferenceError,
            build_receiver_beliefs,
        )

        receiver_config = config["receiver"]
        input_root = (
            args.input_root.resolve()
            if args.input_root
            else resolve_project_path(receiver_config["input_directory"])
        )
        output_root = (
            args.output_root.resolve()
            if args.output_root
            else resolve_project_path(receiver_config["output_directory"])
        )
        try:
            result = build_receiver_beliefs(input_root, output_root, receiver_config)
        except ReceiverInferenceError as exc:
            raise SystemExit(f"receiver-inference error: {exc}") from exc
        print(json.dumps(result, indent=2))
        return
    if args.command == "infer-causes":
        from safe_silence.protocol.cause import (
            CauseInferenceError,
            infer_ambiguity_causes,
        )

        cause_config = config["cause"]
        input_root = (
            args.input_root.resolve()
            if args.input_root
            else resolve_project_path(cause_config["input_directory"])
        )
        output_root = (
            args.output_root.resolve()
            if args.output_root
            else resolve_project_path(cause_config["output_directory"])
        )
        try:
            result = infer_ambiguity_causes(input_root, output_root, cause_config)
        except CauseInferenceError as exc:
            raise SystemExit(f"cause-inference error: {exc}") from exc
        print(json.dumps(result, indent=2))
        return
    if args.command == "recover-bounded":
        from safe_silence.protocol.recovery import RecoveryError, run_bounded_recovery

        recovery_config = config["recovery"]
        input_root = (
            args.input_root.resolve()
            if args.input_root
            else resolve_project_path(recovery_config["input_directory"])
        )
        network_root = (
            args.network_root.resolve()
            if args.network_root
            else resolve_project_path(recovery_config["network_directory"])
        )
        output_root = (
            args.output_root.resolve()
            if args.output_root
            else resolve_project_path(recovery_config["output_directory"])
        )
        try:
            result = run_bounded_recovery(
                input_root,
                network_root,
                output_root,
                recovery_config,
                config["network"],
            )
        except RecoveryError as exc:
            raise SystemExit(f"bounded-recovery error: {exc}") from exc
        print(json.dumps(result, indent=2))
        return
    if args.command == "fuse-relay":
        from safe_silence.protocol.fusion import FusionError, run_relay_fusion

        fusion_config = config["fusion"]
        input_root = (
            args.input_root.resolve()
            if args.input_root
            else resolve_project_path(fusion_config["input_directory"])
        )
        output_root = (
            args.output_root.resolve()
            if args.output_root
            else resolve_project_path(fusion_config["output_directory"])
        )
        try:
            result = run_relay_fusion(input_root, output_root, fusion_config)
        except FusionError as exc:
            raise SystemExit(f"relay-fusion error: {exc}") from exc
        print(json.dumps(result, indent=2))
        return
    if args.command == "schedule-risk":
        from safe_silence.protocol.scheduler import SchedulerError, run_scheduler

        scheduler_config = config["scheduler"]
        fusion_root = (
            args.fusion_root.resolve()
            if args.fusion_root
            else resolve_project_path(scheduler_config["fusion_directory"])
        )
        network_root = (
            args.network_root.resolve()
            if args.network_root
            else resolve_project_path(scheduler_config["network_directory"])
        )
        output_root = (
            args.output_root.resolve()
            if args.output_root
            else resolve_project_path(scheduler_config["output_directory"])
        )
        try:
            result = run_scheduler(
                fusion_root,
                network_root,
                output_root,
                scheduler_config,
                config["network"],
            )
        except SchedulerError as exc:
            raise SystemExit(f"risk-scheduler error: {exc}") from exc
        print(json.dumps(result, indent=2))
        return
    if args.command == "evaluate-end-to-end":
        from safe_silence.evaluation.integration import IntegrationError, run_integration

        integration_config = config["integration"]
        scheduler_root = (
            args.scheduler_root.resolve()
            if args.scheduler_root
            else resolve_project_path(integration_config["scheduler_directory"])
        )
        fusion_root = (
            args.fusion_root.resolve()
            if args.fusion_root
            else resolve_project_path(integration_config["fusion_directory"])
        )
        network_root = (
            args.network_root.resolve()
            if args.network_root
            else resolve_project_path(integration_config["network_directory"])
        )
        output_root = (
            args.output_root.resolve()
            if args.output_root
            else resolve_project_path(integration_config["output_directory"])
        )
        try:
            result = run_integration(
                scheduler_root,
                fusion_root,
                network_root,
                output_root,
                integration_config,
                config["scheduler"],
                config["network"],
                config["preprocessing"],
            )
        except IntegrationError as exc:
            raise SystemExit(f"end-to-end integration error: {exc}") from exc
        print(json.dumps(result, indent=2))
        return
    if args.command == "reintegrate-system":
        from safe_silence.evaluation.reintegration import (
            ReintegrationError,
            run_reintegration,
        )

        reintegration_config = config["reintegration"]
        integration_root = (
            args.integration_root.resolve()
            if args.integration_root
            else resolve_project_path(reintegration_config["integration_directory"])
        )
        scheduler_root = (
            args.scheduler_root.resolve()
            if args.scheduler_root
            else resolve_project_path(reintegration_config["scheduler_directory"])
        )
        fusion_root = (
            args.fusion_root.resolve()
            if args.fusion_root
            else resolve_project_path(reintegration_config["fusion_directory"])
        )
        network_root = (
            args.network_root.resolve()
            if args.network_root
            else resolve_project_path(reintegration_config["network_directory"])
        )
        output_root = (
            args.output_root.resolve()
            if args.output_root
            else resolve_project_path(reintegration_config["output_directory"])
        )
        try:
            result = run_reintegration(
                integration_root,
                scheduler_root,
                fusion_root,
                network_root,
                output_root,
                reintegration_config,
                config["integration"],
                config["scheduler"],
                config["network"],
                config["receiver"],
                config["cause"],
                config["recovery"],
                config["preprocessing"],
            )
        except ReintegrationError as exc:
            raise SystemExit(f"system-reintegration error: {exc}") from exc
        print(json.dumps(result, indent=2))
        return
    if args.command == "confirm-independent":
        from safe_silence.evaluation.confirmation import run_confirmation

        confirmation_config = config["confirmation"]
        semantic_root = (
            args.semantic_root.resolve()
            if args.semantic_root
            else resolve_project_path(confirmation_config["semantic_directory"])
        )
        fusion_root = (
            args.fusion_root.resolve()
            if args.fusion_root
            else resolve_project_path(confirmation_config["fusion_directory"])
        )
        prior_reintegration_root = (
            args.prior_reintegration_root.resolve()
            if args.prior_reintegration_root
            else resolve_project_path(confirmation_config["prior_reintegration_directory"])
        )
        output_root = (
            args.output_root.resolve()
            if args.output_root
            else resolve_project_path(confirmation_config["output_directory"])
        )
        try:
            result = run_confirmation(
                semantic_root,
                fusion_root,
                prior_reintegration_root,
                output_root,
                confirmation_config,
                config["network"],
                config["scheduler"],
                config["integration"],
                config["reintegration"],
                config["receiver"],
                config["cause"],
                config["recovery"],
                config["preprocessing"],
            )
        except RuntimeError as exc:
            raise SystemExit(f"independent-confirmation error: {exc}") from exc
        print(json.dumps(result, indent=2))
        return
    if args.json:
        print(json.dumps(config, indent=2, sort_keys=False))
    else:
        print(yaml.safe_dump(config, sort_keys=False))
