"""Stable, allowlisted command entry point for the research workflow."""
import runpy
import sys

COMMANDS={
    "pilot":"experiments.run",
    "pilot-report":"experiments.analyze",
    "deployment":"experiments.deployment",
    "deployment-report":"experiments.analyze_deployment",
    "dense":"experiments.dense_control",
    "cycle3":"experiments.cycle3",
    "cycle3-report":"experiments.analyze_cycle3",
    "verify-cycle3":"experiments.verify_cycle3",
    "cycle4":"experiments.cycle4",
    "cycle4-report":"experiments.analyze_cycle4",
    "verify-cycle4":"experiments.verify_cycle4",
    "verify-followup":"experiments.verify_followup",
    "draft":"experiments.build_paper",
}

if len(sys.argv)<2 or sys.argv[1] not in COMMANDS:
    print("Usage: python -m experiments <command> [arguments]\nCommands: "+", ".join(COMMANDS))
    raise SystemExit(0 if len(sys.argv)<2 or sys.argv[1] in ("-h","--help") else 2)
module=COMMANDS[sys.argv[1]]
sys.argv=[module,*sys.argv[2:]]
runpy.run_module(module,run_name="__main__")
