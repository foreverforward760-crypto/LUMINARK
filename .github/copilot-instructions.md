<!-- Copilot/AI contributor guidance for the LUMINARK AI codebase -->
# LUMINARK AI — Copilot Instructions

Purpose: give AI coding agents the minimal, actionable context to be immediately productive.

- **Big picture**: LUMINARK is a small, modular Python system composed of four conceptual subsystems:
  - Brain: [LUMINARK_AI/nam_logic.py](LUMINARK_AI/nam_logic.py) — `NAMLogicEngine.assess_state` maps inputs to an 0–9 Gate + micro-stage.
  - Heart: [LUMINARK_AI/yunus_safety.py](LUMINARK_AI/yunus_safety.py) — `YunusProtocol.check_humility` enforces safety for Stage 8 (Unity/Trap).
  - Body: [LUMINARK_AI/bio_defense.py](LUMINARK_AI/bio_defense.py) — `BioDefenseSystem.scan_threats` computes threat strings using RISS logic.
  - Pulse: [LUMINARK_AI/physics_engine.py](LUMINARK_AI/physics_engine.py) — `PhysicsEngine.calculate_momentum` computes velocity from state.

- **Primary integration point**: [LUMINARK_AI/main_model.py](LUMINARK_AI/main_model.py) — `LuminarkAI.process_cycle` calls the four subsystems in this order: Brain → Pulse → Body → Heart → Output.

- **Data shapes & invariants**:
  - Input dicts use numeric keys `complexity` and `stability` with expected ranges 0–100.
  - `NAMLogicEngine.assess_state` returns a `ConsciousnessState` dataclass with fields: `gate` (Gate enum), `micro` (0–9), `integrity` (stability), `tension` (100 - stability), `description`.
  - Gate enum uses integer values 0..9. Stage 8 (`UNITY_TRAP`) is treated specially by the safety layer.

- **Project conventions** (follow these when editing/adding code):
  - Keep module-level responsibilities separated (one conceptual subsystem per file).
  - Use dataclasses for state objects (see `ConsciousnessState`).
  - Preserve the existing printed logging style (emoji-prefixed lines), since tests/examples parse/expect these human-readable outputs.
  - No external dependencies — rely on the standard library only.

- **Safety & behavioral rules**:
  - Do not suppress or rename the `Gate.UNITY_TRAP` value; it's the trigger for stricter checks in `YunusProtocol`.
  - `YunusProtocol.arrogance_markers` is the safety list scanned at Stage 8 — changes to it alter core safety behavior.

- **Common tasks / commands**:
  - Restore the project scaffolding (recreates the `LUMINARK_AI` module files):

```bash
python install_luminark.py
```

  - Run the model (from the repository root):

```bash
cd LUMINARK_AI && python main_model.py
```

- **Quick code examples** (use these when writing tests or reproducing behavior):

```python
from main_model import LuminarkAI
ai = LuminarkAI()
ai.process_cycle({'complexity': 40, 'stability': 80}, 'Normal Growth')
```

- **Where to look for changes that affect system behavior**:
  - Stage mapping and micro-stage math: [LUMINARK_AI/nam_logic.py](LUMINARK_AI/nam_logic.py)
  - Output gating/quarantine rules: [LUMINARK_AI/yunus_safety.py](LUMINARK_AI/yunus_safety.py)
  - Threat scoring heuristics: [LUMINARK_AI/bio_defense.py](LUMINARK_AI/bio_defense.py)
  - Time/damping constants affecting velocity: [LUMINARK_AI/physics_engine.py](LUMINARK_AI/physics_engine.py)

- **Notes for AI code edits (do not guess domain intent)**:
  - Preserve numeric ranges and mapping formulas unless the user requests a change and explains the intended effect.
  - When adding new gates or renumbering stages, update all modules that reference `Gate` and run the main scenarios to verify behavior.

If anything here is unclear or you'd like the instructions to include more examples (unit tests, sample edits, or CI commands), tell me what to expand. — Copilot

**Tests & CI**

- **Smoke test**: a minimal integration test lives at `tests/test_smoke.py`. It instantiates `LuminarkAI` and runs a single `process_cycle` to ensure modules import and execute without errors.
- Run locally:

```bash
python -m unittest discover -v
```

- **GitHub Actions**: a CI workflow runs the same unittest discovery on push and PR. See `.github/workflows/ci.yml`.

- **When editing behavior**: after changing `Gate` numbering, `YunusProtocol.arrogance_markers`, or the mapping formula in `NAMLogicEngine.assess_state`, run the smoke test to catch import/runtime mismatches.
