# Research writing, evidence, and publication plan

Status: an outline and evidence program, not a submitted manuscript. The
[implemented fusion objective](fusion-objective.md) and
[Pilot07 controlled study](studies/pilot-07-fusion-transfer.md) provide current method
and experiment material; their qualification limits still apply.
Begin writing while implementing; reserve claims and result prose until the
corresponding experiments have passed their registered gates.

The current project requires reconstruction training without noncommercial
pretrained weights. Paper tables must distinguish entirely random initialization,
MIT V-JEPA initialization with encoder adaptation, frozen-encoder controls, and
the historical NC Gekko reference. Pilot 04's quality result is not evidence that
the project learned reconstruction from scratch. Record encoder and decoder
initialization separately, including teachers, optimizer schedules, and any
warm start. Pilot 05 implements the first end-to-end path; its short synthetic
study is an engineering experiment, not a matched pretraining comparison.

The September 29 primary-task amendment is fixed V-JEPA latent prediction; see
the [latent protocol and implementation](latent-pipeline.md). Method and result
tables must identify the prediction space, fixed versus momentum teacher,
teacher/student licenses, and which encoder blocks train. Show latent maps with
a shared teacher-fitted PCA basis and color scale, label them as feature maps,
and retain RGB failures as separate results. A smaller latent MSE is not evidence
of improved RGB sharpness. Geometric co-visibility, same-room spatial controls,
and downstream matching remain necessary to substantiate the multi-view claim.

The [SOTA evidence program](sota-evidence.md) and
[machine-readable benchmark registry](../configs/train/benchmark-registry.toml)
separate the local synthetic patch-grid diagnostic from external HPatches-240
and future ETH3D protocol parity. HPatches' primary subset is 59 viewpoint
sequences / 295 pairs, matching the official ZeroCo pair list; the other 57
illumination sequences are supplementary. The local hard-match readout and
256px inputs differ from published refinement recipes. Do not insert these
numbers into a published-results comparison as if the protocols matched.

The controlled latent continuation should support four artifact-backed tables:
common-mask latent utility and co-visibility; synthetic correspondence with
paired room intervals; external correspondence with paired sequence intervals;
and native memory/throughput with failures included. Report the mask arms'
different adaptive unfreezing durations, the compact-arm checkpoint recovery,
and the exact optimizer continuation separately. A fixed-teacher MSE objective
is an adaptation experiment, not reproduction of V-JEPA 2.1 pretraining.

## 1. Positioning and scope

Working title: **Sparse Per-View Video Representations for Self-Supervised
Multi-View Fusion**. Keep the title provisional until the strongest supported
contribution is known. Identify the system as `burn_gekko`, explicitly derived
from the Gekko/CroCo research direction, without implying it is the authors'
official implementation.

Possible contributions, all conditional on evidence:

1. A reference-set fusion objective connecting pretrained per-view representations
   to pairwise and set-level reconstruction utility.
2. A measured quality/compute frontier for sparse contextual image encoding,
   including an operationally useful token-budget policy if one is demonstrated.
3. A reproducible procedural-room evaluation and generation protocol that separates
   RGB-only training, geometry-assisted curation, and supervised evaluation.

The Rust implementation is an engineering contribution and reproducibility
artifact. A new framework implementation does not itself establish a novel
learning method. A benchmark paper is viable if the most useful result is a
careful characterization or negative finding rather than a new state of the art.

## 2. Related-work review tasks

Read and maintain a comparison ledger for:

| Family | Starting sources | Questions to answer |
| --- | --- | --- |
| Cross-view SSL | [CroCo](https://github.com/naver/croco), [Gekko](https://arxiv.org/abs/2609.01530) | What objective and architecture components are inherited? |
| Multi-view masked modeling | [MuM](https://arxiv.org/abs/2511.17309), [Muskie](https://arxiv.org/abs/2511.18115) | Which set architectures/objectives already exist? |
| Dense pretrained features | [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) | What role do native image modality and sparse observations play? |
| Geometric foundation models | DUSt3R/MASt3R/VGGT family; verify current official artifacts | Which comparisons require supervised geometry or much larger pretraining? |
| Visibility and matching | Gekko-related co-visibility and learned matching work | What does visibility mean, and how is it evaluated? |
| Sparse and budgeted inference | Token selection, token pruning, multi-view routing | Is the saving in encoder computation, fusion, or only output storage? |
| Synthetic indoor data | Zeroverse and [Hypersim](https://github.com/apple-aiml-research/ml-hypersim) | How are scene diversity, label quality and transfer established? |

Refresh the review before submission. For each source, record version/date,
input modality, pretrained data, supervision, set support, sparsity definition,
released artifacts, compute and evaluation protocols. Distinguish an idea
supported by a source from a new proposal in this project.

## 3. Paper structure

| Section | Content to prepare | Evidence required before finalizing |
| --- | --- | --- |
| Abstract | Problem, evaluated method, supported result, practical scope | Primary completed results and limitations |
| Introduction | Why multi-view utility under a token budget matters | Concrete failure cases and related-work positioning |
| Related work | Objective, representation and multi-view/sparse distinctions | Verified primary-source ledger |
| Method | Per-view encoder, masks, shared decoder, pair/set RI, sparse execution | Equations, information-flow tests, architecture/config artifacts |
| Data | Published generator versions, distributions, calibration and split rules | Dataset qualification and distribution reports |
| Experimental setup | All controls, compute budgets, readouts, seeds and supervision regimes | Frozen protocols and complete run ledger |
| Main results | Pair objective, set extension, sparse quality/cost frontier | Confirmatory H1–H3 comparisons |
| Transfer and analysis | Real-domain result, texture/overlap behavior, calibration, failures | E2–E5 and registered diagnostic suites |
| Limitations | Proxy ambiguity, synthetic gap, pretrained-data uncertainty, compute, scope | Retained failures and measured boundaries |
| Conclusion | Only supported contribution and realistic use cases | Claim/evidence audit |
| Supplement | Full configs, geometry derivation, numerical parity, ablations, seeds | Reproducible artifact references |

Write the method and dataset sections at P2/P3, before results can influence
their description. Keep planned ablations in the internal protocol rather than
presenting them as completed in the manuscript.

## 4. Figures and tables to build

| Artifact | Intended content | Generation source |
| --- | --- | --- |
| Figure 1 | Per-view V-JEPA, three training paths, reference-set extension | Editable vector diagram from final implementation |
| Figure 2 | Dataset examples and distributions across independent rooms | Deterministic scene/sample selection and distribution report |
| Figure 3 | RGB, GT visibility, raw RI, predicted RI and sparse mask | Fixed held-out pairs, including failures |
| Figure 4 | Quality versus measured latency/VRAM at multiple budgets | Full pipeline benchmark and metric artifacts |
| Figure 5 | Overlap/texture/view-count behavior and calibration | Per-scene/per-stratum evaluation records |
| Table 1 | Matched pair-objective results and total training cost | B0–B3, all confirmatory seeds |
| Table 2 | Joint sets versus pair pooling at fixed tokens and compute | M0–M2 protocols |
| Table 3 | Dense/gather/sparse implementations and mixed-budget training | SP0–SP3 protocols |
| Table 4 | External synthetic and real transfer, frozen probes | E3–E5; supervision labeled per row |
| Supplement tables | All seeds, ablations, failed screens, protocols and resources | Complete experiment ledger |

Use vector figures and programmatically generated tables. An attention map may
illustrate behavior, but must not be labeled as ground-truth correspondence.
Qualitative panels need a registered sampling rule, useful captions, consistent
scales, and visible failures. Attractive examples selected after inspection can
appear as illustrations only with selection disclosed.

## 5. Claim-to-evidence ledger

Create `paper/claims.csv` at implementation time with columns:

```text
claim_id, manuscript_location, wording, hypothesis, primary_metric,
baseline_ids, run_ids, protocol_ids, dataset_ids, uncertainty,
counterevidence, scope_limit, status
```

Examples of allowed transitions:

| Draft claim | Required evidence | Wording if evidence fails |
| --- | --- | --- |
| RI improves fusion | Paired AEPE/probe improvement with controlled objectives | RI predicts utility but did not improve this representation under the tested budget |
| Joint multi-view fusion helps | Gain over continued pair pooling at fixed total tokens | Extra views helped only when observation/compute budget increased |
| Sparse encoding accelerates inference | End-to-end timings including encoder and routing at matched quality | Sparse fusion reduced attention size without a demonstrated pipeline speedup |
| Co-visibility prediction generalizes | Untuned held-out real-domain AP/coverage and failure strata | The result is confined to procedural-room distributions |
| Training is self-supervised | Loss/sampling provenance and no GT-gradient inputs | RGB losses used geometry-assisted pair curation or label-assisted selection |

Report negative and incomplete studies honestly. If a result was obtained after
the originally registered budget, it is a follow-up, not the original controlled
comparison. Never replace missing test cells with forecasts or preliminary values.

## 6. Writing milestones

| Milestone | Writing/artifact work |
| --- | --- |
| P0 | Hypotheses, protocol ledger, bibliography skeleton and contribution alternatives |
| P1 | Encoder provenance, conversion, parity appendix and model-card skeleton |
| P2 | Dataset section, geometry appendix, distribution figures and data card |
| P3 | Method equations, information-flow diagram and implementation description |
| P4 | Pair-objective results draft, learning curves and first failure analysis |
| P5/P6 | Multi-view and sparsity figures; update contribution wording to match results |
| P7 | Transfer/probe analysis and limitations |
| P8 | Lock table-generation inputs and run a claim/evidence audit |
| P9 | Full manuscript revision, reproducibility dry run and release package |

After P7, choose between a methods paper, a systems/reproducibility paper, or a
focused negative-result study. Select a venue based on actual findings and then
verify its current formatting, anonymity, page limit, artifact and deadline
requirements. Do not plan against an unverified deadline today.

## 7. Reproducibility and release package

The minimum artifact includes:

- Pinned source/import manifests, dependency locks, encoder conversion recipe,
  architecture configs, and a small executable correctness fixture.
- Published-package capture recipe, scene seed/split manifests, data schema,
  label-generation protocol, a small redistributable sample set and dataset card.
- Model weights where distribution is permitted, plus their hash, expected
  input normalization, image/video mode, feature outputs and source weight terms.
- Completed run registry, per-scene metrics, statistical analysis code, table
  and figure generation, measured compute, and lists of incomplete/failed runs.
- A single-GPU reproduction path with expected resource bounds and a shorter
  smoke path explicitly labeled as a functional check rather than a replication.
- Documented limitations for geometric visibility versus RGB utility, sparse
  budgets, domain shift, dynamic scenes and optional calibration/probe fitting.

Track licenses separately for imported encoder code, original weights, any
Gekko/CroCo-derived code, synthetic generator components and external datasets.
The [Gekko repository's noncommercial/share-alike terms](https://github.com/thibautloiseau/gekko/blob/63f0ec9957885ea82cc2f4637d502003fbf9afcb/LICENSE)
must not be replaced by the encoder crate's license when reusing its artifacts.
Prefer an independently authored Burn method implementation and document all
actual reuse. Resolve artifact-specific distribution questions before release.

Synthetic rooms reduce dependence on newly captured personal imagery, but the
pretrained encoder and external evaluation datasets retain their own provenance.
Discuss scene and population coverage, stylized people, material/lighting bias,
camera priors, potential surveillance use, and the compute spent on generation
and failed experiments in proportion to what was actually studied.

Release, submission and publication remain separate future work. The project
now includes native training, controlled pilot artifacts and PDF reports;
the initial plan-only status no longer describes the implementation. A paper
must use the completed artifacts and current claim boundaries, rather than
turning this outline's proposed experiments into reported results.
