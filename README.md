# ReliefLens

ReliefLens is a local-first image triage prototype built around Apple's low-cost
`MobileCLIP2-S0` model. It turns a folder of field photos into a ranked CSV,
JSONL, and HTML dashboard showing likely needs, hazards, damage, supplies, and
follow-up actions.

The core idea: in floods, fires, storms, housing inspections, mutual-aid intake,
or community health work, people often have messy photo dumps before they have
clean forms. ReliefLens uses zero-shot image/text matching to make those photos
searchable and actionable without sending them to a hosted vision API.

## Why this model

`apple/MobileCLIP2-S0` is small, current, and underused. Apple lists it as
`11.4M` image parameters plus `63.4M` text parameters, which makes it a good fit
for high-volume local transformation work.

## What it produces

- `triage.csv`: one row per image with top category, confidence, action, and
  per-category scores.
- `triage.jsonl`: machine-readable records for downstream workflows.
- `dashboard.html`: a static, shareable review page with severity counts,
  category filters, search, and top-match evidence.
- `embeddings.npz`: optional cached image embeddings for re-labeling without
  re-encoding images.

## Install

Create a Python environment, then install:

```powershell
pip install -r requirements.txt
```

The model card for MobileCLIP2 says to install Apple's `ml-mobileclip` package.
If it is not available through pip in your environment, install from Apple's
GitHub repo, then rerun ReliefLens:

```powershell
pip install git+https://github.com/apple/ml-mobileclip.git
```

## Run

```powershell
python relieflens.py scan `
  --images C:\path\to\photos `
  --out C:\path\to\relieflens-output `
  --model apple/MobileCLIP2-S0
```

For a faster dry run that validates parsing and output generation without
loading the model:

```powershell
python relieflens.py demo --out C:\path\to\relieflens-demo
```

## Customize the mission

Edit `taxonomy.yaml`. Each category has prompts and an action. ReliefLens embeds
the prompts, compares them to each image, and reports the best match.

Good taxonomies are concrete:

- Prefer `a washed out road or bridge` over `infrastructure`.
- Prefer `standing flood water inside a home` over `water issue`.
- Add local needs: insulin, oxygen tanks, wheelchair ramps, school meals,
  damaged documents, mold, downed power lines, blocked exits.

## Validate

```powershell
python -m unittest discover -s tests
python relieflens.py demo --out relieflens-output
```

Open `relieflens-output/dashboard.html` to review the static dashboard.

`github-actions-test.example.yml` contains a ready-to-copy GitHub Actions
workflow. It is kept as an example because some OAuth tokens cannot create
active workflow files without the extra `workflow` scope.

## Safety notes

ReliefLens is a triage aid, not an emergency decision-maker. Treat its output as
an attention-routing layer. Human review should decide actual priority,
especially for medical, legal, housing, insurance, or emergency response use.

See `IMPACT.md` for deployment ideas and the roadmap.
