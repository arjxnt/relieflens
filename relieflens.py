from __future__ import annotations

import argparse
import csv
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from PIL import ImageDraw

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Category:
    id: str
    label: str
    severity: str
    action: str
    prompts: list[str]


@dataclass(frozen=True)
class ImageResult:
    path: Path
    top_id: str
    top_label: str
    severity: str
    action: str
    confidence: float
    scores: dict[str, float]
    top_matches: list[dict[str, object]]


def load_taxonomy(path: Path) -> list[Category]:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if yaml is not None else parse_simple_taxonomy(text)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid taxonomy in {path}")
    categories = []
    seen_ids: set[str] = set()
    for raw in data.get("categories", []):
        category_id = str(raw["id"])
        if category_id in seen_ids:
            raise ValueError(f"Duplicate category id {category_id!r} in {path}")
        seen_ids.add(category_id)
        categories.append(
            Category(
                id=category_id,
                label=str(raw["label"]),
                severity=str(raw.get("severity", "medium")),
                action=str(raw.get("action", "")),
                prompts=[str(prompt) for prompt in raw.get("prompts", [])],
            )
        )
    if not categories:
        raise ValueError(f"No categories found in {path}")
    for category in categories:
        if not category.prompts:
            raise ValueError(f"Category {category.id!r} has no prompts")
    return categories


def parse_simple_taxonomy(text: str) -> dict[str, list[dict[str, object]]]:
    categories: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    active_list: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped == "categories:":
            continue

        if stripped.startswith("- id:"):
            current = {"id": stripped.split(":", 1)[1].strip(), "prompts": []}
            categories.append(current)
            active_list = None
            continue

        if current is None:
            continue

        if stripped in {"prompts:"}:
            active_list = "prompts"
            continue

        if active_list == "prompts" and stripped.startswith("- "):
            current["prompts"].append(unquote(stripped[2:].strip()))
            continue

        if ":" in stripped:
            key, value = stripped.split(":", 1)
            current[key.strip()] = unquote(value.strip())
            active_list = None

    return {"categories": categories}


def unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def softmax(values: np.ndarray, temperature: float = 0.025) -> np.ndarray:
    scaled = values / temperature
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    return exp / np.sum(exp)


class MobileClipScorer:
    def __init__(self, model_id: str, checkpoint: Path | None, device: str):
        import torch
        import open_clip
        from huggingface_hub import hf_hub_download

        self.torch = torch
        self.open_clip = open_clip
        self.device = self._resolve_device(device)
        model_name = model_id.split("/")[-1]

        if checkpoint is None:
            filename = self._checkpoint_filename(model_name)
            checkpoint = Path(hf_hub_download(repo_id=model_id, filename=filename))

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=str(checkpoint),
            **self._model_kwargs(model_name),
        )
        model.eval()

        try:
            from mobileclip.modules.common.mobileone import reparameterize_model

            model = reparameterize_model(model)
        except Exception:
            # Reparameterization is recommended for export and speed. Some package
            # builds omit it, so inference still works without hard failing here.
            pass

        self.model = model.to(self.device)
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @staticmethod
    def _model_kwargs(model_name: str) -> dict[str, tuple[int, int, int]]:
        if model_name in {"MobileCLIP2-S3", "MobileCLIP2-S4"} or model_name.endswith("L-14"):
            return {}
        return {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if self.torch.cuda.is_available():
            return "cuda"
        if hasattr(self.torch.backends, "mps") and self.torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _checkpoint_filename(model_name: str) -> str:
        known = {
            "MobileCLIP2-S0": "mobileclip2_s0.pt",
            "MobileCLIP2-S2": "mobileclip2_s2.pt",
            "MobileCLIP2-S3": "mobileclip2_s3.pt",
            "MobileCLIP2-S4": "mobileclip2_s4.pt",
            "MobileCLIP-S0": "mobileclip_s0.pt",
            "MobileCLIP-S1": "mobileclip_s1.pt",
            "MobileCLIP-S2": "mobileclip_s2.pt",
            "MobileCLIP-S3": "mobileclip_s3.pt",
            "MobileCLIP-S4": "mobileclip_s4.pt",
        }
        if model_name not in known:
            raise ValueError(
                f"Don't know checkpoint filename for {model_name}. "
                "Pass --checkpoint explicitly."
            )
        return known[model_name]

    def encode_text(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vectors = []
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                tokens = self.tokenizer(batch).to(self.device)
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                vectors.append(features.detach().cpu().float().numpy())
        return np.vstack(vectors)

    def encode_images(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        vectors = []
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        with self.torch.no_grad():
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start : start + batch_size]
                images = []
                for path in batch_paths:
                    with Image.open(path) as image:
                        images.append(self.preprocess(image.convert("RGB")))
                tensor = self.torch.stack(images).to(self.device)
                features = self.model.encode_image(tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                vectors.append(features.detach().cpu().float().numpy())
        return np.vstack(vectors)


def category_prompt_embeddings(
    scorer: MobileClipScorer, categories: list[Category]
) -> tuple[list[str], np.ndarray]:
    prompts = []
    prompt_to_category = []
    for category in categories:
        for prompt in category.prompts:
            prompts.append(prompt)
            prompt_to_category.append(category.id)

    prompt_embeddings = scorer.encode_text(prompts)
    category_vectors = []
    ids = []
    for category in categories:
        indexes = [i for i, category_id in enumerate(prompt_to_category) if category_id == category.id]
        vector = normalize(prompt_embeddings[indexes]).mean(axis=0, keepdims=True)
        category_vectors.append(normalize(vector)[0])
        ids.append(category.id)
    return ids, np.vstack(category_vectors)


def score_images(
    paths: list[Path],
    image_vectors: np.ndarray,
    category_ids: list[str],
    category_vectors: np.ndarray,
    categories: list[Category],
) -> list[ImageResult]:
    if len(paths) != len(image_vectors):
        raise ValueError("paths and image_vectors must have the same length")
    if len(category_ids) != len(category_vectors):
        raise ValueError("category_ids and category_vectors must have the same length")
    if image_vectors.ndim != 2 or category_vectors.ndim != 2:
        raise ValueError("image_vectors and category_vectors must be 2D arrays")
    if image_vectors.shape[1] != category_vectors.shape[1]:
        raise ValueError("image and category vectors must have the same embedding dimension")

    category_by_id = {category.id: category for category in categories}
    ordered_categories = [category_by_id[category_id] for category_id in category_ids]
    similarity = normalize(image_vectors) @ normalize(category_vectors).T
    results = []

    for path, row in zip(paths, similarity):
        probabilities = softmax(row)
        best_index = int(np.argmax(probabilities))
        top_id = category_ids[best_index]
        top_category = category_by_id[top_id]
        top_indexes = np.argsort(probabilities)[::-1][:3]
        top_matches = [
            {
                "id": ordered_categories[index].id,
                "label": ordered_categories[index].label,
                "severity": ordered_categories[index].severity,
                "score": round(float(probabilities[index]), 6),
            }
            for index in top_indexes
        ]
        scores = {
            category_id: round(float(probabilities[index]), 6)
            for index, category_id in enumerate(category_ids)
        }
        results.append(
            ImageResult(
                path=path,
                top_id=top_id,
                top_label=top_category.label,
                severity=top_category.severity,
                action=top_category.action,
                confidence=round(float(probabilities[best_index]), 6),
                scores=scores,
                top_matches=top_matches,
            )
        )
    return results


def write_csv(results: list[ImageResult], categories: list[Category], out_path: Path) -> None:
    score_fields = unique_score_fields([category.id for category in categories])
    fieldnames = ["image", "top_id", "top_label", "severity", "confidence", "action", "top_matches"]
    fieldnames.extend(score_fields.values())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "image": str(result.path),
                "top_id": result.top_id,
                "top_label": result.top_label,
                "severity": result.severity,
                "confidence": result.confidence,
                "action": result.action,
                "top_matches": json.dumps(result.top_matches, ensure_ascii=True),
            }
            for category in categories:
                row[score_fields[category.id]] = result.scores[category.id]
            writer.writerow(row)


def write_jsonl(results: list[ImageResult], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    {
                        "image": str(result.path),
                        "top_id": result.top_id,
                        "top_label": result.top_label,
                        "severity": result.severity,
                        "confidence": result.confidence,
                        "action": result.action,
                        "top_matches": result.top_matches,
                        "scores": result.scores,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )


def write_html(results: list[ImageResult], out_path: Path) -> None:
    out_dir = out_path.parent.resolve()
    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    grouped: dict[str, list[ImageResult]] = {}
    ordered_results = sorted(
        results,
        key=lambda item: (severity_rank.get(item.severity, 2), -item.confidence, item.top_label),
    )
    for result in ordered_results:
        grouped.setdefault(result.top_label, []).append(result)

    counts_by_severity: dict[str, int] = {}
    for result in results:
        counts_by_severity[result.severity] = counts_by_severity.get(result.severity, 0) + 1

    stat_cards = "".join(
        f"""
        <div class="stat severity-{html.escape(severity)}">
          <span>{html.escape(severity.title())}</span>
          <strong>{count}</strong>
        </div>
        """
        for severity, count in sorted(
            counts_by_severity.items(),
            key=lambda item: severity_rank.get(item[0], 2),
        )
    )

    category_buttons = "".join(
        f'<button type="button" data-category="{html.escape(label)}">{html.escape(label)} <span>{len(items)}</span></button>'
        for label, items in grouped.items()
    )

    sections = []
    for label, items in grouped.items():
        cards = []
        for item in items:
            image_src = image_source(item.path, out_dir)
            matches = "".join(
                f"<li><span>{html.escape(str(match['label']))}</span><b>{float(match['score']):.3f}</b></li>"
                for match in item.top_matches
            )
            cards.append(
                f"""
                <article class="card severity-{html.escape(item.severity)}" data-category="{html.escape(item.top_label)}" data-severity="{html.escape(item.severity)}" data-search="{html.escape((str(item.path) + ' ' + item.top_label + ' ' + item.action).lower())}">
                  <img src="{image_src}" alt="">
                  <div class="body">
                    <div class="meta">
                      <span>{html.escape(item.top_label)}</span>
                      <b>{item.confidence:.3f}</b>
                    </div>
                    <div class="badge">{html.escape(item.severity.title())}</div>
                    <div class="path">{html.escape(str(item.path))}</div>
                    <p>{html.escape(item.action)}</p>
                    <ol>{matches}</ol>
                  </div>
                </article>
                """
            )
        sections.append(
            f"""
            <section>
              <h2>{html.escape(label)} <span>{len(items)}</span></h2>
              <div class="grid">{''.join(cards)}</div>
            </section>
            """
        )

    out_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ReliefLens Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7f4;
      color: #1d241f;
    }}
    body {{ margin: 0; }}
    header {{
      padding: 32px clamp(18px, 4vw, 52px);
      background: linear-gradient(135deg, #163326, #315f43);
      color: white;
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(28px, 4vw, 44px); letter-spacing: 0; }}
    header p {{ margin: 0; max-width: 760px; color: #dbe8df; line-height: 1.5; }}
    main {{ padding: 24px clamp(18px, 4vw, 48px) 48px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 18px; }}
    .stat {{ background: white; border: 1px solid #d9ded5; border-left: 6px solid #7b897e; border-radius: 8px; padding: 12px; }}
    .stat span {{ display: block; color: #5d665f; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .stat strong {{ font-size: 30px; }}
    .severity-critical {{ border-left-color: #b3261e; }}
    .severity-high {{ border-left-color: #c46a19; }}
    .severity-medium {{ border-left-color: #9b7a16; }}
    .severity-low {{ border-left-color: #4f7f61; }}
    .toolbar {{ display: grid; grid-template-columns: minmax(180px, 1fr) auto; gap: 12px; align-items: start; }}
    input {{ width: 100%; box-sizing: border-box; border: 1px solid #bfc8bd; border-radius: 8px; padding: 12px 14px; font: inherit; background: white; }}
    .filters {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    button {{ border: 1px solid #bfc8bd; background: white; border-radius: 999px; padding: 9px 12px; font: inherit; cursor: pointer; }}
    button.active {{ background: #214432; color: white; border-color: #214432; }}
    button span {{ opacity: 0.75; }}
    section {{ margin-top: 28px; }}
    h2 {{ font-size: 20px; margin: 0 0 12px; display: flex; gap: 10px; align-items: center; }}
    h2 span {{ font-size: 13px; padding: 2px 8px; border-radius: 999px; background: #dfe8da; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }}
    .card {{
      background: white;
      border: 1px solid #d9ded5;
      border-left-width: 6px;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 2px rgba(20, 30, 20, 0.06);
    }}
    img {{ width: 100%; aspect-ratio: 4 / 3; object-fit: cover; background: #e5e8e1; display: block; }}
    .body {{ padding: 12px; }}
    .meta {{ font-weight: 700; margin-bottom: 6px; display: flex; justify-content: space-between; gap: 8px; }}
    .badge {{ display: inline-block; font-size: 12px; padding: 3px 8px; border-radius: 999px; background: #edf2eb; margin-bottom: 8px; }}
    .path {{ color: #5d665f; font-size: 12px; overflow-wrap: anywhere; margin-bottom: 8px; }}
    p {{ margin: 0; line-height: 1.4; }}
    ol {{ margin: 12px 0 0; padding: 0; list-style: none; display: grid; gap: 4px; }}
    li {{ display: flex; justify-content: space-between; gap: 8px; font-size: 12px; color: #48524a; }}
    .hidden {{ display: none; }}
    @media (max-width: 720px) {{ .toolbar {{ grid-template-columns: 1fr; }} .filters {{ justify-content: flex-start; }} }}
  </style>
</head>
<body>
  <header>
    <h1>ReliefLens Dashboard</h1>
    <p>Local zero-shot image triage for routing urgent needs, hazards, damage, supplies, and casework follow-up.</p>
  </header>
  <main>
    <div class="summary">
      <div class="stat"><span>Total images</span><strong>{len(results)}</strong></div>
      {stat_cards}
    </div>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Search paths, actions, and labels">
      <div class="filters">
        <button type="button" class="active" data-category="all">All <span>{len(results)}</span></button>
        {category_buttons}
      </div>
    </div>
    {''.join(sections)}
  </main>
  <script>
    const cards = [...document.querySelectorAll('.card')];
    const buttons = [...document.querySelectorAll('button[data-category]')];
    const search = document.querySelector('#search');
    let category = 'all';

    function update() {{
      const term = search.value.trim().toLowerCase();
      for (const card of cards) {{
        const matchesCategory = category === 'all' || card.dataset.category === category;
        const matchesSearch = !term || card.dataset.search.includes(term);
        card.classList.toggle('hidden', !(matchesCategory && matchesSearch));
      }}
    }}

    for (const button of buttons) {{
      button.addEventListener('click', () => {{
        category = button.dataset.category;
        buttons.forEach(item => item.classList.toggle('active', item === button));
        update();
      }});
    }}
    search.addEventListener('input', update);
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def image_source(path: Path, out_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return html.escape(resolved.relative_to(out_dir).as_posix(), quote=True)
    except ValueError:
        return html.escape(resolved.as_uri(), quote=True)


def unique_score_fields(ids: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    used: set[str] = set()
    for category_id in ids:
        base = "score_" + re.sub(r"[^a-zA-Z0-9_]+", "_", category_id).strip("_")
        if base == "score_":
            base = "score_category"
        field = base
        suffix = 2
        while field in used:
            field = f"{base}_{suffix}"
            suffix += 1
        fields[category_id] = field
        used.add(field)
    return fields


def run_scan(args: argparse.Namespace) -> None:
    image_root = Path(args.images)
    out_dir = Path(args.out)
    taxonomy_path = Path(args.taxonomy)
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = load_taxonomy(taxonomy_path)
    paths = list(iter_images(image_root))
    if not paths:
        raise SystemExit(f"No images found under {image_root}")

    scorer = MobileClipScorer(
        model_id=args.model,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        device=args.device,
    )
    category_ids, category_vectors = category_prompt_embeddings(scorer, categories)
    image_vectors = scorer.encode_images(paths, batch_size=args.batch_size)
    results = score_images(paths, image_vectors, category_ids, category_vectors, categories)

    np.savez_compressed(
        out_dir / "embeddings.npz",
        paths=np.array([str(path) for path in paths]),
        image_vectors=image_vectors,
        category_ids=np.array(category_ids),
        category_vectors=category_vectors,
    )
    write_csv(results, categories, out_dir / "triage.csv")
    write_jsonl(results, out_dir / "triage.jsonl")
    write_html(results, out_dir / "dashboard.html")
    print(f"Wrote {len(results)} triage records to {out_dir}")


def run_demo(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    categories = load_taxonomy(Path(args.taxonomy))

    demo_root = out_dir / "demo_images"
    demo_root.mkdir(exist_ok=True)
    samples = [
        ("flooded_room.jpg", (69, 116, 134), "Flooded room", "flood_or_water_damage"),
        ("supply_table.jpg", (194, 154, 74), "Supply table", "food_water_supplies"),
        ("blocked_road.jpg", (92, 96, 88), "Blocked road", "blocked_transport"),
    ]
    paths = []
    target_ids = []
    for filename, color, label, target_id in samples:
        path = demo_root / filename
        image = Image.new("RGB", (900, 620), color)
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 410, 860, 570), fill=(255, 255, 255))
        draw.text((72, 455), label, fill=(20, 30, 20))
        draw.text((72, 500), "Demo placeholder image", fill=(74, 84, 78))
        image.save(path, quality=90)
        paths.append(path)
        target_ids.append(target_id)

    category_ids = [category.id for category in categories]
    image_vectors = np.zeros((len(paths), len(category_ids)), dtype=np.float32)
    for row_index, target_id in enumerate(target_ids):
        image_vectors[row_index, category_ids.index(target_id)] = 1.0
    category_vectors = np.eye(len(category_ids), dtype=np.float32)
    results = score_images(paths, image_vectors, category_ids, category_vectors, categories)

    write_csv(results, categories, out_dir / "triage.csv")
    write_jsonl(results, out_dir / "triage.jsonl")
    write_html(results, out_dir / "dashboard.html")
    print(f"Wrote demo dashboard to {out_dir / 'dashboard.html'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local-first image triage with MobileCLIP.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--taxonomy", default=str(Path(__file__).with_name("taxonomy.yaml")))
    common.add_argument("--out", required=True)

    scan = subparsers.add_parser("scan", parents=[common])
    scan.add_argument("--images", required=True)
    scan.add_argument("--model", default="apple/MobileCLIP2-S0")
    scan.add_argument("--checkpoint")
    scan.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    scan.add_argument("--batch-size", type=positive_int, default=16)
    scan.set_defaults(func=run_scan)

    demo = subparsers.add_parser("demo", parents=[common])
    demo.set_defaults(func=run_demo)
    return parser


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
