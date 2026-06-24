# Apple MobileCLIP Connection

ReliefLens is designed as a downstream demonstration of Apple's
`apple/MobileCLIP2-S0` model: a small image-text model that can turn large image
folders into searchable, local review queues.

## Model anchor

- Hugging Face model: <https://huggingface.co/apple/MobileCLIP2-S0>
- Official code: <https://github.com/apple/ml-mobileclip>
- Model family: MobileCLIP2
- Default checkpoint filename: `mobileclip2_s0.pt`
- Listed size on the Apple model card: `11.4M` image parameters plus `63.4M`
  text parameters

ReliefLens defaults to this model because it sits in a rare sweet spot: useful
zero-shot vision-language behavior, low local inference cost, and enough speed
for high-volume field photo review.

## Upstream contribution

To make this useful beyond this repository, I opened an upstream contribution to
Apple's official implementation:

<https://github.com/apple/ml-mobileclip/pull/11>

That PR adds a small folder-level zero-shot triage example to
`apple/ml-mobileclip`. The example uses MobileCLIP2-S0 by default, scores every
image in a folder against text labels, and writes a CSV review queue.

## Why this matters for Apple

MobileCLIP is strongest when its speed changes the shape of a workflow. ReliefLens
is a concrete example:

- Run locally instead of sending sensitive field photos to hosted APIs.
- Iterate on text labels without retraining.
- Convert unstructured images into queues that humans can review.
- Use a small Apple model for public-interest workflows where cost and privacy
  matter.

## Compatibility target

ReliefLens follows the OpenCLIP-style MobileCLIP2 inference path documented by
Apple:

```python
model, _, preprocess = open_clip.create_model_and_transforms(
    "MobileCLIP2-S0",
    pretrained="/path/to/mobileclip2_s0.pt",
)
tokenizer = open_clip.get_tokenizer("MobileCLIP2-S0")
```

The current demo and tests validate the local artifact pipeline without loading
the model. Full inference requires installing the Apple/OpenCLIP stack described
in `README.md`.
