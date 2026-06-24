# ReliefLens Impact Playbook

ReliefLens is designed for teams that have images before they have clean data.
The first target is disaster and housing recovery, but the same workflow applies
to mutual aid, accessibility audits, supply distribution, environmental cleanup,
and public-interest field documentation.

## The transformation

Most field workflows lose time in the handoff between "someone took a photo"
and "someone knows what to do next." ReliefLens makes that handoff concrete:

1. Collect images from phones, drones, case workers, volunteers, or residents.
2. Run local triage on a laptop or workstation.
3. Review the dashboard by severity and category.
4. Export CSV or JSONL into dispatch, casework, GIS, or inventory systems.
5. Update the taxonomy as the mission changes.

The point is not to replace human judgment. The point is to get the right human
looking at the right image sooner.

## High-leverage deployments

- Flood recovery: rank standing water, mold risk, damaged belongings, blocked
  roads, and shelter damage from resident photo submissions.
- Accessibility response: find mobility aids, blocked entrances, broken ramps,
  medical equipment, and transport barriers from inspection photos.
- Supply networks: separate images of needs from images of available supplies,
  then route water, food, diapers, hygiene kits, blankets, or pet supplies.
- Housing casework: surface documents, damaged rooms, unsafe exits, and repair
  evidence before an intake worker opens every image manually.
- Community audits: scan sidewalk, transit stop, school, clinic, or shelter
  photos for visible barriers and follow-up needs.

## Why local-first matters

Field photos can contain homes, documents, faces, children, medical devices, and
location clues. A local-first workflow keeps raw images on controlled machines
and avoids per-image API billing while teams experiment with prompts, categories,
and review policy.

## What to add next

- Geotag extraction and map output when EXIF GPS is present.
- Human review states: confirmed, rejected, needs context, dispatched.
- Active learning: turn reviewer corrections into a small adapter or calibrated
  classifier.
- Duplicate and near-duplicate grouping from cached embeddings.
- Redaction pass for faces, IDs, addresses, and documents before sharing.
- Direct exports for Airtable, Google Sheets, ArcGIS, QGIS, and 211/CRM tools.
