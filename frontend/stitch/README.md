# Stitch design assets — Elixir Apple-Inspired Wine RAG

Reference exports from Google Stitch. **Not** served as the runtime app; implement the product UI in `../src/`.

## IDs

| Item | Value |
|------|--------|
| Project name | Elixir Apple-Inspired Wine RAG |
| Project ID | `15506381115647961256` |
| Design System ID | `asset-stub-assets_d83f0ee48bae4cbb92026a2df42d7be2` |

## Layout

```
stitch/
  design-system/<id>/   # Design System HTML + screenshots
  screens/<screen-id>/  # Per-screen HTML + images
  manual-export/        # Drop zone when exporting from Stitch web UI
  manifest.json         # id → local paths + fetch metadata
```

## Fetch methods

### Path A — SDK (preferred)

Requires `STITCH_API_KEY` in the **shell** or local `ELIXIR/.env` (gitignored). Never commit the key.

```bash
cd ELIXIR/frontend
npm install
npm run fetch:stitch
```

The script uses `@google/stitch-sdk`, then downloads hosted HTML/image URLs with `curl -L` (or `curl.exe -L` on Windows) into this folder and updates `manifest.json`.

### Path B — Manual export

1. In Stitch, export Design System + all screens (HTML + screenshots).
2. Place files under `manual-export/` **or** directly into `design-system/` and `screens/` matching the layout above.
3. Update `manifest.json` paths / status to `manual`.

## Status

See `manifest.json` — as of scaffold, automated fetch was **blocked** (no `STITCH_API_KEY`); assets pending.
