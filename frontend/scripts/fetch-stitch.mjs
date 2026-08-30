/**
 * One-off: fetch Google Stitch project assets into ../stitch/
 *
 * Requires STITCH_API_KEY in the environment (never commit the key).
 * Usage: npm run fetch:stitch
 */

import { spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const PROJECT_ID = "15506381115647961256";
const DESIGN_SYSTEM_ID = "asset-stub-assets_d83f0ee48bae4cbb92026a2df42d7be2";

const __dirname = dirname(fileURLToPath(import.meta.url));
const stitchRoot = join(__dirname, "..", "stitch");

function curlDownload(url, outPath) {
  mkdirSync(dirname(outPath), { recursive: true });
  const curl = process.platform === "win32" ? "curl.exe" : "curl";
  const result = spawnSync(curl, ["-L", "--fail", "-o", outPath, url], {
    encoding: "utf8",
  });
  if (result.status !== 0) {
    throw new Error(
      `curl failed for ${url}: ${result.stderr || result.stdout || result.status}`,
    );
  }
  return outPath;
}

function pickUrl(value) {
  if (!value) return null;
  if (typeof value === "string" && /^https?:\/\//i.test(value)) return value;
  if (typeof value === "object") {
    for (const key of ["url", "href", "downloadUrl", "htmlUrl", "imageUrl"]) {
      if (typeof value[key] === "string" && /^https?:\/\//i.test(value[key])) {
        return value[key];
      }
    }
  }
  return null;
}

async function main() {
  if (!process.env.STITCH_API_KEY) {
    console.error(
      "STITCH_API_KEY is not set. Export it in the shell (or load from local .env) and retry.",
    );
    console.error(
      "Or use Path B: drop exports into stitch/manual-export/ — see stitch/README.md",
    );
    process.exit(1);
  }

  let stitch;
  try {
    ({ stitch } = await import("@google/stitch-sdk"));
  } catch {
    console.error(
      "Install @google/stitch-sdk first: npm install -D @google/stitch-sdk",
    );
    process.exit(1);
  }

  const project = stitch.project(PROJECT_ID);
  const manifest = {
    projectId: PROJECT_ID,
    projectName: "Elixir Apple-Inspired Wine RAG",
    fetchedAt: new Date().toISOString(),
    method: "sdk+curl",
    status: "fetched",
    designSystem: {
      id: DESIGN_SYSTEM_ID,
      localDir: `design-system/${DESIGN_SYSTEM_ID}`,
      files: [],
    },
    screens: [],
  };

  const dsDir = join(stitchRoot, "design-system", DESIGN_SYSTEM_ID);
  mkdirSync(dsDir, { recursive: true });

  try {
    if (typeof project.listDesignSystems === "function") {
      const systems = await project.listDesignSystems();
      writeFileSync(
        join(dsDir, "list-design-systems.json"),
        JSON.stringify(systems, null, 2),
        "utf8",
      );
      manifest.designSystem.files.push("list-design-systems.json");
    }
  } catch (err) {
    console.warn("listDesignSystems failed:", err?.message || err);
  }

  try {
    if (typeof project.designSystem === "function") {
      const ds = await project.designSystem(DESIGN_SYSTEM_ID);
      writeFileSync(
        join(dsDir, "design-system.json"),
        JSON.stringify(ds, null, 2),
        "utf8",
      );
      manifest.designSystem.files.push("design-system.json");

      for (const [label, getter] of [
        ["design-system.html", "getHtml"],
        ["design-system.png", "getImage"],
      ]) {
        try {
          if (typeof ds?.[getter] === "function") {
            const asset = await ds[getter]();
            const url = pickUrl(asset) || pickUrl(asset?.data);
            if (url) {
              curlDownload(url, join(dsDir, label));
              manifest.designSystem.files.push(label);
            } else if (typeof asset === "string" && asset.includes("<")) {
              writeFileSync(join(dsDir, label), asset, "utf8");
              manifest.designSystem.files.push(label);
            }
          }
        } catch (inner) {
          console.warn(`designSystem.${getter} failed:`, inner?.message || inner);
        }
      }
    }
  } catch (err) {
    console.warn("designSystem() failed:", err?.message || err);
  }

  let screens = [];
  try {
    screens = await project.screens();
  } catch (err) {
    console.warn("screens() failed:", err?.message || err);
  }

  for (const screen of screens || []) {
    const id =
      screen.id || screen.screenId || screen.name || `screen-${manifest.screens.length}`;
    const screenDir = join(stitchRoot, "screens", String(id));
    mkdirSync(screenDir, { recursive: true });
    const entry = { id: String(id), localDir: `screens/${id}`, files: [] };

    try {
      const html = await screen.getHtml?.();
      const htmlUrl = pickUrl(html) || pickUrl(html?.data);
      if (htmlUrl) {
        curlDownload(htmlUrl, join(screenDir, "screen.html"));
        entry.files.push("screen.html");
      } else if (typeof html === "string") {
        writeFileSync(join(screenDir, "screen.html"), html, "utf8");
        entry.files.push("screen.html");
      }
    } catch (err) {
      console.warn(`getHtml failed for ${id}:`, err?.message || err);
    }

    try {
      const image = await screen.getImage?.();
      const imageUrl = pickUrl(image) || pickUrl(image?.data);
      if (imageUrl) {
        const ext = imageUrl.includes(".webp") ? "webp" : "png";
        const name = `screen.${ext}`;
        curlDownload(imageUrl, join(screenDir, name));
        entry.files.push(name);
      }
    } catch (err) {
      console.warn(`getImage failed for ${id}:`, err?.message || err);
    }

    manifest.screens.push(entry);
  }

  if (!existsSync(stitchRoot)) mkdirSync(stitchRoot, { recursive: true });
  writeFileSync(
    join(stitchRoot, "manifest.json"),
    JSON.stringify(manifest, null, 2),
    "utf8",
  );
  console.log(`Wrote ${join(stitchRoot, "manifest.json")}`);
  console.log(
    `Design system files: ${manifest.designSystem.files.length}; screens: ${manifest.screens.length}`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
