/**
 * Build the preprocessing parity fixture from Google's Quick Draw files.
 *
 * Reads the head of `full/raw/<category>.ndjson` (unsimplified device
 * coordinates) and `full/simplified/<category>.ndjson` (Google's output),
 * pairs records by `key_id`, and writes the first N pairs to
 * `test/fixtures/quickdraw-<category>-parity.json`. Both files list sketches
 * in the same order, so a few megabytes of each already overlap.
 *
 *   node --experimental-strip-types scripts/fetch-parity-fixture.ts [category] [count]
 */
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const category = process.argv[2] ?? "cat";
const count = Number(process.argv[3] ?? 120);
const base = "https://storage.googleapis.com/quickdraw_dataset/full";
const rawUrl = `${base}/raw/${encodeURIComponent(category)}.ndjson`;
const simplifiedUrl = `${base}/simplified/${encodeURIComponent(category)}.ndjson`;
const rawBytes = 6_000_000;
const simplifiedBytes = 3_000_000;

interface RawRecord {
  key_id: string;
  recognized: boolean;
  drawing: [number[], number[], number[]][];
}
interface SimplifiedRecord {
  key_id: string;
  drawing: [number[], number[]][];
}

async function head<T>(url: string, bytes: number): Promise<Map<string, T>> {
  const response = await fetch(url, {
    headers: { Range: `bytes=0-${bytes - 1}` },
  });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const lines = (await response.text()).split("\n");
  lines.pop(); // the range cut the last line
  const records = new Map<string, T>();
  for (const line of lines) {
    if (!line.trim()) continue;
    const record = JSON.parse(line) as T & { key_id: string };
    records.set(record.key_id, record);
  }
  return records;
}

const [raw, simplified] = await Promise.all([
  head<RawRecord>(rawUrl, rawBytes),
  head<SimplifiedRecord>(simplifiedUrl, simplifiedBytes),
]);
const pairs = [];
for (const [keyId, record] of raw) {
  const reference = simplified.get(keyId);
  if (!reference) continue;
  pairs.push({
    keyId,
    recognized: record.recognized,
    raw: record.drawing.map(([x, y]) => ({ x, y })),
    simplified: reference.drawing.map(([x, y]) => ({ x, y })),
  });
  if (pairs.length >= count) break;
}
if (pairs.length < count)
  throw new Error(
    `Only ${pairs.length} overlapping records; raise the byte ranges.`,
  );

const fixture = {
  source: {
    raw: rawUrl,
    simplified: simplifiedUrl,
    rawBytes,
    simplifiedBytes,
    license: "CC BY 4.0, Google Quick Draw dataset",
    retrieved: new Date().toISOString().slice(0, 10),
  },
  pairs,
};
const directory = resolve(import.meta.dirname, "../test/fixtures");
await mkdir(directory, { recursive: true });
const path = resolve(directory, `quickdraw-${category}-parity.json`);
await writeFile(path, JSON.stringify(fixture) + "\n");
console.log(
  `${path}: ${pairs.length} pairs from ${raw.size} raw and ${simplified.size} simplified records`,
);
