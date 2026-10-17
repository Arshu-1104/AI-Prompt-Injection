// Draws the PromptGuard icon: an original, simple shield with a check mark (no third-party
// marks). Writes transparent RGBA PNGs at 16, 32, 48 and 128 px to icons/.
// Run with: node scripts/create-icons.mjs
import { mkdirSync, writeFileSync } from "node:fs";
import { deflateSync } from "node:zlib";

const SIZES = [16, 32, 48, 128];
const SHIELD = [27, 108, 168]; // #1B6CA8, the extension's "safe" blue
const EDGE = [13, 61, 99]; // #0D3D63, darker rim
const CHECK = [255, 255, 255];
const SAMPLES = 4; // 4x4 supersampling per pixel for anti-aliasing

function crc32(data) {
  let crc = 0xffffffff;
  for (const byte of data) {
    crc ^= byte;
    for (let i = 0; i < 8; i += 1) crc = crc & 1 ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1;
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const typeBuf = Buffer.from(type, "ascii");
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])));
  return Buffer.concat([length, typeBuf, data, crc]);
}

function png(size, rgba) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // RGBA
  const rows = [];
  for (let y = 0; y < size; y += 1) rows.push(Buffer.from([0]), rgba.subarray(y * size * 4, (y + 1) * size * 4));
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(Buffer.concat(rows))),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

// Shield in unit coordinates: a gently peaked top, straight sides, and a curved point at the bottom.
function insideShield(x, y, inset) {
  const half = 0.36 - inset;
  const dx = Math.abs(x - 0.5);
  if (dx > half) return false;
  const top = 0.17 - 0.09 * (1 - dx / 0.36) + inset;
  if (y < top) return false;
  const shoulder = 0.52;
  if (y <= shoulder) return true;
  const bottom = 0.95 - inset * 1.4;
  if (y > bottom) return false;
  const t = (y - shoulder) / (bottom - shoulder);
  return dx <= half * Math.sqrt(1 - t * t);
}

function distanceToSegment(px, py, ax, ay, bx, by) {
  const vx = bx - ax;
  const vy = by - ay;
  const t = Math.max(0, Math.min(1, ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy)));
  return Math.hypot(px - (ax + t * vx), py - (ay + t * vy));
}

function onCheck(x, y, size) {
  const width = size <= 16 ? 0.075 : 0.055; // thicker stroke so it stays visible at 16 px
  return (
    distanceToSegment(x, y, 0.33, 0.52, 0.46, 0.65) <= width ||
    distanceToSegment(x, y, 0.46, 0.65, 0.69, 0.38) <= width
  );
}

function draw(size) {
  const rgba = Buffer.alloc(size * size * 4);
  const rim = size <= 16 ? 0 : 0.035; // no rim at 16 px: it would only blur the edge
  for (let py = 0; py < size; py += 1) {
    for (let px = 0; px < size; px += 1) {
      let r = 0, g = 0, b = 0, a = 0;
      for (let sy = 0; sy < SAMPLES; sy += 1) {
        for (let sx = 0; sx < SAMPLES; sx += 1) {
          const x = (px + (sx + 0.5) / SAMPLES) / size;
          const y = (py + (sy + 0.5) / SAMPLES) / size;
          if (!insideShield(x, y, 0)) continue;
          const color = onCheck(x, y, size) ? CHECK : insideShield(x, y, rim) ? SHIELD : EDGE;
          r += color[0]; g += color[1]; b += color[2]; a += 1;
        }
      }
      const i = (py * size + px) * 4;
      if (a) {
        rgba[i] = Math.round(r / a);
        rgba[i + 1] = Math.round(g / a);
        rgba[i + 2] = Math.round(b / a);
        rgba[i + 3] = Math.round((255 * a) / (SAMPLES * SAMPLES));
      }
    }
  }
  return png(size, rgba);
}

mkdirSync("icons", { recursive: true });
for (const size of SIZES) writeFileSync(`icons/icon${size}.png`, draw(size));
console.log(`Wrote icons/icon{${SIZES.join(",")}}.png`);
