import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  MINE_PATTERN_KINDS,
  mineFilePatterns,
  rankFilesWithMining,
} from '../src/utils/ast-miner.js';

describe('Phase 3: ast-miner.ts', () => {
  it('mines all five required bug-pattern categories accurately', async () => {
    const tsSample = `
      function configure(opts: { port?: number; count?: number }) {
        const port = opts.port || 3000;
        if (!opts.count) return 0;
        return port;
      }

      function routeBucket(key: string, buckets: Map<string, string[]>, defaultBucket: string[]) {
        const target = buckets.get(key) || defaultBucket;
        target.push(key);
      }

      function findMatches(items: string[], seen: string[]) {
        const out: string[] = [];
        for (const item of items) {
          if (seen.includes(item)) {
            out.push(item);
          }
        }
        return out;
      }

      async function handleReq(req: { params: { file: string }; query: { raw: string } }) {
        const parsed = JSON.parse(req.query.raw);
        const target = path.join('/var/data', req.params.file);
        return { parsed, target };
      }

      function runWithLock(mutex: { acquire(): void }) {
        mutex.acquire();
        return new Promise(async (resolve) => {
          while (true) {
            // spin
          }
          resolve(true);
        });
      }
    `;

    const result = await mineFilePatterns('src/buggy.ts', tsSample);
    for (const kind of MINE_PATTERN_KINDS) {
      assert.ok(
        result.countsByKind[kind] >= 1,
        `Expected at least 1 finding for ${kind}, got ${result.countsByKind[kind]}`,
      );
    }
    assert.ok(result.score > 0);
  });

  it('detects Python mutable defaults and shell=True hazards and ranks buggier files first', async () => {
    const cleanFile = {
      path: 'src/clean.py',
      content: `
def add(a: int, b: int) -> int:
    return a + b
`,
    };

    const riskyFile = {
      path: 'src/risky.py',
      content: `
import subprocess

def append_bucket(item, bucket=[]):
    bucket.append(item)
    return bucket

def run_cmd(user_arg: str):
    return subprocess.run(f"echo {user_arg}", shell=True)
`,
    };

    const ranked = await rankFilesWithMining([cleanFile, riskyFile]);
    assert.equal(ranked.length, 2);
    assert.equal(ranked[0]?.filePath, 'src/risky.py');
    assert.ok(ranked[0]!.countsByKind.DEFAULT_BUCKET_PARTITION >= 1);
    assert.ok(ranked[0]!.countsByKind.FRAMEWORK_IO_ABSTRACTION >= 1);
    assert.equal(ranked[1]?.filePath, 'src/clean.py');
    assert.equal(ranked[1]?.score, 0);
  });
});
