import fs from 'node:fs/promises';
import path from 'node:path';

export const MINE_PATTERN_KINDS = [
  'ZERO_STATE_CLAMP',
  'DEFAULT_BUCKET_PARTITION',
  'HOT_LOOP_COMPLEXITY',
  'FRAMEWORK_IO_ABSTRACTION',
  'CONCURRENCY_LIVENESS',
] as const;

export type MinePatternKind = (typeof MINE_PATTERN_KINDS)[number];

export interface MinedFinding {
  kind: MinePatternKind;
  filePath: string;
  line: number;
  severity: 'high' | 'medium' | 'low';
  confidence: number;
  message: string;
  snippet: string;
}

export interface FileMiningResult {
  filePath: string;
  score: number;
  countsByKind: Record<MinePatternKind, number>;
  findings: MinedFinding[];
}

export interface RankedFileMiningResult extends FileMiningResult {
  distinctKinds: number;
}

export interface RankMiningOptions {
  limit?: number;
  minScore?: number;
  kinds?: MinePatternKind[];
}

const SEVERITY_WEIGHT: Record<MinedFinding['severity'], number> = {
  high: 5,
  medium: 3,
  low: 1,
};

const LOOP_HEADER_RE =
  /^\s*(?:for\s*\(|for\s+[A-Za-z_][\w,\s]*\s+in\s+|while\s*\(|while\s+.+:\s*$|\S+\.(?:forEach|map|filter|reduce|flatMap)\s*\()/;

const ZERO_SENSITIVE_VAR_RE =
  /\b(?:count|idx|index|offset|port|limit|timeout|delay|retries|retry|size|score|level|depth|start|min|max|zero|balance|amount|qty|quantity|status|code)\b/i;

function makeEmptyCounts(): Record<MinePatternKind, number> {
  return {
    ZERO_STATE_CLAMP: 0,
    DEFAULT_BUCKET_PARTITION: 0,
    HOT_LOOP_COMPLEXITY: 0,
    FRAMEWORK_IO_ABSTRACTION: 0,
    CONCURRENCY_LIVENESS: 0,
  };
}

function getIndent(line: string): number {
  const match = line.match(/^(\s*)/);
  return match?.[1]?.length ?? 0;
}

function addFinding(
  findings: MinedFinding[],
  seenKeys: Set<string>,
  finding: MinedFinding,
): void {
  const key = `${finding.kind}:${finding.line}:${finding.message}`;
  if (seenKeys.has(key)) {
    return;
  }
  seenKeys.add(key);
  findings.push(finding);
}

function detectZeroStateClamp(
  filePath: string,
  lines: string[],
  isPython: boolean,
  findings: MinedFinding[],
  seen: Set<string>,
): void {
  for (let i = 0; i < lines.length; i += 1) {
    const raw = lines[i]!;
    const trimmed = raw.trim();
    if (!trimmed || trimmed.startsWith('//') || trimmed.startsWith('#')) {
      continue;
    }

    // Falsy `||` or Python `or` fallback on zero-sensitive identifiers (where 0 is valid).
    const falsyFallbackMatch = isPython
      ? trimmed.match(
          /([A-Za-z_][\w.]*)\s+or\s+(-?\d+|""|''|\[|{|\bTrue\b|\bFalse\b)/,
        )
      : trimmed.match(
          /([A-Za-z_$][\w.?]*)\s*\|\|\s*(-?\d+|['"`]{2}|true|false|\[\]|\{\})/,
        );

    if (
      falsyFallbackMatch?.[1] &&
      ZERO_SENSITIVE_VAR_RE.test(falsyFallbackMatch[1])
    ) {
      addFinding(findings, seen, {
        kind: 'ZERO_STATE_CLAMP',
        filePath,
        line: i + 1,
        severity: 'high',
        confidence: 0.85,
        message: `Falsy fallback on "${falsyFallbackMatch[1]}" silently overrides 0/empty state; prefer nullish/None check.`,
        snippet: trimmed,
      });
    }

    // Negation check `!x` or `not x` on zero-sensitive variables.
    const falsyGuardMatch = isPython
      ? trimmed.match(/^(?:if|elif)\s+not\s+([A-Za-z_][\w.]*)\s*:/)
      : trimmed.match(/^if\s*\(\s*!([A-Za-z_$][\w.]*)\s*\)/);

    if (
      falsyGuardMatch?.[1] &&
      ZERO_SENSITIVE_VAR_RE.test(falsyGuardMatch[1])
    ) {
      addFinding(findings, seen, {
        kind: 'ZERO_STATE_CLAMP',
        filePath,
        line: i + 1,
        severity: 'medium',
        confidence: 0.78,
        message: `Truthiness check on "${falsyGuardMatch[1]}" treats 0 as missing/invalid.`,
        snippet: trimmed,
      });
    }

    // Suspicious clamping: Math.max(1, ...) or <= 0 guard on index/offset/count.
    if (
      /Math\.max\(\s*1\s*,/.test(trimmed) ||
      (ZERO_SENSITIVE_VAR_RE.test(trimmed) &&
        /\b(?:<=|>=)\s*0\b/.test(trimmed) &&
        /\b(?:return|throw|clamp|max|min)\b/i.test(trimmed))
    ) {
      addFinding(findings, seen, {
        kind: 'ZERO_STATE_CLAMP',
        filePath,
        line: i + 1,
        severity: 'medium',
        confidence: 0.72,
        message: 'Boundary clamp or <= 0 guard may reject valid zero state.',
        snippet: trimmed,
      });
    }
  }
}

function detectDefaultBucketPartition(
  filePath: string,
  lines: string[],
  isPython: boolean,
  findings: MinedFinding[],
  seen: Set<string>,
): void {
  for (let i = 0; i < lines.length; i += 1) {
    const trimmed = lines[i]!.trim();
    if (!trimmed || trimmed.startsWith('//') || trimmed.startsWith('#')) {
      continue;
    }

    // Python mutable default argument in function definition.
    if (isPython && /^def\s+\w+\s*\([^)]*=\s*(?:\[\]|\{\}|set\(\))/.test(trimmed)) {
      addFinding(findings, seen, {
        kind: 'DEFAULT_BUCKET_PARTITION',
        filePath,
        line: i + 1,
        severity: 'high',
        confidence: 0.94,
        message: 'Mutable default collection argument shares state across calls/partitions.',
        snippet: trimmed,
      });
    }

    // Shared default bucket / fallback partition lookup + mutation or assignment.
    if (
      /(?:bucket|partition|group|shard|slot)s?(?:\[[^\]]+\]|\.get\([^)]+\))\s*(?:\|\||\?\?|or)\s*(?:default|other|fallback|shared)/i.test(
        trimmed,
      ) ||
      /(?:defaultBucket|fallbackBucket|DEFAULT_BUCKET|SHARED_BUCKET)\s*\.\s*(?:push|set|add|append|update)\(/.test(
        trimmed,
      )
    ) {
      addFinding(findings, seen, {
        kind: 'DEFAULT_BUCKET_PARTITION',
        filePath,
        line: i + 1,
        severity: 'high',
        confidence: 0.86,
        message: 'Fallback collapses unmatched keys into a shared default bucket/partition.',
        snippet: trimmed,
      });
    }

    // Switch/mapping fallback routing directly to `'default'` or `'other'` bucket without isolation.
    if (
      /default\s*:\s*.*(?:bucket|partition|group|shard)\s*\[?\s*['"`](?:default|other|unknown)['"`]/i.test(
        trimmed,
      ) ||
      /\[(?:key|type|kind|category)\s*\|\|\s*['"`](?:default|other)['"`]\]/.test(
        trimmed,
      )
    ) {
      addFinding(findings, seen, {
        kind: 'DEFAULT_BUCKET_PARTITION',
        filePath,
        line: i + 1,
        severity: 'medium',
        confidence: 0.78,
        message: 'Default partition key coalesces distinct categories into a single bucket.',
        snippet: trimmed,
      });
    }
  }
}

function detectHotLoopComplexity(
  filePath: string,
  lines: string[],
  isPython: boolean,
  findings: MinedFinding[],
  seen: Set<string>,
): void {
  const loopStack: Array<{ startLine: number; braceDepth: number; indent: number }> = [];
  let braceDepth = 0;

  for (let i = 0; i < lines.length; i += 1) {
    const raw = lines[i]!;
    const trimmed = raw.trim();
    if (!trimmed || trimmed.startsWith('//') || trimmed.startsWith('#')) {
      continue;
    }

    const indent = getIndent(raw);

    if (isPython) {
      while (
        loopStack.length > 0 &&
        indent <= loopStack[loopStack.length - 1]!.indent
      ) {
        loopStack.pop();
      }
    } else {
      while (
        loopStack.length > 0 &&
        braceDepth <= loopStack[loopStack.length - 1]!.braceDepth
      ) {
        loopStack.pop();
      }
    }

    const isLoopHeader = LOOP_HEADER_RE.test(raw);
    const inLoop = loopStack.length > 0;

    if (inLoop) {
      if (isLoopHeader) {
        addFinding(findings, seen, {
          kind: 'HOT_LOOP_COMPLEXITY',
          filePath,
          line: i + 1,
          severity: 'medium',
          confidence: 0.8,
          message: 'Nested loop inside hot path introduces potential O(n^2) complexity.',
          snippet: trimmed,
        });
      }

      if (
        /\.(?:includes|indexOf|find|some|filter)\s*\(/.test(trimmed) ||
        (isPython && /\bin\s+[A-Za-z_][\w.]*\b/.test(trimmed) && !isLoopHeader)
      ) {
        addFinding(findings, seen, {
          kind: 'HOT_LOOP_COMPLEXITY',
          filePath,
          line: i + 1,
          severity: 'high',
          confidence: 0.84,
          message: 'Linear scan inside loop body causes O(n^2) scaling; consider a Set or Map.',
          snippet: trimmed,
        });
      }

      if (
        /\.(?:shift|unshift)\s*\(/.test(trimmed) ||
        /\[\.\.\.[A-Za-z_$][\w]*\s*,/.test(trimmed) ||
        (isPython && /\.pop\(\s*0\s*\)|\.insert\(\s*0\s*,/.test(trimmed))
      ) {
        addFinding(findings, seen, {
          kind: 'HOT_LOOP_COMPLEXITY',
          filePath,
          line: i + 1,
          severity: 'high',
          confidence: 0.86,
          message: 'O(n) array/list reallocation or front-removal inside loop.',
          snippet: trimmed,
        });
      }

      if (
        /new\s+RegExp\(|re\.compile\(|JSON\.(?:parse|stringify)\(|readFileSync\(/.test(
          trimmed,
        )
      ) {
        addFinding(findings, seen, {
          kind: 'HOT_LOOP_COMPLEXITY',
          filePath,
          line: i + 1,
          severity: 'medium',
          confidence: 0.82,
          message: 'Expensive regex compilation, serialization, or sync I/O inside loop.',
          snippet: trimmed,
        });
      }
    }

    if (isLoopHeader) {
      loopStack.push({ startLine: i + 1, braceDepth, indent });
    }

    if (!isPython) {
      const opens = (raw.match(/\{/g) ?? []).length;
      const closes = (raw.match(/\}/g) ?? []).length;
      braceDepth += opens - closes;
      if (braceDepth < 0) braceDepth = 0;
    }
  }
}

function detectFrameworkIoAbstraction(
  filePath: string,
  lines: string[],
  isPython: boolean,
  findings: MinedFinding[],
  seen: Set<string>,
): void {
  let tryDepth = 0;

  for (let i = 0; i < lines.length; i += 1) {
    const raw = lines[i]!;
    const trimmed = raw.trim();
    if (!trimmed || trimmed.startsWith('//') || trimmed.startsWith('#')) {
      continue;
    }

    if (/\btry\s*[:{]/.test(trimmed)) {
      tryDepth += 1;
    }
    if (/\b(?:catch|except)\b/.test(trimmed) && tryDepth > 0) {
      tryDepth -= 1;
    }

    if (
      /\b(?:exec|execSync)\s*\(\s*`[^`]*\$\{/.test(trimmed) ||
      (isPython && /(?:subprocess\.\w+\([^)]*shell\s*=\s*True|os\.system\()/.test(trimmed))
    ) {
      addFinding(findings, seen, {
        kind: 'FRAMEWORK_IO_ABSTRACTION',
        filePath,
        line: i + 1,
        severity: 'high',
        confidence: 0.92,
        message: 'Shell command execution with interpolation or shell=True bypasses safe arg arrays.',
        snippet: trimmed,
      });
    }

    if (
      /`\s*(?:SELECT|INSERT|UPDATE|DELETE)\b[^`]*\$\{/i.test(trimmed) ||
      (isPython && /f['"]\s*(?:SELECT|INSERT|UPDATE|DELETE)\b[^'"]*\{/i.test(trimmed))
    ) {
      addFinding(findings, seen, {
        kind: 'FRAMEWORK_IO_ABSTRACTION',
        filePath,
        line: i + 1,
        severity: 'high',
        confidence: 0.93,
        message: 'Raw SQL string interpolation bypasses parameterized query abstraction.',
        snippet: trimmed,
      });
    }

    if (
      /path\.(?:join|resolve)\([^)]*\b(?:req\.|params\.|query\.|body\.|userInput)/.test(
        trimmed,
      )
    ) {
      addFinding(findings, seen, {
        kind: 'FRAMEWORK_IO_ABSTRACTION',
        filePath,
        line: i + 1,
        severity: 'high',
        confidence: 0.85,
        message: 'Path constructed from request/user input without verified containment check.',
        snippet: trimmed,
      });
    }

    if (!isPython && /\bJSON\.parse\s*\(/.test(trimmed) && tryDepth === 0) {
      addFinding(findings, seen, {
        kind: 'FRAMEWORK_IO_ABSTRACTION',
        filePath,
        line: i + 1,
        severity: 'medium',
        confidence: 0.76,
        message: 'Unguarded JSON.parse() outside try/catch can crash request/pipeline handlers.',
        snippet: trimmed,
      });
    }

    if (
      /\b(?:readFileSync|writeFileSync|appendFileSync)\s*\(/.test(trimmed) &&
      lines
        .slice(Math.max(0, i - 10), i + 1)
        .some((l) => /\b(?:app|router)\.(?:get|post|put|delete|use)\s*\(|async\s+/.test(l))
    ) {
      addFinding(findings, seen, {
        kind: 'FRAMEWORK_IO_ABSTRACTION',
        filePath,
        line: i + 1,
        severity: 'medium',
        confidence: 0.8,
        message: 'Synchronous filesystem I/O inside async or HTTP handler blocks the event loop.',
        snippet: trimmed,
      });
    }
  }
}

function detectConcurrencyLiveness(
  filePath: string,
  lines: string[],
  isPython: boolean,
  findings: MinedFinding[],
  seen: Set<string>,
): void {
  const fullText = lines.join('\n');

  for (let i = 0; i < lines.length; i += 1) {
    const trimmed = lines[i]!.trim();
    if (!trimmed || trimmed.startsWith('//') || trimmed.startsWith('#')) {
      continue;
    }

    // `new Promise(async ...)` swallows rejections and breaks error propagation.
    if (/new\s+Promise\s*\(\s*async\b/.test(trimmed)) {
      addFinding(findings, seen, {
        kind: 'CONCURRENCY_LIVENESS',
        filePath,
        line: i + 1,
        severity: 'high',
        confidence: 0.94,
        message: 'Async Promise executor swallows thrown errors and violates liveness/error contracts.',
        snippet: trimmed,
      });
    }

    // Infinite spin loop without break, return, await, or sleep nearby.
    if (
      (isPython && /^while\s+True\s*:/.test(trimmed)) ||
      (!isPython && /^(?:while\s*\(\s*true\s*\)|for\s*\(\s*;\s*;\s*\))/.test(trimmed))
    ) {
      const window = lines.slice(i + 1, Math.min(lines.length, i + 15)).join('\n');
      if (!/\b(?:break|return|await|sleep|yield|throw|raise)\b/.test(window)) {
        addFinding(findings, seen, {
          kind: 'CONCURRENCY_LIVENESS',
          filePath,
          line: i + 1,
          severity: 'high',
          confidence: 0.85,
          message: 'Unbounded loop lacks visible yield/await/break exit, risking event-loop starvation.',
          snippet: trimmed,
        });
      }
    }

    // Lock acquisition without try/finally release nearby.
    if (/\.(?:acquire|lock)\s*\(/.test(trimmed)) {
      const window = lines.slice(i, Math.min(lines.length, i + 16)).join('\n');
      if (!/\bfinally\b/.test(window)) {
        addFinding(findings, seen, {
          kind: 'CONCURRENCY_LIVENESS',
          filePath,
          line: i + 1,
          severity: 'high',
          confidence: 0.82,
          message: 'Lock acquisition without nearby finally block risks deadlock on exception.',
          snippet: trimmed,
        });
      }
    }

    // Check-then-act race across await boundaries.
    if (
      /\bawait\b.*(?:exists|access|stat|has)\s*\(/.test(trimmed) &&
      lines
        .slice(i + 1, Math.min(lines.length, i + 6))
        .some((l) => /\bawait\b.*(?:write|create|mkdir|unlink|rm|set)\s*\(/.test(l))
    ) {
      addFinding(findings, seen, {
        kind: 'CONCURRENCY_LIVENESS',
        filePath,
        line: i + 1,
        severity: 'medium',
        confidence: 0.78,
        message: 'TOCTOU check-then-act across await boundary is vulnerable to concurrent races.',
        snippet: trimmed,
      });
    }

    // Unbounded setInterval without clearInterval in module.
    if (/\bsetInterval\s*\(/.test(trimmed) && !/\bclearInterval\b/.test(fullText)) {
      addFinding(findings, seen, {
        kind: 'CONCURRENCY_LIVENESS',
        filePath,
        line: i + 1,
        severity: 'medium',
        confidence: 0.8,
        message: 'setInterval created without any corresponding clearInterval cleanup in file.',
        snippet: trimmed,
      });
    }
  }
}

export async function mineFilePatterns(
  filePath: string,
  content?: string,
): Promise<FileMiningResult> {
  const source = content ?? (await fs.readFile(filePath, 'utf8'));
  const lines = source.split(/\r?\n/);
  const ext = path.extname(filePath).toLowerCase();
  const isPython = ext === '.py';

  const findings: MinedFinding[] = [];
  const seen = new Set<string>();

  detectZeroStateClamp(filePath, lines, isPython, findings, seen);
  detectDefaultBucketPartition(filePath, lines, isPython, findings, seen);
  detectHotLoopComplexity(filePath, lines, isPython, findings, seen);
  detectFrameworkIoAbstraction(filePath, lines, isPython, findings, seen);
  detectConcurrencyLiveness(filePath, lines, isPython, findings, seen);

  findings.sort((a, b) => a.line - b.line || a.kind.localeCompare(b.kind));

  const countsByKind = makeEmptyCounts();
  let score = 0;
  for (const finding of findings) {
    countsByKind[finding.kind] += 1;
    score += SEVERITY_WEIGHT[finding.severity];
  }

  return {
    filePath,
    score,
    countsByKind,
    findings,
  };
}

export async function rankFilesWithMining(
  files: Array<string | { path: string; content?: string }>,
  options: RankMiningOptions = {},
): Promise<RankedFileMiningResult[]> {
  const allowedKinds = options.kinds ? new Set(options.kinds) : null;
  const minScore = options.minScore ?? 0;

  const mined = await Promise.all(
    files.map(async (entry) => {
      const targetPath = typeof entry === 'string' ? entry : entry.path;
      const targetContent = typeof entry === 'string' ? undefined : entry.content;
      const result = await mineFilePatterns(targetPath, targetContent);

      if (!allowedKinds) {
        return result;
      }

      const filteredFindings = result.findings.filter((f) =>
        allowedKinds.has(f.kind),
      );
      const countsByKind = makeEmptyCounts();
      let score = 0;
      for (const f of filteredFindings) {
        countsByKind[f.kind] += 1;
        score += SEVERITY_WEIGHT[f.severity];
      }
      return {
        filePath: result.filePath,
        score,
        countsByKind,
        findings: filteredFindings,
      };
    }),
  );

  const ranked: RankedFileMiningResult[] = mined
    .map((item) => {
      const distinctKinds = MINE_PATTERN_KINDS.filter(
        (kind) => item.countsByKind[kind] > 0,
      ).length;
      return {
        ...item,
        distinctKinds,
      };
    })
    .filter((item) => item.score >= minScore)
    .sort(
      (a, b) =>
        b.score - a.score ||
        b.distinctKinds - a.distinctKinds ||
        a.filePath.localeCompare(b.filePath),
    );

  if (options.limit && options.limit > 0) {
    return ranked.slice(0, options.limit);
  }
  return ranked;
}
