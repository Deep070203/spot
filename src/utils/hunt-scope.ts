import { execFile } from 'node:child_process';
import path from 'node:path';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

const SOURCE_EXTENSIONS = new Set([
  '.ts',
  '.tsx',
  '.mts',
  '.cts',
  '.js',
  '.jsx',
  '.mjs',
  '.cjs',
  '.py',
  '.go',
  '.rs',
  '.java',
  '.kt',
  '.rb',
  '.php',
  '.c',
  '.cc',
  '.cpp',
  '.h',
  '.hpp',
  '.cs',
  '.swift',
  '.scala',
  '.sh',
]);

const IGNORED_PATH_SEGMENTS = new Set([
  'node_modules',
  '.git',
  'dist',
  'build',
  'out',
  '.next',
  '.nuxt',
  'coverage',
  '__pycache__',
  '.venv',
  'venv',
  'target',
  'vendor',
  'third_party',
  'tests',
  'test',
  '__tests__',
  'spec',
  'specs',
  'fixtures',
  '__mocks__',
  'docs',
  'doc',
]);

const IGNORED_FILE_BASENAMES = new Set([
  'conftest.py',
  'package-lock.json',
  'yarn.lock',
  'pnpm-lock.yaml',
  'cargo.lock',
  'poetry.lock',
]);

export interface DiskChanges {
  staged: string[];
  modified: string[];
  untracked: string[];
  all: string[];
  hasChanges: boolean;
}

export interface ChurnCluster {
  scope: string;
  churnScore: number;
  fileCount: number;
  files: string[];
}

export interface HuntScopeOptions {
  extensions?: string[];
  maxFiles?: number;
  churnCommitLimit?: number;
  scopeDepth?: number;
  includeDirtyBoost?: boolean;
}

export interface HuntScopeResult {
  repoDir: string;
  totalTrackedFiles: number;
  sourceFilesCount: number;
  scopeDirs: string[];
  selectedFiles: string[];
  churnClusters: ChurnCluster[];
  dirtyFiles: string[];
}

function normalizePosixPath(filePath: string): string {
  return filePath.replace(/\\/g, '/').replace(/^\.\//, '');
}

function normalizeExtension(ext: string): string {
  const lower = ext.trim().toLowerCase();
  return lower.startsWith('.') ? lower : `.${lower}`;
}

export function isHuntSource(filePath: string): boolean {
  const normalized = normalizePosixPath(filePath);
  if (!normalized || normalized.endsWith('/')) {
    return false;
  }

  const segments = normalized.split('/');
  const baseName = segments[segments.length - 1] ?? '';
  const lowerBase = baseName.toLowerCase();

  if (IGNORED_FILE_BASENAMES.has(lowerBase)) {
    return false;
  }

  for (let i = 0; i < segments.length - 1; i += 1) {
    const seg = segments[i]!.toLowerCase();
    if (IGNORED_PATH_SEGMENTS.has(seg)) {
      return false;
    }
  }

  if (
    lowerBase.endsWith('.d.ts') ||
    lowerBase.endsWith('.min.js') ||
    lowerBase.endsWith('.min.css') ||
    /\.test\.[^.]+$/.test(lowerBase) ||
    /\.spec\.[^.]+$/.test(lowerBase) ||
    /^test_.*\.py$/.test(lowerBase) ||
    /_test\.(py|go)$/.test(lowerBase)
  ) {
    return false;
  }

  const ext = path.posix.extname(lowerBase);
  return SOURCE_EXTENSIONS.has(ext);
}

/**
 * Computes scoped directory prefixes from `files` that contain at least one file
 * matching any of `extensions`.
 */
export function scopeContainingExtensions(
  files: string[],
  extensions: Iterable<string>,
  options: { depth?: number } = {},
): string[] {
  const extSet = new Set(Array.from(extensions, normalizeExtension));
  if (extSet.size === 0) {
    return [];
  }

  const depth = Math.max(1, options.depth ?? 2);
  const scopes = new Set<string>();

  for (const rawFile of files) {
    const normalized = normalizePosixPath(rawFile);
    const ext = path.posix.extname(normalized).toLowerCase();
    if (!extSet.has(ext)) {
      continue;
    }

    const dir = path.posix.dirname(normalized);
    if (dir === '.' || dir === '') {
      scopes.add('.');
      continue;
    }

    const parts = dir.split('/').filter(Boolean);
    scopes.add(parts.slice(0, depth).join('/'));
  }

  return [...scopes].sort();
}

export async function changesOnDisk(repoDir: string): Promise<DiskChanges> {
  const resolvedRepo = path.resolve(repoDir);
  const { stdout } = await execFileAsync(
    'git',
    ['status', '--porcelain=v1', '-z'],
    { cwd: resolvedRepo, maxBuffer: 10 * 1024 * 1024 },
  );

  const staged = new Set<string>();
  const modified = new Set<string>();
  const untracked = new Set<string>();

  const records = stdout.split('\0');
  for (let i = 0; i < records.length; i += 1) {
    const entry = records[i];
    if (!entry || entry.length < 4) continue;

    const indexStatus = entry[0]!;
    const worktreeStatus = entry[1]!;
    let filePath = entry.slice(3);

    // Renames/copies in -z format have an extra NUL-separated source path.
    if (indexStatus === 'R' || indexStatus === 'C') {
      i += 1;
    }

    filePath = normalizePosixPath(filePath);
    if (!filePath) continue;

    if (indexStatus === '?' && worktreeStatus === '?') {
      untracked.add(filePath);
      continue;
    }
    if (indexStatus !== ' ' && indexStatus !== '?') {
      staged.add(filePath);
    }
    if (worktreeStatus !== ' ' && worktreeStatus !== '?') {
      modified.add(filePath);
    }
  }

  const all = [...new Set([...staged, ...modified, ...untracked])].sort();
  return {
    staged: [...staged].sort(),
    modified: [...modified].sort(),
    untracked: [...untracked].sort(),
    all,
    hasChanges: all.length > 0,
  };
}

async function listTrackedFiles(repoDir: string): Promise<string[]> {
  const { stdout } = await execFileAsync('git', ['ls-files', '-z'], {
    cwd: repoDir,
    maxBuffer: 20 * 1024 * 1024,
  });
  return stdout
    .split('\0')
    .map(normalizePosixPath)
    .filter((file) => file.length > 0);
}

async function collectGitChurn(
  repoDir: string,
  commitLimit: number,
): Promise<Map<string, number>> {
  const counts = new Map<string, number>();
  try {
    const { stdout } = await execFileAsync(
      'git',
      ['log', '--name-only', '--format=', '-n', String(commitLimit)],
      { cwd: repoDir, maxBuffer: 20 * 1024 * 1024 },
    );
    for (const line of stdout.split(/\r?\n/)) {
      const normalized = normalizePosixPath(line.trim());
      if (!normalized) continue;
      counts.set(normalized, (counts.get(normalized) ?? 0) + 1);
    }
  } catch {
    // Empty repo or detached shallow history without commits.
  }
  return counts;
}

function getScopeKey(filePath: string, depth: number): string {
  const dir = path.posix.dirname(filePath);
  if (dir === '.' || dir === '') {
    return '.';
  }
  return dir.split('/').slice(0, depth).join('/');
}

/**
 * Selects a focused hunt scope for large/monorepo repositories using `git ls-files -z`,
 * git churn clustering, source filtering, and optional uncommitted disk changes.
 */
export async function selectHuntScope(
  repoDir: string,
  options: HuntScopeOptions = {},
): Promise<HuntScopeResult> {
  const resolvedRepo = path.resolve(repoDir);
  const maxFiles = options.maxFiles ?? 120;
  const churnCommitLimit = options.churnCommitLimit ?? 100;
  const scopeDepth = Math.max(1, options.scopeDepth ?? 2);
  const extFilter = options.extensions
    ? new Set(options.extensions.map(normalizeExtension))
    : null;

  const [trackedFiles, churnMap, diskChanges] = await Promise.all([
    listTrackedFiles(resolvedRepo),
    collectGitChurn(resolvedRepo, churnCommitLimit),
    changesOnDisk(resolvedRepo).catch(
      (): DiskChanges => ({
        staged: [],
        modified: [],
        untracked: [],
        all: [],
        hasChanges: false,
      }),
    ),
  ]);

  const sourceFiles = trackedFiles.filter((file) => {
    if (!isHuntSource(file)) return false;
    if (extFilter && !extFilter.has(path.posix.extname(file).toLowerCase())) {
      return false;
    }
    return true;
  });

  const dirtySet = new Set(diskChanges.all);
  const clusterMap = new Map<
    string,
    { scope: string; churnScore: number; files: Array<{ path: string; score: number }> }
  >();

  for (const file of sourceFiles) {
    const scope = getScopeKey(file, scopeDepth);
    const baseChurn = churnMap.get(file) ?? 0;
    const dirtyBonus = options.includeDirtyBoost !== false && dirtySet.has(file) ? 5 : 0;
    const fileScore = baseChurn + dirtyBonus;

    let cluster = clusterMap.get(scope);
    if (!cluster) {
      cluster = { scope, churnScore: 0, files: [] };
      clusterMap.set(scope, cluster);
    }
    cluster.churnScore += fileScore;
    cluster.files.push({ path: file, score: fileScore });
  }

  const churnClusters: ChurnCluster[] = [...clusterMap.values()]
    .map((cluster) => {
      cluster.files.sort((a, b) => b.score - a.score || a.path.localeCompare(b.path));
      return {
        scope: cluster.scope,
        churnScore: cluster.churnScore,
        fileCount: cluster.files.length,
        files: cluster.files.map((f) => f.path),
      };
    })
    .sort(
      (a, b) =>
        b.churnScore - a.churnScore ||
        b.fileCount - a.fileCount ||
        a.scope.localeCompare(b.scope),
    );

  const selectedFiles: string[] = [];
  const selectedScopes: string[] = [];

  for (const cluster of churnClusters) {
    if (selectedFiles.length >= maxFiles) break;
    selectedScopes.push(cluster.scope);
    for (const file of cluster.files) {
      if (selectedFiles.length >= maxFiles) break;
      selectedFiles.push(file);
    }
  }

  const activeExtensions =
    extFilter && extFilter.size > 0
      ? extFilter
      : new Set(
          selectedFiles
            .map((f) => path.posix.extname(f).toLowerCase())
            .filter(Boolean),
        );

  const verifiedScopes = scopeContainingExtensions(selectedFiles, activeExtensions, {
    depth: scopeDepth,
  });

  return {
    repoDir: resolvedRepo,
    totalTrackedFiles: trackedFiles.length,
    sourceFilesCount: sourceFiles.length,
    scopeDirs: verifiedScopes.length > 0 ? verifiedScopes : selectedScopes,
    selectedFiles,
    churnClusters,
    dirtyFiles: diskChanges.all,
  };
}
