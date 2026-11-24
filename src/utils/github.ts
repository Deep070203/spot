import fs from 'node:fs/promises';
import path from 'node:path';
import { Octokit } from 'octokit';
import { simpleGit, type SimpleGit } from 'simple-git';

export interface RepoSlug {
  owner: string;
  repo: string;
  fullName: string;
}

export interface GitHubIssueSummary {
  number: number;
  title: string;
  body: string;
  state: string;
  labels: string[];
  url: string;
  htmlUrl: string;
  author: string;
  createdAt: string;
  updatedAt: string;
  comments: number;
  score: number;
}

export interface OpenBugIssueQueryOptions {
  labels?: string[];
  excludeLabels?: string[];
  perPage?: number;
  preferLabeled?: boolean;
  query?: string;
}

export interface CloneRepoOptions {
  branch?: string;
  depth?: number;
  pullIfExists?: boolean;
}

export interface CloneRepoResult {
  localPath: string;
  defaultBranch: string;
  commitSha: string;
  clonedFresh: boolean;
}

export interface ForkRepoResult {
  owner: string;
  repo: string;
  fullName: string;
  htmlUrl: string;
  cloneUrl: string;
  defaultBranch: string;
}

export interface PRFileChange {
  filename: string;
  status: string;
  additions: number;
  deletions: number;
  changes: number;
  patch?: string;
  blobUrl?: string;
}

export interface RepoTemplateFile {
  path: string;
  name: string;
  content: string;
}

export interface RepoTemplatesResult {
  issueTemplates: RepoTemplateFile[];
  prTemplates: RepoTemplateFile[];
}

export interface ContributingFileResult {
  path: string;
  content: string;
}

export interface RepoMetadata {
  owner: string;
  repo: string;
  fullName: string;
  description: string | null;
  defaultBranch: string;
  language: string | null;
  stars: number;
  isFork: boolean;
  htmlUrl: string;
  cloneUrl: string;
}

export type OctokitFactory = (token?: string) => Octokit;
export type SimpleGitFactory = (baseDir?: string) => SimpleGit;

export interface GitHubServiceOptions {
  token?: string;
  octokitFactory?: OctokitFactory;
  gitFactory?: SimpleGitFactory;
}

const DEFAULT_BUG_LABELS = [
  'bug',
  'defect',
  'regression',
  'crash',
  'type: bug',
  'kind/bug',
  'confirmed',
  'priority: high',
];

const DEFAULT_EXCLUDED_LABELS = [
  'wontfix',
  'duplicate',
  'invalid',
  'question',
  'documentation',
  'enhancement',
  'feature',
];

function getHttpStatus(error: unknown): number | undefined {
  if (typeof error !== 'object' || error === null) {
    return undefined;
  }
  const maybeStatus = (error as { status?: unknown }).status;
  if (typeof maybeStatus === 'number') {
    return maybeStatus;
  }
  const maybeResponse = (error as { response?: { status?: unknown } }).response;
  if (maybeResponse && typeof maybeResponse.status === 'number') {
    return maybeResponse.status;
  }
  return undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function decodeBase64Content(encoded: string): string {
  return Buffer.from(encoded.replace(/\n/g, ''), 'base64').toString('utf8');
}

export function parseRepoSlug(input: string): RepoSlug {
  const trimmed = input.trim().replace(/\.git$/, '').replace(/\/+$/, '');
  const urlMatch = trimmed.match(/github\.com[:/]([^/]+)\/([^/]+)$/i);
  if (urlMatch?.[1] && urlMatch[2]) {
    return {
      owner: urlMatch[1],
      repo: urlMatch[2],
      fullName: `${urlMatch[1]}/${urlMatch[2]}`,
    };
  }

  const slashParts = trimmed.split('/').filter(Boolean);
  if (slashParts.length === 2 && slashParts[0] && slashParts[1]) {
    return {
      owner: slashParts[0],
      repo: slashParts[1],
      fullName: `${slashParts[0]}/${slashParts[1]}`,
    };
  }

  throw new Error(`Invalid GitHub repository slug or URL: "${input}"`);
}

export class GitHubService {
  private readonly token?: string;
  private readonly octokitFactory: OctokitFactory;
  private readonly gitFactory: SimpleGitFactory;
  private primaryClient: Octokit;
  private unauthenticatedClient: Octokit | null = null;
  private fellBackToUnauthenticated = false;

  constructor(options: GitHubServiceOptions = {}) {
    const envToken = options.token ?? process.env.GITHUB_TOKEN;
    this.token = envToken && envToken.trim().length > 0 ? envToken.trim() : undefined;
    this.octokitFactory =
      options.octokitFactory ??
      ((authToken?: string) =>
        new Octokit(authToken ? { auth: authToken } : undefined));
    this.gitFactory =
      options.gitFactory ??
      ((baseDir?: string) => (baseDir ? simpleGit(baseDir) : simpleGit()));
    this.primaryClient = this.octokitFactory(this.token);
  }

  get isAuthenticated(): boolean {
    return Boolean(this.token) && !this.fellBackToUnauthenticated;
  }

  private getUnauthenticatedClient(): Octokit {
    if (!this.unauthenticatedClient) {
      this.unauthenticatedClient = this.octokitFactory(undefined);
    }
    return this.unauthenticatedClient;
  }

  async withAuthFallback<T>(
    operation: (client: Octokit) => Promise<T>,
  ): Promise<T> {
    if (this.fellBackToUnauthenticated || !this.token) {
      return operation(this.getUnauthenticatedClient());
    }

    try {
      return await operation(this.primaryClient);
    } catch (error) {
      if (getHttpStatus(error) === 401) {
        this.fellBackToUnauthenticated = true;
        return operation(this.getUnauthenticatedClient());
      }
      throw error;
    }
  }

  async getRepoMetadata(owner: string, repo: string): Promise<RepoMetadata> {
    return this.withAuthFallback(async (client) => {
      const { data } = await client.rest.repos.get({ owner, repo });
      return {
        owner: data.owner.login,
        repo: data.name,
        fullName: data.full_name,
        description: data.description,
        defaultBranch: data.default_branch,
        language: data.language,
        stars: data.stargazers_count,
        isFork: data.fork,
        htmlUrl: data.html_url,
        cloneUrl: data.clone_url,
      };
    });
  }

  async openBugIssueQuery(
    owner: string,
    repo: string,
    options: OpenBugIssueQueryOptions = {},
  ): Promise<GitHubIssueSummary[]> {
    const perPage = options.perPage ?? 30;
    const bugLabels = (options.labels ?? DEFAULT_BUG_LABELS).map((l) =>
      l.toLowerCase(),
    );
    const excludeLabels = new Set(
      (options.excludeLabels ?? DEFAULT_EXCLUDED_LABELS).map((l) =>
        l.toLowerCase(),
      ),
    );

    const searchParts = [
      `repo:${owner}/${repo}`,
      'is:issue',
      'is:open',
    ];
    if (options.query?.trim()) {
      searchParts.push(options.query.trim());
    }

    const rawItems = await this.withAuthFallback(async (client) => {
      const q = searchParts.join(' ');
      const searchResp = await client.rest.search.issuesAndPullRequests({
        q,
        sort: 'updated',
        order: 'desc',
        per_page: perPage,
      });
      return searchResp.data.items;
    });

    const normalized: GitHubIssueSummary[] = [];
    for (const item of rawItems) {
      if ('pull_request' in item && item.pull_request) {
        continue;
      }

      const labelNames = (item.labels ?? [])
        .map((label) => (typeof label === 'string' ? label : label.name ?? ''))
        .filter((name) => name.length > 0);

      const lowerLabels = labelNames.map((l) => l.toLowerCase());
      if (lowerLabels.some((l) => excludeLabels.has(l))) {
        continue;
      }

      const matchingBugLabels = lowerLabels.filter((l) =>
        bugLabels.some((bugLabel) => l === bugLabel || l.includes('bug')),
      );

      let score = matchingBugLabels.length * 10;
      if (item.comments > 0) score += Math.min(item.comments, 5);
      if ((item.body ?? '').length > 120) score += 3;

      normalized.push({
        number: item.number,
        title: item.title,
        body: item.body ?? '',
        state: item.state,
        labels: labelNames,
        url: item.url,
        htmlUrl: item.html_url,
        author: item.user?.login ?? 'unknown',
        createdAt: item.created_at,
        updatedAt: item.updated_at,
        comments: item.comments,
        score,
      });
    }

    if (options.preferLabeled !== false) {
      return this.preferLabeledIssues(normalized, bugLabels);
    }

    return normalized;
  }

  preferLabeledIssues(
    issues: GitHubIssueSummary[],
    preferredLabels: string[] = DEFAULT_BUG_LABELS,
  ): GitHubIssueSummary[] {
    const preferredSet = new Set(preferredLabels.map((l) => l.toLowerCase()));

    return [...issues].sort((a, b) => {
      const aHasBugLabel = a.labels.some((l) => {
        const lower = l.toLowerCase();
        return preferredSet.has(lower) || lower.includes('bug');
      });
      const bHasBugLabel = b.labels.some((l) => {
        const lower = l.toLowerCase();
        return preferredSet.has(lower) || lower.includes('bug');
      });

      if (aHasBugLabel !== bHasBugLabel) {
        return aHasBugLabel ? -1 : 1;
      }
      if (a.score !== b.score) {
        return b.score - a.score;
      }
      return b.updatedAt.localeCompare(a.updatedAt);
    });
  }

  async cloneRepo(
    owner: string,
    repo: string,
    targetDir: string,
    options: CloneRepoOptions = {},
  ): Promise<CloneRepoResult> {
    const resolvedDir = path.resolve(targetDir);
    const gitDir = path.join(resolvedDir, '.git');
    let exists = false;
    try {
      await fs.access(gitDir);
      exists = true;
    } catch {
      exists = false;
    }

    const remoteUrl =
      this.token && !this.fellBackToUnauthenticated
        ? `https://x-access-token:${this.token}@github.com/${owner}/${repo}.git`
        : `https://github.com/${owner}/${repo}.git`;

    if (!exists) {
      await fs.mkdir(path.dirname(resolvedDir), { recursive: true });
      const cloneArgs: string[] = [];
      if (options.depth && options.depth > 0) {
        cloneArgs.push('--depth', String(options.depth));
      }
      if (options.branch) {
        cloneArgs.push('--branch', options.branch);
      }

      const rootGit = this.gitFactory();
      await rootGit.clone(remoteUrl, resolvedDir, cloneArgs);
    } else if (options.pullIfExists) {
      const repoGit = this.gitFactory(resolvedDir);
      await repoGit.fetch(['origin']);
      if (options.branch) {
        await repoGit.checkout(options.branch);
      }
      await repoGit.pull();
    }

    const repoGit = this.gitFactory(resolvedDir);
    const branchSummary = await repoGit.branchLocal();
    const commitSha = (await repoGit.revparse(['HEAD'])).trim();
    const defaultBranch =
      options.branch ?? (branchSummary.current || 'main');

    return {
      localPath: resolvedDir,
      defaultBranch,
      commitSha,
      clonedFresh: !exists,
    };
  }

  async forkRepo(
    owner: string,
    repo: string,
    organization?: string,
  ): Promise<ForkRepoResult> {
    if (!this.token || this.fellBackToUnauthenticated) {
      throw new Error('Forking a repository requires a valid GITHUB_TOKEN');
    }

    const { data } = await this.primaryClient.rest.repos.createFork({
      owner,
      repo,
      organization,
    });

    return {
      owner: data.owner.login,
      repo: data.name,
      fullName: data.full_name,
      htmlUrl: data.html_url,
      cloneUrl: data.clone_url,
      defaultBranch: data.default_branch,
    };
  }

  async getPRDiff(
    owner: string,
    repo: string,
    pullNumber: number,
  ): Promise<string> {
    return this.withAuthFallback(async (client) => {
      const response = await client.rest.pulls.get({
        owner,
        repo,
        pull_number: pullNumber,
        mediaType: {
          format: 'diff',
        },
      });
      return typeof response.data === 'string'
        ? response.data
        : String(response.data ?? '');
    });
  }

  async getPRFiles(
    owner: string,
    repo: string,
    pullNumber: number,
  ): Promise<PRFileChange[]> {
    return this.withAuthFallback(async (client) => {
      const { data } = await client.rest.pulls.listFiles({
        owner,
        repo,
        pull_number: pullNumber,
        per_page: 100,
      });

      return data.map((file) => ({
        filename: file.filename,
        status: file.status,
        additions: file.additions,
        deletions: file.deletions,
        changes: file.changes,
        patch: file.patch,
        blobUrl: file.blob_url,
      }));
    });
  }

  async getTemplates(
    owner: string,
    repo: string,
    ref?: string,
  ): Promise<RepoTemplatesResult> {
    const issueTemplates: RepoTemplateFile[] = [];
    const prTemplates: RepoTemplateFile[] = [];

    const issueDirFiles = await this.readDirectoryFiles(
      owner,
      repo,
      '.github/ISSUE_TEMPLATE',
      ref,
    );
    issueTemplates.push(...issueDirFiles);

    for (const candidate of [
      '.github/ISSUE_TEMPLATE.md',
      'ISSUE_TEMPLATE.md',
    ]) {
      const file = await this.readSingleFile(owner, repo, candidate, ref);
      if (file) issueTemplates.push(file);
    }

    const prDirFiles = await this.readDirectoryFiles(
      owner,
      repo,
      '.github/PULL_REQUEST_TEMPLATE',
      ref,
    );
    prTemplates.push(...prDirFiles);

    for (const candidate of [
      '.github/PULL_REQUEST_TEMPLATE.md',
      '.github/pull_request_template.md',
      'PULL_REQUEST_TEMPLATE.md',
      'pull_request_template.md',
    ]) {
      const file = await this.readSingleFile(owner, repo, candidate, ref);
      if (file) prTemplates.push(file);
    }

    return { issueTemplates, prTemplates };
  }

  async getContributing(
    owner: string,
    repo: string,
    ref?: string,
  ): Promise<ContributingFileResult | null> {
    const candidates = [
      'CONTRIBUTING.md',
      '.github/CONTRIBUTING.md',
      'docs/CONTRIBUTING.md',
      'contributing.md',
    ];

    for (const candidate of candidates) {
      const file = await this.readSingleFile(owner, repo, candidate, ref);
      if (file) {
        return {
          path: file.path,
          content: file.content,
        };
      }
    }

    return null;
  }

  private async readSingleFile(
    owner: string,
    repo: string,
    filePath: string,
    ref?: string,
  ): Promise<RepoTemplateFile | null> {
    try {
      return await this.withAuthFallback(async (client) => {
        const { data } = await client.rest.repos.getContent({
          owner,
          repo,
          path: filePath,
          ref,
        });

        if (Array.isArray(data) || !isRecord(data) || data.type !== 'file') {
          return null;
        }

        const rawContent =
          typeof data.content === 'string'
            ? decodeBase64Content(data.content)
            : '';

        return {
          path: typeof data.path === 'string' ? data.path : filePath,
          name:
            typeof data.name === 'string'
              ? data.name
              : path.posix.basename(filePath),
          content: rawContent,
        };
      });
    } catch (error) {
      if (getHttpStatus(error) === 404) {
        return null;
      }
      throw error;
    }
  }

  private async readDirectoryFiles(
    owner: string,
    repo: string,
    dirPath: string,
    ref?: string,
  ): Promise<RepoTemplateFile[]> {
    try {
      const entries = await this.withAuthFallback(async (client) => {
        const { data } = await client.rest.repos.getContent({
          owner,
          repo,
          path: dirPath,
          ref,
        });
        return Array.isArray(data) ? data : [];
      });

      const files: RepoTemplateFile[] = [];
      for (const entry of entries) {
        if (!isRecord(entry) || entry.type !== 'file' || typeof entry.path !== 'string') {
          continue;
        }
        const loaded = await this.readSingleFile(owner, repo, entry.path, ref);
        if (loaded) {
          files.push(loaded);
        }
      }
      return files;
    } catch (error) {
      if (getHttpStatus(error) === 404) {
        return [];
      }
      throw error;
    }
  }
}
