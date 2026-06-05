export interface RepoView {
  id: number;
  owner: string;
  name: string;
  fullName: string;
  addedAt: string | null;
  lastSyncedAt: string | null;
}

export interface Label {
  name: string;
  color: string;
}

export interface IssueCounts {
  open: number;
  closed: number;
  total: number;
}

export interface PrCounts {
  open: number;
  closed: number;
  merged: number;
  draft: number;
  pendingReview: number;
  total: number;
}

export interface Contributor {
  login: string;
  commits: number;
  prsOpened: number;
  issuesOpened: number;
  reviews: number;
  total: number;
}

export interface RepoSummary {
  fullName: string;
  openIssues: number;
  openPrs: number;
  lastSyncedAt: string | null;
}

export interface Dashboard {
  repoCount: number;
  issues: IssueCounts;
  prs: PrCounts;
  contributors: Contributor[];
  repos: RepoSummary[];
}

export interface IssueListItem {
  id: number;
  repo: string;
  number: number;
  title: string;
  author: string | null;
  state: string;
  commentsCount: number;
  labels: Label[];
  updatedAt: string | null;
  url: string;
}

export interface IssueDetail {
  id: number;
  repo: string;
  number: number;
  title: string;
  author: string | null;
  state: string;
  body: string | null;
  commentsCount: number;
  assignees: string[];
  labels: Label[];
  createdAt: string | null;
  updatedAt: string | null;
  closedAt: string | null;
  url: string;
}

export interface ActionItem {
  repo: string;
  type: string;
  number: number;
  title: string;
  author: string | null;
  url: string;
  ageDays: number;
  detail: string;
}

export interface ActionCenter {
  staleDays: number;
  prsAwaitingReview: ActionItem[];
  prsFailingCi: ActionItem[];
  prsOpenTooLong: ActionItem[];
  staleIssues: ActionItem[];
  unassignedIssues: ActionItem[];
}

export interface WeekCount {
  week: string;
  count: number;
}

export interface WeekPair {
  week: string;
  opened: number;
  closed: number;
}

export interface Bucket {
  label: string;
  count: number;
}

export interface InsightContributor {
  login: string;
  total: number;
}

export interface Insights {
  prThroughput: WeekCount[];
  medianTimeToMergeHours: number;
  avgTimeToMergeHours: number;
  issueFlow: WeekPair[];
  contributors: InsightContributor[];
  stalePrBuckets: Bucket[];
}

export interface SyncRepoResult {
  fullName: string;
  status: string;
  pullRequests: number;
  issues: number;
  error: string | null;
}

export interface SyncResponse {
  repos: number;
  results: SyncRepoResult[];
}
