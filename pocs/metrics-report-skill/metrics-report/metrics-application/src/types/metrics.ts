export type TestType = 'unit' | 'integration' | 'contract' | 'e2e' | 'css' | 'stress' | 'chaos' | 'mutation' | 'observability';

export const TEST_TYPES: TestType[] = ['unit', 'integration', 'contract', 'e2e', 'css', 'stress', 'chaos', 'mutation', 'observability'];

export type QualityRating = 'poor' | 'fair' | 'good' | 'excellent';

export interface Repository {
  name: string;
  url: string;
  branch: string;
  commit: string;
}

export interface FileStats {
  total: number;
  backend: number;
  frontend: number;
  testFiles: number;
  byExtension: Record<string, number>;
}

export interface TestCase {
  name: string;
  status: 'pass' | 'fail';
  duration: number;
  error?: string;
  line: number;
  githubUrl: string;
}

export interface TestFile {
  path: string;
  type: TestType;
  testCount: number;
  passing: number;
  failing: number;
  author: string;
  githubUrl: string;
  tests: TestCase[];
}

export interface TestTypeStats {
  total: number;
  passing: number;
  failing: number;
  duration: number;
  files: TestFile[];
}

export interface TestStats {
  total: number;
  passing: number;
  failing: number;
  byType: Record<TestType, TestTypeStats>;
}

export interface FailureDetail {
  test: string;
  type: TestType;
  file: string;
  line: number;
  error: string;
  stackTrace?: string;
  githubUrl: string;
  failureCount?: number;
}

export interface FailureStats {
  byType: Partial<Record<TestType, number>>;
  details: FailureDetail[];
}

export interface FileCoverage {
  file: string;
  layer: 'backend' | 'frontend';
  githubUrl: string;
  coverage: Partial<Record<TestType, { tool: number | null; llm: boolean }>>;
}

export interface CoverageStats {
  backend: FileCoverage[];
  frontend: FileCoverage[];
}

export interface AuthorStats {
  unit: number;
  integration: number;
  contract: number;
  e2e: number;
  css: number;
  stress: number;
  chaos: number;
  mutation: number;
  observability: number;
  total: number;
}

export interface QualityEvaluation {
  rating: QualityRating;
  justification: string;
}

export interface QualityStats {
  byType: Partial<Record<TestType, QualityEvaluation>>;
}

export interface ScoreBreakdown {
  coverageBreadth: number;
  typeDiversity: number;
  passRate: number;
  testQuality: number;
  codeToTestRatio: number;
}

export interface ScoreStats {
  total: number;
  breakdown: ScoreBreakdown;
}

export interface MetricsReport {
  timestamp: string;
  repository: Repository;
  stacks: string[];
  files: FileStats;
  tests: TestStats;
  failures: FailureStats;
  coverage: CoverageStats;
  authors: Record<string, AuthorStats>;
  quality: QualityStats;
  score: ScoreStats;
}
