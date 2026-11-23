export type Finding = {
  readonly file: string;
  readonly rule: string;
  readonly message: string;
  readonly line: number;
};

export type Expectation =
  | { readonly expect: "caught"; readonly rule: string }
  | { readonly expect: "missed"; readonly gap: string };

export type Sample = {
  readonly file: string;
  readonly title: string;
  readonly slop: string;
} & Expectation;

export type Verdict = "caught" | "missed";

export type SampleResult = Sample & {
  readonly verdict: Verdict;
  readonly asExpected: boolean;
  readonly findings: readonly Finding[];
  readonly source: string;
};

export type RuleCoverage = { readonly rule: string; readonly hits: number };

export type Report = {
  readonly samples: readonly SampleResult[];
  readonly caught: number;
  readonly missed: number;
  readonly surprises: number;
  readonly rules: readonly RuleCoverage[];
};

export interface Linter {
  run(): Promise<readonly Finding[]>;
}

export interface SourceReader {
  read(file: string): Promise<string>;
}
