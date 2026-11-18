import { SQL } from "bun";
import type { EventRow, StatusCount, StepRow, WorkflowDetail, WorkflowFilter, WorkflowList, WorkflowRow } from "../src/types.ts";
import { decodeStored } from "./decode.ts";

export interface Db {
  list(filter: WorkflowFilter): Promise<WorkflowList>;
  detail(id: string): Promise<WorkflowDetail | null>;
}

const LIMIT = 50;

interface WorkflowRecord {
  id: string;
  name: string | null;
  status: string;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  executor_id: string | null;
  application_version: string | null;
  recovery_attempts: string | null;
  queue_name: string | null;
  steps: string;
}

interface DetailRecord extends WorkflowRecord {
  serialization: string | null;
  inputs: string | null;
  output: string | null;
  error: string | null;
}

interface StepRecord {
  function_id: number;
  function_name: string;
  output: string | null;
  error: string | null;
  serialization: string | null;
  child_workflow_id: string | null;
  started_at_epoch_ms: string | null;
  completed_at_epoch_ms: string | null;
}

interface EventRecord {
  function_id: number | null;
  key: string;
  value: string;
  serialization: string | null;
}

const toNumber = (value: string | null): number | null => (value === null ? null : Number(value));

function toWorkflow(record: WorkflowRecord): WorkflowRow {
  return {
    id: record.id,
    name: record.name ?? "",
    status: record.status,
    created_at: Number(record.created_at),
    updated_at: Number(record.updated_at),
    completed_at: toNumber(record.completed_at),
    executor: record.executor_id ?? "",
    app_version: record.application_version ?? "",
    recovery_attempts: Number(record.recovery_attempts ?? 0),
    queue: record.queue_name ?? "",
    steps: Number(record.steps),
  };
}

function toStep(record: StepRecord, fallback: string | null): StepRow {
  const format = record.serialization ?? fallback;
  return {
    id: record.function_id,
    name: record.function_name,
    output: decodeStored(record.output, format),
    error: decodeStored(record.error, format),
    child_workflow: record.child_workflow_id ?? "",
    started_at: toNumber(record.started_at_epoch_ms),
    completed_at: toNumber(record.completed_at_epoch_ms),
  };
}

function toEvent(record: EventRecord, fallback: string | null): EventRow {
  return { function_id: record.function_id, key: record.key, value: decodeStored(record.value, record.serialization ?? fallback) };
}

export function postgresDb(url: string): Db {
  const sql = new SQL(url);
  const columns = sql`
    s.workflow_uuid as id, s.name, s.status, s.created_at, s.updated_at, s.completed_at, s.executor_id,
    s.application_version, s.recovery_attempts, s.queue_name,
    (select count(*) from dbos.operation_outputs o where o.workflow_uuid = s.workflow_uuid and o.function_name not like 'DBOS.%') as steps`;

  return {
    async list(filter) {
      const records = (await sql`
        select ${columns} from dbos.workflow_status s
        where (${filter.status} = '' or s.status = ${filter.status})
          and (${filter.search} = '' or s.workflow_uuid ilike ${`%${filter.search}%`} or s.name ilike ${`%${filter.search}%`})
        order by s.created_at desc limit ${LIMIT}`) as WorkflowRecord[];
      const counts = (await sql`select status, count(*)::int as count from dbos.workflow_status group by status order by status`) as StatusCount[];
      return { counts, workflows: records.map(toWorkflow) };
    },

    async detail(id) {
      const [record] = (await sql`
        select ${columns}, s.serialization,
          coalesce(i.inputs, s.inputs) as inputs, coalesce(o.output, s.output) as output, coalesce(o.error, s.error) as error
        from dbos.workflow_status s
        left join dbos.workflow_input i on i.workflow_uuid = s.workflow_uuid
        left join dbos.workflow_output o on o.workflow_uuid = s.workflow_uuid
        where s.workflow_uuid = ${id}`) as DetailRecord[];
      if (!record) return null;
      const format = record.serialization;
      const steps = (await sql`
        select function_id, function_name, output, error, serialization, child_workflow_id, started_at_epoch_ms, completed_at_epoch_ms
        from dbos.operation_outputs where workflow_uuid = ${id} order by function_id`) as StepRecord[];
      const events = (await sql`
        select null::int as function_id, key, value, serialization from dbos.workflow_events where workflow_uuid = ${id} order by key`) as EventRecord[];
      const history = (await sql`
        select function_id, key, value, null as serialization from dbos.workflow_events_history where workflow_uuid = ${id} order by function_id`) as EventRecord[];
      return {
        workflow: toWorkflow(record),
        serialization: format ?? "none",
        input: decodeStored(record.inputs, format),
        output: decodeStored(record.output, format),
        error: decodeStored(record.error, format),
        steps: steps.map(step => toStep(step, format)),
        events: events.map(event => toEvent(event, format)),
        history: history.map(event => toEvent(event, format)),
      };
    },
  };
}
