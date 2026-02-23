export interface ColumnInfo {
  table_name: string;
  column_name: string;
  data_type: string;
  is_nullable: string;
  column_default: string | null;
}

export interface PrimaryKeyInfo {
  table_name: string;
  column_name: string;
}

export interface TableSchema {
  name: string;
  columns: ColumnInfo[];
  primaryKeys: string[];
}
