DROP POLICY IF EXISTS kv_read_all ON kv;
CREATE POLICY kv_read_all ON kv FOR SELECT
  USING (current_setting('app.read_all', true) = 'on');
