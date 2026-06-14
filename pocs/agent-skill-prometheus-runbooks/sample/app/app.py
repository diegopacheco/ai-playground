from http.server import BaseHTTPRequestHandler, HTTPServer

METRICS = "\n".join(
    [
        "# HELP app_error_rate Current request error ratio",
        "# TYPE app_error_rate gauge",
        "app_error_rate 0.95",
        "# HELP app_memory_usage_ratio Memory used over limit",
        "# TYPE app_memory_usage_ratio gauge",
        "app_memory_usage_ratio 0.92",
        "# HELP app_request_latency_seconds Last request latency",
        "# TYPE app_request_latency_seconds gauge",
        "app_request_latency_seconds 1.8",
        "# HELP app_up Application liveness",
        "# TYPE app_up gauge",
        "app_up 1",
        "",
    ]
).encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(METRICS)))
            self.end_headers()
            self.wfile.write(METRICS)
        else:
            body = b"sample app: metrics at /metrics\n"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
