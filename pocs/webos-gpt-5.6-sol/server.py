import html
import ipaddress
import json
import re
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

MAX_RESPONSE_SIZE = 5 * 1024 * 1024


def validate_url(value):
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Only public HTTP and HTTPS addresses are allowed")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    if not addresses:
        raise ValueError("The website address could not be resolved")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("Local and private network addresses are blocked")
    return parsed.geturl()


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        validate_url(new_url)
        return super().redirect_request(request, file_pointer, code, message, headers, new_url)


class LumaHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/api/health":
            payload = b'{"service":"lumaos","reader":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/api/browser":
            self.serve_browser(parsed)
            return
        super().do_GET()

    def serve_browser(self, parsed):
        query = urllib.parse.parse_qs(parsed.query)
        target = query.get("url", [""])[0]
        try:
            target = validate_url(target)
            request = urllib.request.Request(
                target,
                headers={
                    "User-Agent": "Mozilla/5.0 LumaOS/1.0",
                    "Accept": "text/html,application/xhtml+xml,image/*;q=0.8,*/*;q=0.5",
                    "Accept-Language": "en-US,en;q=0.9"
                }
            )
            opener = urllib.request.build_opener(SafeRedirectHandler())
            with opener.open(request, timeout=15) as response:
                final_url = validate_url(response.geturl())
                content_type = response.headers.get_content_type()
                charset = response.headers.get_content_charset() or "utf-8"
                payload = response.read(MAX_RESPONSE_SIZE + 1)
                if len(payload) > MAX_RESPONSE_SIZE:
                    raise ValueError("The website response is too large")
                if content_type in {"text/html", "application/xhtml+xml"}:
                    payload = self.prepare_html(payload, charset, final_url)
                    content_type = "text/html; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                self.wfile.write(payload)
        except (ValueError, urllib.error.URLError, urllib.error.HTTPError, socket.error) as error:
            self.send_browser_error(target, str(error))

    def prepare_html(self, payload, charset, final_url):
        text = payload.decode(charset, errors="replace")
        text = re.sub(r"<meta[^>]+http-equiv=[\"']?Content-Security-Policy[\"']?[^>]*>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<base[^>]*>", "", text, flags=re.IGNORECASE)
        bridge = """
<script>
document.addEventListener("click",function(event){var link=event.target.closest("a[href]");if(!link)return;event.preventDefault();parent.postMessage({lumaUrl:link.href},"*")},true);
document.addEventListener("submit",function(event){var form=event.target;if((form.method||"get").toLowerCase()!=="get"){event.preventDefault();parent.postMessage({lumaError:"Only safe GET forms are supported"},"*");return}event.preventDefault();var destination=new URL(form.action||location.href);new FormData(form).forEach(function(value,key){destination.searchParams.set(key,value)});parent.postMessage({lumaUrl:destination.href},"*")},true);
</script>
"""
        injection = f'<base href="{html.escape(final_url, quote=True)}">{bridge}'
        if re.search(r"<head[^>]*>", text, flags=re.IGNORECASE):
            text = re.sub(r"(<head[^>]*>)", lambda match: match.group(1) + injection, text, count=1, flags=re.IGNORECASE)
        else:
            text = injection + text
        return text.encode("utf-8")

    def send_browser_error(self, target, message):
        safe_target = html.escape(target or "Unknown address")
        safe_message = html.escape(message)
        payload = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Website unavailable</title>
<style>
body{{margin:0;padding:60px 30px;background:linear-gradient(145deg,#eef6ff,#fff);color:#263b55;text-align:center;font:14px Tahoma,sans-serif}}
h1{{color:#0b55b7;font:italic 800 28px "Trebuchet MS",sans-serif}}
p{{max-width:620px;margin:10px auto;line-height:1.55;overflow-wrap:anywhere}}
</style>
</head>
<body>
<h1>Website unavailable</h1>
<p>{safe_target}</p>
<p>{safe_message}</p>
</body>
</html>""".encode("utf-8")
        self.send_response(502)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


def main():
    port = int(sys.argv[1])
    directory = sys.argv[2]
    handler = lambda *args, **kwargs: LumaHandler(*args, directory=directory, **kwargs)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(json.dumps({"port": port}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
