import http.server
import socketserver
import os
import gzip
import signal


class GzipAwareHandler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()

        if path.endswith('.gz'):
            self.send_response(200)
            self.send_header("Content-type", self.guess_type(path[:-3]))  # Remove .gz
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
            fs = os.fstat(os.open(path, os.O_RDONLY))
            self.send_header("Content-Length", str(fs[6]))
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()
            return open(path, 'rb')

        return super().send_head()

    def guess_type(self, path):
        base, ext = os.path.splitext(path)
        if ext == '.gz':
            base, ext = os.path.splitext(base)
        return super().guess_type(base + ext)


def shutdown_server(sig, frame):
    print("Closing server...")
    httpd.shutdown()


PORT = 8000

# Register SIGINT handler for Ctrl+C
signal.signal(signal.SIGINT, shutdown_server)

with socketserver.TCPServer(("", PORT), GzipAwareHandler) as httpd:
    print(f"Serving gzip-aware content at http://localhost:{PORT}")
    httpd.serve_forever()
