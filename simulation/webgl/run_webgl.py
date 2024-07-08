import http.server
import socketserver
import webbrowser
import threading

PORT = 8000


def run_server():
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()


threading.Thread(target=run_server, daemon=True).start()

webbrowser.open(f'http://localhost:{PORT}/index.html')

input("Press Enter to stop the server...")
