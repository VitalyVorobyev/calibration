#!/usr/bin/env python3
"""
Simple HTTP server for serving Doxygen documentation locally.
This script starts a local web server to view the generated documentation.
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

def serve_docs(port=8080, open_browser=True):
    """Serve the Doxygen documentation on a local HTTP server."""

    # Find the documentation directory
    script_dir = Path(__file__).parent
    doc_dir = script_dir / "doc" / "doxygen" / "html"

    if not doc_dir.exists():
        print("❌ Error: Documentation not found!")
        print("   Please run './generate_docs.sh' first to generate documentation.")
        sys.exit(1)

    # Change to documentation directory
    os.chdir(doc_dir)

    # Create server
    handler = http.server.SimpleHTTPRequestHandler

    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"🌐 Serving documentation at http://localhost:{port}")
            print(f"📁 Document root: {doc_dir}")
            print("🔧 Press Ctrl+C to stop the server")

            if open_browser:
                webbrowser.open(f"http://localhost:{port}")

            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"❌ Error: Port {port} is already in use")
            print(f"   Try a different port: python3 serve_docs.py --port {port + 1}")
        else:
            print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Serve Calibration Library documentation")
    parser.add_argument("--port", "-p", type=int, default=8080,
                       help="Port to serve on (default: 8080)")
    parser.add_argument("--no-browser", action="store_true",
                       help="Don't automatically open browser")

    args = parser.parse_args()

    serve_docs(port=args.port, open_browser=not args.no_browser)
