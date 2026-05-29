"""
Expose the Streamlit app on the public internet.

Usage:
    python run_public.py

Requires: streamlit, and either pyngrok (`pip install pyngrok`) or the ngrok CLI on PATH.

University / library firewalls often block ngrok's default tunnels. This script tries:
  1. ngrok HTTP tunnel with region ap (Asia-Pacific) or in (India — close to Pakistan)
  2. ngrok agent connectivity via port 443, then 80 (common allowed outbound ports)
  3. ngrok TCP tunnel as a last resort

If ngrok is fully blocked, use one of these (no install; needs OpenSSH client on Windows 10+):

  # localhost.run — free HTTPS URL
  ssh -R 80:localhost:8501 nokey@localhost.run

  # serveo.net — same idea, different host
  ssh -R 80:localhost:8501 serveo.net

Run Streamlit locally first (or let this script start it):
  streamlit run app.py --server.port 8501
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time

STREAMLIT_PORT = 8501
# Regions near Pakistan first
NGROK_REGIONS = ("ap", "in", "eu", "us")


def start_streamlit(port: int) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(port),
        "--server.headless", "true",
    ]
    print(f"Starting Streamlit: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def try_pyngrok(port: int) -> str | None:
    try:
        from pyngrok import conf, ngrok
    except ImportError:
        print("pyngrok not installed — skip (pip install pyngrok) or use ngrok CLI.")
        return None

    for region in NGROK_REGIONS:
        try:
            conf.get_default().region = region
            # ngrok v3 agent uses HTTPS (port 443) to reach ngrok cloud by default
            tunnel = ngrok.connect(str(port), "http")
            url = tunnel.public_url
            print(f"pyngrok OK [region={region}]: {url}")
            return url
        except Exception as exc:
            print(f"pyngrok failed [region={region}]: {exc}")
    return None


def try_ngrok_cli(port: int) -> str | None:
    ngrok_bin = shutil.which("ngrok")
    if not ngrok_bin:
        print("ngrok CLI not found on PATH.")
        return None

    attempts = []
    for region in NGROK_REGIONS:
        attempts.append(
            [ngrok_bin, "http", str(port), f"--region={region}", "--log=stdout"]
        )
    # Some campuses only allow outbound 443/80 to the ngrok agent
    for region in NGROK_REGIONS:
        attempts.append(
            [ngrok_bin, "http", str(port), f"--region={region}",
             "--log=stdout", "--scheme=https"]
        )

    for cmd in attempts:
        print(f"Trying: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            time.sleep(4)
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                print(out or "ngrok exited immediately.")
                continue
            print("ngrok CLI started — open http://127.0.0.1:4040 for the public URL.")
            return "http://127.0.0.1:4040 (inspect ngrok web UI)"
        except Exception as exc:
            print(f"ngrok CLI error: {exc}")
    return None


def try_ngrok_tcp_cli(port: int) -> str | None:
    ngrok_bin = shutil.which("ngrok")
    if not ngrok_bin:
        return None
    for region in NGROK_REGIONS:
        cmd = [ngrok_bin, "tcp", str(port), f"--region={region}", "--log=stdout"]
        print(f"Trying TCP tunnel: {' '.join(cmd)}")
        try:
            subprocess.Popen(cmd)
            time.sleep(3)
            print("TCP tunnel may be up — check http://127.0.0.1:4040")
            return "tcp://127.0.0.1:4040"
        except Exception as exc:
            print(f"TCP tunnel failed: {exc}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Run Streamlit + public tunnel")
    parser.add_argument("--port", type=int, default=STREAMLIT_PORT)
    parser.add_argument("--no-streamlit", action="store_true",
                        help="Only start tunnel; assume Streamlit is already running")
    args = parser.parse_args()

    proc = None
    if not args.no_streamlit:
        proc = start_streamlit(args.port)
        time.sleep(3)

    url = try_pyngrok(args.port) or try_ngrok_cli(args.port) or try_ngrok_tcp_cli(args.port)

    if url:
        print("\n" + "=" * 60)
        print("Public access:")
        print(f"  {url}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("All ngrok attempts failed (firewall may block ngrok).")
        print("Alternatives (run in a second terminal while Streamlit is on port 8501):")
        print("  ssh -R 80:localhost:8501 nokey@localhost.run")
        print("  ssh -R 80:localhost:8501 serveo.net")
        print("=" * 60)

    print("Press Ctrl+C to stop.")
    try:
        if proc:
            proc.wait()
        else:
            while True:
                time.sleep(60)
    except KeyboardInterrupt:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
