import argparse
import subprocess
import time

from pyngrok import ngrok, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authtoken", required=True, help="Ngrok authtoken dari dashboard.ngrok.com")
    parser.add_argument("--port", type=int, default=8501, help="Port Streamlit (default: 8501)")
    args = parser.parse_args()

    conf.get_default().auth_token = args.authtoken

    # Jalankan Streamlit sebagai proses background
    process = subprocess.Popen(
        [
            "streamlit", "run", "app.py",
            "--server.port", str(args.port),
            "--server.headless", "true",
        ]
    )

    time.sleep(5)  # beri waktu Streamlit untuk start

    public_url = ngrok.connect(args.port, "http")
    print("=" * 60)
    print(f"🚀 Web app kamu sudah bisa diakses publik di:")
    print(f"   {public_url}")
    print("=" * 60)
    print("Tekan CTRL+C untuk menghentikan.")

    try:
        process.wait()
    except KeyboardInterrupt:
        print("\nMenghentikan tunnel & server...")
        ngrok.disconnect(public_url)
        process.terminate()


if __name__ == "__main__":
    main()