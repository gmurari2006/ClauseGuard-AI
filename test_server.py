"""
ClauseGuard AI - In-Process HTTP Server Test
Runs FastAPI and verifies all endpoints directly.
"""
import threading
import time
import urllib.request
import json
import uvicorn
from main import app

class ServerThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.config = uvicorn.Config(app, host="127.0.0.1", port=8001, log_level="warning")
        self.server = uvicorn.Server(self.config)
        self.daemon = True

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True

def test_endpoints():
    server = ServerThread()
    server.start()
    time.sleep(2)

    try:
        # 1. Health check
        with urllib.request.urlopen("http://127.0.0.1:8001/api/health") as resp:
            data = json.loads(resp.read().decode())
            print("Health endpoint response:", data)
            assert data["status"] == "healthy"
            assert data["ml_model_ready"] is True

        # 2. Samples list
        with urllib.request.urlopen("http://127.0.0.1:8001/api/samples") as resp:
            data = json.loads(resp.read().decode())
            print(f"Loaded {len(data['samples'])} sample contracts via HTTP")
            assert len(data["samples"]) == 5

        # 3. Load sample agreement via HTTP
        with urllib.request.urlopen("http://127.0.0.1:8001/api/samples/sample_internship") as resp:
            data = json.loads(resp.read().decode())
            print(f"Sample internship risk score: {data['risk_score']}/100, clauses: {len(data['clauses'])}")
            assert data["risk_score"] > 70
            assert len(data["clauses"]) > 5

        # 4. Q&A via HTTP
        req_data = json.dumps({"query": "What happens if I quit?", "document_id": data["document_id"]}).encode()
        req = urllib.request.Request("http://127.0.0.1:8001/api/qa", data=req_data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            qa_res = json.loads(resp.read().decode())
            print(f"Q&A response status: {qa_res['risk_status']}, citations: {len(qa_res['cited_clauses'])}")
            assert len(qa_res["cited_clauses"]) > 0

        # 5. Frontend static HTML via HTTP
        with urllib.request.urlopen("http://127.0.0.1:8001/") as resp:
            html = resp.read().decode()
            print(f"Frontend served successfully! ({len(html)} bytes)")
            assert "ClauseGuard AI" in html
            assert "Live Insights" in html

        print("\nALL HTTP ENDPOINTS AND FRONTEND ROUTES VERIFIED!")

    finally:
        server.stop()

if __name__ == "__main__":
    test_endpoints()
