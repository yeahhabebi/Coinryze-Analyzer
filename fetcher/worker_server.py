# fetcher/worker_server.py
import os
from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler

# import your existing fetch function
from fetcher.fetch_coinryze import fetch_and_save

app = FastAPI()
scheduler = BackgroundScheduler()

FETCH_INTERVAL_MIN = int(os.getenv("FETCH_INTERVAL_MIN", "1"))

# schedule the job
scheduler.add_job(fetch_and_save, "interval", minutes=FETCH_INTERVAL_MIN)
scheduler.start()
print(f"Scheduler started, will fetch every {FETCH_INTERVAL_MIN} minute(s)")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.on_event("shutdown")
def _shutdown():
    try:
        scheduler.shutdown(wait=False)
    except Exception:
        pass
