# backend/gunicorn.conf.py
workers = 2
worker_class = 'uvicorn.workers.UvicornWorker'
max_requests = 1000
max_requests_jitter = 50
timeout = 120
keepalive = 5
