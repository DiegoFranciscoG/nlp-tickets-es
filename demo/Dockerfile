# syntax=docker/dockerfile:1.7
FROM python:3.12.14-slim-bookworm
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py auth.py examples.py ./
COPY .streamlit ./.streamlit
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"]
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
