# No explicit Python version is pinned elsewhere in the project (requirements.txt
# only constrains package versions), so this defaults to a recent stable slim image
# per the milestone's fallback guidance.
FROM python:3.12-slim

# Avoids .pyc bloat in the image and keeps log output unbuffered so it shows up
# immediately in `docker compose logs` instead of waiting on a flush.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# A dedicated system account to run the app as, rather than root. WORKDIR
# created /app as root before this ran, and `COPY --chown` below only fixes
# ownership of the files it copies, not the pre-existing directory itself --
# chown it explicitly so appuser can write into its own working directory
# (e.g. any library that wants a local cache/state file there) at runtime.
RUN groupadd --system appuser \
    && useradd --system --gid appuser --home-dir /app --shell /usr/sbin/nologin appuser \
    && chown appuser:appuser /app

# Installed before the rest of the app code is copied in, so this layer (the
# slow part) is only rebuilt when requirements.txt itself changes, not on
# every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# Matches how app/main.py's own __main__ block starts uvicorn, except bound to
# 0.0.0.0 (rather than the default 127.0.0.1) so it's reachable from outside
# the container, and without --reload since this is not a dev server.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
