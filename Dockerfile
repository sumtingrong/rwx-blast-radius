# Multi-stage build: build per-test coverage & AST maps in builder using exact pinned Python 3.13 + requirements.lock,
# then copy only runtime selector + split pickle maps + pinned sqlglot/tests source into final slim image.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS builder

WORKDIR /build
COPY requirements.lock /build/requirements.lock
RUN pip install --no-cache-dir -r /build/requirements.lock

RUN python -c "import urllib.request, tarfile, shutil; \
    url='https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz'; \
    urllib.request.urlretrieve(url, '/tmp/sg.tgz'); \
    tarfile.open('/tmp/sg.tgz').extractall('/tmp'); \
    shutil.move('/tmp/sqlglot-eb11aec4be8cb2122afe713a4816662c3f0fb2fa', '/build/sqlglot_ref')"

COPY build_maps.py /build/build_maps.py
RUN mkdir -p /build/out && python /build/build_maps.py /build/sqlglot_ref /build/out/maps.pkl

FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /build/out /app/
COPY --from=builder /build/sqlglot_ref/sqlglot /app/sqlglot_ref/sqlglot
COPY --from=builder /build/sqlglot_ref/tests /app/sqlglot_ref/tests
COPY selector.py /app/selector.py
COPY verifier.py /app/verifier.py
RUN chmod -R a+rX /app
CMD ["python", "/app/selector.py"]
