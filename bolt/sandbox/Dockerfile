# Image used to execute generated code. Deliberately minimal: the sandbox runs it
# with no network, so anything needed must be baked in here.
FROM python:3.13-slim

RUN pip install --no-cache-dir pytest==8.3.4 pytest-timeout==2.3.1

WORKDIR /work
CMD ["pytest", "-q"]
