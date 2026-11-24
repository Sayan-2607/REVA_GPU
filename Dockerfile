# Code-only image (no LaTeX): runs the package, tests and experiments.
FROM python:3.12-slim
WORKDIR /rava
COPY . /rava
RUN pip install --no-cache-dir -e ".[test]"
CMD ["bash", "-lc", "pytest -q -m 'not slow' && rava example && rava check"]
