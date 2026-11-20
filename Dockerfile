# PhysGen HTTP 服务镜像
#
# 轻量镜像（仅基础依赖 + server，无 GPU 物理）:
#   docker build -t physgen .
# 含真实物理依赖（Genesis/torch，镜像较大）:
#   docker build --build-arg EXTRAS="server,real" -t physgen-gpu .
#
# 运行（config.yaml 与 .env 由挂载提供，产物落在挂载的 runs/）:
#   docker run -d -p 8000:8000 \
#     -v $(pwd)/config.yaml:/app/config.yaml:ro \
#     -v $(pwd)/.env:/app/.env:ro \
#     -v $(pwd)/runs:/app/runs \
#     -e PHYSGEN_SERVER_TOKEN=your-secret \
#     physgen
FROM python:3.12-slim

ARG EXTRAS=server
WORKDIR /app

COPY pyproject.toml README.md ./
COPY physgen ./physgen
COPY cli.py server.py ./
COPY config.yaml .env.example ./

RUN pip install --no-cache-dir -e ".[${EXTRAS}]"

EXPOSE 8000
ENV PHYSGEN_RUNS_DIR=/app/runs
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
