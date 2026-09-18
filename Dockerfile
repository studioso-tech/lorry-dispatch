# 石油ローリー配車エージェント — Cloud Run 用イメージ
#
# 設計上の注意:
#   - ベースは python:3.12-slim。開発機は 3.14 だが、コンテナは実績のある
#     3.12 に寄せる。ortools は cp312 / cp314 の両方に wheel があるので
#     どちらでも入るが、**本番で初めて動く組み合わせを増やさない**。
#   - .venv / tests / scripts は入れない（.dockerignore で除外）。
#   - CP-SAT は vCPU を使い切る。Cloud Run のインスタンスに割り当てた
#     vCPU 数と SOLVE_WORKERS を揃えること。ずれると、割り当て以上の
#     ワーカーが互いを待って遅くなる。

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 依存を先に入れてレイヤーキャッシュを効かせる
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY core/ ./core/
# 画面が読むデモ用のFAX画像（data/ の他のファイルは入れない）
COPY data/demo/ ./data/demo/

# Cloud Run は PORT を渡してくる。ローカルでは 8080 を既定にする。
ENV PORT=8080

CMD exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --timeout-graceful-shutdown 5
