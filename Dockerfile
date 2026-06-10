FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CMDSTAN=/root/.cmdstan \
    PORT_API=8000 \
    PORT_APP=7860

WORKDIR /app

# System deps for LightGBM and Prophet / CmdStan build
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ make libgomp1 libstdc++6 \
        wget curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CmdStan (required by Prophet)
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

# Copy project
COPY . .

# Pre-train and pickle models during build (bakes them into the image)
# Set SKIP_PRETRAIN=1 as build arg to skip (models must then exist in models/)
ARG SKIP_PRETRAIN=0
RUN if [ "$SKIP_PRETRAIN" = "0" ] && [ -f "data/delhi_ncr_darkstore_fruit_vegetable_sales_2022_2025.csv" ]; then \
        python scripts/train_and_pickle.py; \
    fi

RUN chmod +x start.sh

EXPOSE 7860

CMD ["bash", "start.sh"]