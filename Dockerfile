# Stage 1: Builder Base - Installs common development tools and Python/Node.js
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder-base
LABEL stage=builder-base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    build-essential \
    cmake \
    ffmpeg \
    curl \
    # For cairosvg (system libraries often improve stability)
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf2.0-dev && \
    # Node.js and npm (using NodeSource LTS distributions)
    curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#-----------------------------------------------------------------------------------
# Stage 2: Python Builder - Builds Python dependencies and custom packages
FROM builder-base AS python-builder
LABEL stage=python-builder

# Install PyTorch
# Using PyTorch 2.3.1 as a stable alternative for CUDA 11.8. Adjust if 2.5.1 is strictly needed and available.
RUN python3 -m pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Install segment-anything from GitHub
RUN python3 -m pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

WORKDIR /app

# Install root Python requirements
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Install diffvg
RUN git clone https://github.com/BachiLi/diffvg.git /tmp/diffvg && \
    cd /tmp/diffvg && \
    git submodule update --init --recursive && \
    python3 -m pip install --no-cache-dir svgwrite svgpathtools cssutils torch-tools && \
    python3 setup.py install && \
    rm -rf /tmp/diffvg

# Install picosvg
RUN (git clone git@github.com:googlefonts/picosvg.git /tmp/picosvg || \
     git clone https://github.com/googlefonts/picosvg.git /tmp/picosvg) && \
    cd /tmp/picosvg && \
    python3 -m pip install --no-cache-dir -e . && \
    rm -rf /tmp/picosvg

# Install backend Python requirements (from web/backend)
COPY web/backend/requirements.txt /app/web/backend/requirements.txt
RUN if [ -f /app/web/backend/requirements.txt ]; then \
        python3 -m pip install --no-cache-dir -r /app/web/backend/requirements.txt; \
    else \
        echo "Warning: web/backend/requirements.txt not found. Skipping backend-specific pip install."; \
    fi

# DEBUG COMMANDS TO HELP IDENTIFY SITE-PACKAGES LOCATION
RUN echo "DEBUG: Listing potential site-packages locations in python-builder stage"
RUN echo "Listing /usr/local/lib/python3.10/site-packages:" && (ls -lA /usr/local/lib/python3.10/site-packages || echo "Not found or empty")
RUN echo "Listing /usr/lib/python3.10/site-packages:" && (ls -lA /usr/lib/python3.10/site-packages || echo "Not found or empty")
RUN echo "Listing /root/.local/lib/python3.10/site-packages:" && (ls -lA /root/.local/lib/python3.10/site-packages || echo "Not found or empty")
RUN echo "Listing /usr/lib/python3/dist-packages:" && (ls -lA /usr/lib/python3/dist-packages || echo "Not found or empty") # Often the case for apt-installed Python
RUN echo "Pip list in python-builder:" && python3 -m pip list
RUN echo "Which pip3:" && which pip3
RUN echo "Which python3:" && which python3

#-----------------------------------------------------------------------------------
# Stage 3: Frontend Builder - Builds Node.js dependencies
FROM builder-base AS frontend-builder
LABEL stage=frontend-builder

WORKDIR /app/web
COPY web/package.json web/package-lock.json* ./
RUN npm install
# If there's a build step for the frontend (e.g., npm run build), run it here.
# RUN npm run build # Example if you have a build script

#-----------------------------------------------------------------------------------
# Stage 4: Final Runtime Image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.local/bin:${PATH}"

# Install minimal runtime system dependencies (Python, Node.js, ffmpeg)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \ 
    curl \ 
    ca-certificates && \
    # Install Node.js runtime
    curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    # Install critical Python packages directly in the runtime stage
    python3 -m pip install --no-cache-dir uvicorn fastapi python-dotenv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from python-builder stage
# We're now using the correct paths based on the sys.path output
COPY --from=python-builder /usr/local/lib/python3.10/dist-packages/ /usr/local/lib/python3.10/dist-packages/
COPY --from=python-builder /usr/lib/python3/dist-packages/ /usr/lib/python3/dist-packages/
# Removed the /root/.local path because it doesn't exist in the builder stage
COPY --from=python-builder /usr/local/bin/ /usr/local/bin/

# Copy frontend node_modules (and build artifacts if any) from frontend-builder stage
COPY --from=frontend-builder /app/web/node_modules /app/web/node_modules
# If you had an `npm run build` step in frontend-builder, copy its output:
# COPY --from=frontend-builder /app/web/dist /app/web/dist # Example

# Copy application code (do this after dependencies to leverage caching)
COPY . .

# Ensure scripts are executable and set up startup script
COPY <<EOF /app/start-services.sh
#!/bin/bash
set -m # Enable Job Control

# Load .env variables if the file exists in /app/.env
if [ -f /app/.env ]; then
  echo "Loading environment variables from /app/.env"
  export \$(grep -v '^#' /app/.env | xargs)
else
  echo "Warning: /app/.env file not found. API keys and other configurations might be missing."
fi

# Check for essential API key
if [ -z "\$OPENAI_API_KEY" ]; then
  echo "ERROR: OPENAI_API_KEY is not set. Please provide it in the .env file mounted to /app/.env"
  # exit 1 # Consider exiting if critical
fi

echo "Starting Python backend (Uvicorn) on port 8000..."
cd /app/web/backend
# Ensure uvicorn is executable if copied from /usr/local/bin or ensure PATH is correct
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=\$!

echo "Starting Node.js frontend (npm start) on port 3000..."
cd /app/web
npm start &
FRONTEND_PID=\$!

echo "Backend PID: \$BACKEND_PID, Frontend PID: \$FRONTEND_PID"
echo "Access web UI at http://localhost:3000 (if ports are mapped)"
echo "Backend API at http://localhost:8000 (if ports are mapped)"

wait -n
kill \$BACKEND_PID \$FRONTEND_PID 2>/dev/null
wait
EOF
RUN chmod +x /app/start-services.sh

# Expose ports
EXPOSE 8000
EXPOSE 3000

# Environment variables for NVIDIA runtime - fixed format
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["/app/start-services.sh"]