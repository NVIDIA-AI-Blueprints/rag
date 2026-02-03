# Build stage - OSRB approved Ubuntu base (Bug 3840915)
FROM nvcr.io/nvidia/base/ubuntu:jammy-20251013 AS builder

ARG DOWNLOAD_LEGAL_COMPLIANCE=false

# Snapshot packages BEFORE installing (conditional on DOWNLOAD_LEGAL_COMPLIANCE)
RUN if [ "$DOWNLOAD_LEGAL_COMPLIANCE" = "true" ]; then \
        apt list --installed 2>/dev/null | tail -n +2 | sed 's|/.*||' | sort -u > /tmp/packages-before.txt; \
    fi

# Install required OS dependencies for building
# Update all packages to address security vulnerabilities including CVE-2026-0861 (libc6)
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    curl \
    gnupg2 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js LTS and pnpm
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g pnpm

# Print versions for debugging
RUN node -v && pnpm -v

# Set working directory and copy license
WORKDIR /app
RUN mkdir -p /legal
COPY LICENSE-3rd-party.txt /legal/

# Download ONLY sources for packages WE installed (conditional on DOWNLOAD_LEGAL_COMPLIANCE)
RUN if [ "$DOWNLOAD_LEGAL_COMPLIANCE" = "true" ]; then \
        sed -i 's/# deb-src/deb-src/g' /etc/apt/sources.list && \
        apt-get update && mkdir -p /legal/source && \
        cd /legal/source && \
        apt list --installed 2>/dev/null | tail -n +2 | sed 's|/.*||' | sort -u > /tmp/packages-after.txt && \
        comm -13 /tmp/packages-before.txt /tmp/packages-after.txt > /legal/source/packages-manifest.txt && \
        echo "Downloading sources for $(wc -l < /legal/source/packages-manifest.txt) newly installed packages (out of $(wc -l < /tmp/packages-after.txt) total)" && \
        cat /legal/source/packages-manifest.txt | \
        xargs -n 1 bash -c 'apt-get source --download-only "$0" 2>&1 || echo "Warning: Source not available for $0"' > /legal/source/download-log.txt 2>&1 && \
        echo "APT source download complete: $(du -sh /legal/source | cut -f1)"; \
    else \
        echo "Legal compliance download skipped (DOWNLOAD_LEGAL_COMPLIANCE=false)"; \
    fi

WORKDIR /app

# Environment variables for build
ARG NEXT_PUBLIC_MODEL_NAME
ARG VITE_EMBEDDING_MODEL
ARG VITE_RERANKER_MODEL
ARG VITE_API_CHAT_URL
ARG VITE_API_VDB_URL
ARG VITE_MODEL_NAME

ENV NEXT_PUBLIC_MODEL_NAME=${NEXT_PUBLIC_MODEL_NAME}
ENV VITE_EMBEDDING_MODEL=${VITE_EMBEDDING_MODEL}
ENV VITE_RERANKER_MODEL=${VITE_RERANKER_MODEL}
ENV VITE_API_CHAT_URL=${VITE_API_CHAT_URL}
ENV VITE_API_VDB_URL=${VITE_API_VDB_URL}
ENV VITE_MODEL_NAME=${VITE_MODEL_NAME}

# Copy frontend files to the working directory
WORKDIR /app/frontend
COPY . .

# Set environment variable to make pnpm non-interactive
ENV CI=true
ENV PNPM_HOME=/usr/local/bin
ENV npm_config_yes=true

# Build the application
RUN rm -rf node_modules && pnpm install --frozen-lockfile
RUN pnpm add -D @types/node
RUN pnpm run build

# Prepare legal compliance files for production stage (conditional on DOWNLOAD_LEGAL_COMPLIANCE)
RUN if [ "$DOWNLOAD_LEGAL_COMPLIANCE" = "true" ] && [ -d /legal ]; then \
        cd / && \
        echo "Compressing legal directory..." && \
        tar -czf /legal.tar.gz legal/ && \
        ORIGINAL_SIZE=$(du -sh /legal | cut -f1) && \
        COMPRESSED_SIZE=$(du -sh /legal.tar.gz | cut -f1) && \
        rm -rf /legal && \
        mkdir -p /legal && \
        mv /legal.tar.gz /legal/ && \
        echo "Legal compliance archive created: $COMPRESSED_SIZE (original: $ORIGINAL_SIZE)"; \
    else \
        mkdir -p /legal && \
        touch /legal/.placeholder && \
        echo "No legal compliance files (DOWNLOAD_LEGAL_COMPLIANCE=false)"; \
    fi

# Production stage - NVIDIA distroless (pre-approved)
# Updated to latest version to address CVE-2025-9230 (libssl3)
FROM nvcr.io/nvidia/distroless/node:24-v3.1.3

# Copy built application and config for production preview
WORKDIR /app/frontend
# Copy legal compliance directory from builder stage (always copy, will contain placeholder or actual files)
COPY --from=builder /legal /legal
COPY --from=builder /app/frontend/dist ./dist
COPY --from=builder /app/frontend/package.json ./package.json
COPY --from=builder /app/frontend/vite.config.ts ./vite.config.ts
COPY --from=builder /app/frontend/node_modules ./node_modules

# Expose port
EXPOSE 3000

# Use simple static server + proxy approach
# Routes to rag-server (CHAT): /api/generate, /api/summary, /api/configuration
# Routes to ingestor-server (VDB): all other /api/* endpoints
CMD ["node", "-e", "const http=require('http');const fs=require('fs');const path=require('path');const {URL}=require('url');const CHAT=process.env.VITE_API_CHAT_URL||'http://rag-server:8081/v1';const VDB=process.env.VITE_API_VDB_URL||'http://ingestor-server:8082/v1';const chatRoutes=['/api/generate','/api/summary','/api/configuration'];const server=http.createServer((req,res)=>{const isChatRoute=chatRoutes.some(r=>req.url.startsWith(r));if(req.url.startsWith('/api/')){const u=new URL(isChatRoute?CHAT:VDB);const opt={hostname:u.hostname,port:u.port,path:req.url.replace('/api',''),method:req.method,headers:{...req.headers,host:u.host}};const pr=http.request(opt,r=>{res.writeHead(r.statusCode,r.headers);r.pipe(res)});pr.on('error',e=>{res.writeHead(500);res.end('Error:'+e.message)});req.pipe(pr)}else{let f=req.url==='/'?'/index.html':req.url;f=path.join('./dist',f);if(!fs.existsSync(f)&&!f.includes('.'))f=path.join('./dist','index.html');fs.readFile(f,(e,d)=>{if(e){res.writeHead(404);res.end('Not found')}else{res.writeHead(200,{'Content-Type':f.endsWith('.html')?'text/html':f.endsWith('.js')?'application/javascript':f.endsWith('.css')?'text/css':'text/plain'});res.end(d)}})}});server.listen(3000,'0.0.0.0',()=>console.log('Server running on port 3000'));"]