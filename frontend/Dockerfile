# Build stage - OSRB approved Ubuntu base (Bug 3840915)
FROM nvcr.io/nvidia/base/ubuntu:jammy-20250415.1 AS builder

# Install required OS dependencies for building
RUN apt-get update && apt-get install -y \
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

# Production stage - NVIDIA distroless (pre-approved)
FROM nvcr.io/nvidia/distroless/node:24-v3.0.2

# Copy built application and config for production preview
WORKDIR /app/frontend
COPY --from=builder /legal/LICENSE-3rd-party.txt /legal/LICENSE-3rd-party.txt
COPY --from=builder /app/frontend/dist ./dist
COPY --from=builder /app/frontend/package.json ./package.json
COPY --from=builder /app/frontend/vite.config.ts ./vite.config.ts
COPY --from=builder /app/frontend/node_modules ./node_modules

# Expose port
EXPOSE 3000

# Use simple static server + proxy approach
CMD ["node", "-e", "const http=require('http');const fs=require('fs');const path=require('path');const {URL}=require('url');const CHAT=process.env.VITE_API_CHAT_URL||'http://rag-server:8081/v1';const VDB=process.env.VITE_API_VDB_URL||'http://ingestor-server:8082/v1';const server=http.createServer((req,res)=>{if(req.url.startsWith('/api/generate')){const u=new URL(CHAT);const opt={hostname:u.hostname,port:u.port,path:req.url.replace('/api',''),method:req.method,headers:{...req.headers,host:u.host}};const pr=http.request(opt,r=>{res.writeHead(r.statusCode,r.headers);r.pipe(res)});pr.on('error',e=>{res.writeHead(500);res.end('Error:'+e.message)});req.pipe(pr)}else if(req.url.startsWith('/api/')){const u=new URL(VDB);const opt={hostname:u.hostname,port:u.port,path:req.url.replace('/api',''),method:req.method,headers:{...req.headers,host:u.host}};const pr=http.request(opt,r=>{res.writeHead(r.statusCode,r.headers);r.pipe(res)});pr.on('error',e=>{res.writeHead(500);res.end('Error:'+e.message)});req.pipe(pr)}else{let f=req.url==='/'?'/index.html':req.url;f=path.join('./dist',f);if(!fs.existsSync(f)&&!f.includes('.'))f=path.join('./dist','index.html');fs.readFile(f,(e,d)=>{if(e){res.writeHead(404);res.end('Not found')}else{res.writeHead(200,{'Content-Type':f.endsWith('.html')?'text/html':f.endsWith('.js')?'application/javascript':f.endsWith('.css')?'text/css':'text/plain'});res.end(d)}})}});server.listen(3000,'0.0.0.0',()=>console.log('Server running on port 3000'));"]