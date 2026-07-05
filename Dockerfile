# ---- build stage ----
FROM node:20-alpine AS build
WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# Vite bakes VITE_* env vars into the JS bundle at BUILD time, not at
# container start -- this is baked in here, unlike a backend env var read
# at runtime. If you need to point this build at a different API URL,
# rebuild the image with a different --build-arg, don't expect changing
# it at `docker run` time to do anything.
#
# Default matches the docker-compose setup: the browser (not the nginx
# container) calls the API, so this must be a host-reachable URL
# (localhost:8000, published by the api service), NOT the in-network
# service name "api" -- that hostname only resolves inside the compose
# network, not from your browser.
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# ---- serve stage ----
FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
