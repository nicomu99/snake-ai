# Builds the frontend and serves it with nginx.
# Build context is frontend/:  docker build -t snake-ai-frontend frontend

FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# nginx-unprivileged runs as a non-root user and listens on 8080.
FROM nginxinc/nginx-unprivileged:1.28-alpine
# Where nginx forwards /ws and /models. The nginx entrypoint fills it into the
# config template on start, so it can be changed with `docker run -e`.
ENV BACKEND_URL=http://backend:8080
COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
