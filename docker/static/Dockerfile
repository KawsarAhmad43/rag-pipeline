FROM nginx:alpine

# Copy static assets
COPY static /usr/share/nginx/html

# Copy custom nginx configuration
COPY docker/static/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
