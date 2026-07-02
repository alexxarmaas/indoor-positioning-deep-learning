# Primera Etapa: Construir la aplicación Angular
FROM node:18-alpine AS node

WORKDIR /app

# Copiar el archivo package.json e instalar dependencias
COPY package.json package-lock.json ./
RUN npm ci

# Copiar el resto de los archivos y construir la aplicación Angular
COPY . /app

# Ajustar el presupuesto en el archivo angular.json
RUN sed -i 's/"maximumError": "1mb"/"maximumError": "2mb"/' /app/angular.json

RUN npm run build -- --configuration production

# Segunda Etapa: Configurar el servidor NGINX
FROM nginx:stable-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf


# Copiar los archivos de la primera etapa al directorio del servidor NGINX
COPY --from=node /app/dist/ipscm.web-client /usr/share/nginx/html

