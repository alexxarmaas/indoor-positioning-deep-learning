# Empezamos descargando la imagen de Python
FROM python:3.11-slim-bookworm

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos el archivo de requerimientos dentro de la imagen
COPY requirements.txt .

# Actualizamos pip
RUN pip install --upgrade pip

# Instalamos las dependencias especificadas en requirements.txt
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código de la aplicación
COPY . .

# Exponemos el puerto deseado
EXPOSE 5000

# Establecemos el comando para ejecutar la aplicación
CMD ["python", "app.py"]
