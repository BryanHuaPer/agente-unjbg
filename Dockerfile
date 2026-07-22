# Usa la imagen oficial de Python 3.12
FROM python:3.12-slim

WORKDIR /app

# Copia y instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código
COPY . .

# Expone el puerto y ejecuta
EXPOSE 10000
CMD ["python", "app.py"]