FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Train models if not present
RUN python train_model.py

EXPOSE 5000

ENV FLASK_APP=app.py

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
