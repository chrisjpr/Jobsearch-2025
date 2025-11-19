FROM python:3.11-slim

# Create and activate virtualenv
ENV VIRTUAL_ENV=/venv
RUN python -m venv $VIRTUAL_ENV \
    && . $VIRTUAL_ENV/bin/activate \
    && pip install --upgrade pip

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install system deps (optional but useful for pandas, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies into venv
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY app.py .
COPY helpers.py .
COPY layout.html .


EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
