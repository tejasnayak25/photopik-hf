FROM python:3.10-slim
WORKDIR /app

# Install system dependencies required by OpenCV and other libs
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	ca-certificates \
	wget \
	libglib2.0-0 \
	libsm6 \
	libxrender1 \
	libxext6 \
	libx11-6 \
	libxcb1 \
	libfontconfig1 \
	libgl1 \
	libglvnd0 \
	libgl1-mesa-dri \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
