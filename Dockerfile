FROM python:3.10-slim@sha256:a3699f905b890636146817f204e73d9aa61329127b0c60e46310c44f9f0612b2

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/Pytorch_Retinaface:/app/STAR

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1 && \
    grep -v -E '^(torch|torchvision|opencv_python|opencv_contrib_python)(==|$)' requirements.txt > /tmp/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir opencv-contrib-python-headless==4.11.0.86

COPY . /app

RUN if [ ! -f Pytorch_Retinaface/models/retinaface.py ]; then \
        rm -rf Pytorch_Retinaface && \
        git clone --depth 1 https://github.com/biubug6/Pytorch_Retinaface.git Pytorch_Retinaface; \
    fi && \
    if [ ! -f STAR/demo.py ]; then \
        rm -rf STAR && \
        git clone --depth 1 https://github.com/ZhenglinZhou/STAR.git STAR; \
    fi

ARG DOWNLOAD_OPENFACE_WEIGHTS=1
RUN if [ "$DOWNLOAD_OPENFACE_WEIGHTS" = "1" ]; then \
        mkdir -p weights && \
        if [ ! -f weights/WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl ]; then \
            curl -L --retry 3 -o weights/WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl \
                https://huggingface.co/nutPace/openface_weights/resolve/main/Landmark_98.pkl; \
        fi && \
        if [ ! -f weights/mobilenet0.25_Final.pth ]; then \
            curl -L --retry 3 -o weights/mobilenet0.25_Final.pth \
                https://huggingface.co/nutPace/openface_weights/resolve/main/Alignment_RetinaFace.pth; \
        fi && \
        if [ ! -f weights/stage2_epoch_7_loss_1.1606_acc_0.5589.pth ]; then \
            curl -L --retry 3 -o weights/stage2_epoch_7_loss_1.1606_acc_0.5589.pth \
                https://huggingface.co/nutPace/openface_weights/resolve/main/MTL_backbone.pth; \
        fi; \
    fi

RUN sed -i -e '/^import dlib$/d' -e '/^import gradio as gr$/d' STAR/demo.py && \
    sed -i 's/from scipy.integrate import simps/from scipy.integrate import simpson as simps/' STAR/lib/metric/fr_and_auc.py && \
    sed -i 's/net = net.to(self.config.device_id)/net = net.cpu() if self.config.device_id == -1 else net.to(self.config.device_id)/' STAR/demo.py && \
    sed -i 's/input_tensor = input_tensor.to(self.config.device_id)/input_tensor = input_tensor.cpu() if self.config.device_id == -1 else input_tensor.to(self.config.device_id)/' STAR/demo.py

CMD ["python", "interface.py", "--image", "images/89.jpg"]
