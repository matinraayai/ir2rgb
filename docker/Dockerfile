FROM nvidia/cuda:10.0-devel
ARG PYTHON_VERSION=3.6
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
	 libglib2.0-0 \ 
	 libsm6 \
	 libnvinfer6 \
	 libnvinfer-plugin6 \
	 libxrender1 \
         libpng-dev \
	 libxext6 \
	 wget
RUN wget https://drive.google.com/uc?export=download&confirm=av_R&id=1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x Miniconda3-latest-Linux-x86_64.sh && \
     ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
     rm Miniconda3-latest-Linux-x86_64.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     /opt/conda/bin/pip install torch torchvision opencv-python dominate pytz "tensorflow<2.0.0" && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
