FROM python:slim
WORKDIR /app
EXPOSE  8501
COPY . .
COPY requirements.txt ./requirements.txt
RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda create -n autoencoder-builder && \
    conda activate autoencoder-builder && \
    conda install python=3.9 pip 
RUN pip install -r requirements.txt
RUN echo 'conda activate autoencoder-builder' >> /root/.bashrc
ENTRYPOINT ["streamlit", "run"]
CMD ["Home.py"]