FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu18.04-0327


RUN apt-get update && apt-get install -y wget vim curl cmake git sox libsndfile1 libsndfile-dev  ffmpeg gfortran   \
    swig   pkg-config zip unzip zlib1g-dev  libbz2-dev sudo  libffi-dev libssl-dev libsqlite3-dev  libavcodec-dev libavformat-dev \
    locales  build-essential liblzma-dev python-lzma  && apt-get install -y --allow-downgrades --allow-change-held-packages libnccl2 libnccl-dev  && \
    cd /usr/lib/x86_64-linux-gnu && ln -s libcudnn.so.8 libcudnn.so && \
    cd /usr/local/cuda-11.2/targets/x86_64-linux/lib  && ln -s libcublas.so.11.3.1.68 libcublas.so && \
    ln -s libcusolver.so.11.1.0.152 libcusolver.so && ln -s libcusparse.so.11 libcusparse.so && \
    ln -s libcufft.so.10.4.1.152 libcufft.so

RUN echo "set meta-flag on" >> /etc/inputrc && echo "set convert-meta off" >> /etc/inputrc && \
    locale-gen en_US.UTF-8 && /sbin/ldconfig -v && groupadd -g 10001 paddle && \
    useradd -m -s /bin/bash -N -u 10001 paddle -g paddle && chmod g+w /etc/passwd && \
    echo "paddle ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers


# official download site: https://www.python.org/ftp/python/3.7.13/Python-3.7.13.tgz
RUN wget https://cdn.npmmirror.com/binaries/python/3.7.13/Python-3.7.13.tgz && tar xvf Python-3.7.13.tgz && \
    cd Python-3.7.13 && ./configure --prefix=/workspace/env/python3.7 && make -j8 && make install && \
    rm -rf ../Python-3.7.13 ../Python-3.7.13.tgz 


RUN ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    rm /usr/bin/python &&  ln -s /workspace/env/python3.7/bin/python3.7 /usr/bin/python

ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 LANGUAGE=en_US.UTF-8 TZ=Asia/Shanghai \
    PATH=/workspace/env/python3.7/bin:$PATH \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.2/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}


# paddlespeech 
COPY ./paddlepaddle_gpu-2.4.0rc0.post112-cp37-cp37m-linux_x86_64.whl  /workspace/
RUN python3 -m pip install --upgrade pip  && \
    pip install /workspace/paddlepaddle_gpu-2.4.0rc0.post112-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple  && \
    rm /workspace/paddlepaddle_gpu-2.4.0rc0.post112-cp37-cp37m-linux_x86_64.whl 

ADD   ./PaddleSpeech.zip   /workspace/ 

WORKDIR /workspace/PaddleSpeech/

RUN unzip -d  /workspace/  /workspace/PaddleSpeech.zip && rm /workspace/PaddleSpeech.zip && \
    pip3 install pytest-runner paddleaudio  -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install -e .[develop] -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install importlib-metadata==4.2.0 urllib3==1.25.10 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install typeguard==2.13.3 -i https://pypi.tuna.tsinghua.edu.cn/simple 


COPY torch* /workspace/ 

# for jupyter and pytorch 
RUN pip3 install  jupyter jupyterlab  onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple  && \
    pip3 install /workspace/torch-1.10.1+cu111-cp37-cp37m-linux_x86_64.whl && \
    pip3 install /workspace/torchaudio-0.10.1+cu111-cp37-cp37m-linux_x86_64.whl && \
    pip3  install /workspace/torchvision-0.11.2+cu111-cp37-cp37m-linux_x86_64.whl  && \
    rm /workspace/torch*

# for tts_front_server 
COPY  ./nltk_data.tar.gz   /usr/share/
RUN pip install  gunicorn  boto3  -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    tar  -zxvf  /usr/share/nltk_data.tar.gz -C /usr/share/  && rm /usr/share/nltk_data.tar.gz

CMD ['bash']



