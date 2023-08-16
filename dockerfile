FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu18.04
  2 
  3 
  4 RUN apt-get update && apt-get install -y wget vim curl cmake git sox libsndfile1 libsndfile-dev  ffmpeg gfortran   \
  5     swig   pkg-config zip unzip zlib1g-dev  libbz2-dev sudo  libffi-dev libssl-dev libsqlite3-dev  libavcodec-dev libavformat-dev \
  6     locales  build-essential liblzma-dev python-lzma  && apt-get install -y --allow-downgrades --allow-change-held-packages libnccl2 libnccl-dev  && \
  7     cd /usr/lib/x86_64-linux-gnu && ln -s libcudnn.so.8 libcudnn.so && \
  8     cd /usr/local/cuda-11.2/targets/x86_64-linux/lib  && ln -s libcublas.so.11.3.1.68 libcublas.so && \
  9     ln -s libcusolver.so.11.1.0.152 libcusolver.so && ln -s libcusparse.so.11 libcusparse.so && \
 10     ln -s libcufft.so.10.4.1.152 libcufft.so
 11 
 12 RUN echo "set meta-flag on" >> /etc/inputrc && echo "set convert-meta off" >> /etc/inputrc && \
 13     locale-gen en_US.UTF-8 && /sbin/ldconfig -v && groupadd -g 10001 paddle && \
 14     useradd -m -s /bin/bash -N -u 10001 paddle -g paddle && chmod g+w /etc/passwd && \
 15     echo "paddle ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
 16 
 17 
 18 # official download site: https://www.python.org/ftp/python/3.7.13/Python-3.7.13.tgz
 19 RUN wget https://cdn.npmmirror.com/binaries/python/3.7.13/Python-3.7.13.tgz && tar xvf Python-3.7.13.tgz && \
 20     cd Python-3.7.13 && ./configure --prefix=/workspace/env/python3.7 && make -j8 && make install && \
 21     rm -rf ../Python-3.7.13 ../Python-3.7.13.tgz
 22 
 23 
 24 RUN ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
 25     rm /usr/bin/python &&  ln -s /workspace/env/python3.7/bin/python3.7 /usr/bin/python
 26 
 27 ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 LANGUAGE=en_US.UTF-8 TZ=Asia/Shanghai \
 28     PATH=/workspace/env/python3.7/bin:$PATH \
 29     LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.2/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
 30 
 31 
 32 # paddlespeech 
 33 COPY ./paddlepaddle_gpu-2.4.0rc0.post112-cp37-cp37m-linux_x86_64.whl  /workspace/
 34 RUN python3 -m pip install --upgrade pip  && \
 35     pip install /workspace/paddlepaddle_gpu-2.4.0rc0.post112-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple  && \
 36     rm /workspace/paddlepaddle_gpu-2.4.0rc0.post112-cp37-cp37m-linux_x86_64.whl

    ADD   ./PaddleSpeech.zip   /workspace/
 39 
 40 WORKDIR /workspace/PaddleSpeech/
 41 
 42 RUN unzip -d  /workspace/  /workspace/PaddleSpeech.zip && rm /workspace/PaddleSpeech.zip && \
 43     pip3 install pytest-runner paddleaudio  -i https://pypi.tuna.tsinghua.edu.cn/simple && \
 44     pip3 install -e .[develop] -i https://pypi.tuna.tsinghua.edu.cn/simple && \
 45     pip3 install importlib-metadata==4.2.0 urllib3==1.25.10 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
 46     pip install typeguard==2.13.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
 47 
 48 
 49 COPY torch* /workspace/
 50 
 51 # for jupyter and pytorch 
 52 RUN pip3 install  jupyter jupyterlab  onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple  && \
 53     pip3 install /workspace/torch-1.10.1+cu111-cp37-cp37m-linux_x86_64.whl && \
 54     pip3 install /workspace/torchaudio-0.10.1+cu111-cp37-cp37m-linux_x86_64.whl && \
 55     pip3  install /workspace/torchvision-0.11.2+cu111-cp37-cp37m-linux_x86_64.whl  && \
 56     rm /workspace/torch*
 57 
 58 # for tts_front_server 
 59 COPY  ./nltk_data.tar.gz   /usr/share/
 60 RUN pip install  gunicorn  boto3  -i https://pypi.tuna.tsinghua.edu.cn/simple && \
 61     tar  -zxvf  /usr/share/nltk_data.tar.gz -C /usr/share/  && rm /usr/share/nltk_data.tar.gz
 62 
 63 CMD ['bash']


