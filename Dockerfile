# どのPythonイメージを使うべきか、基本slimはやめた方が良さそう、ライブラリインストール時にgcc入ってなくて困った
FROM python:3.9.15

WORKDIR  /usr/walmart

RUN apt update \
&& apt upgrade -y \
&& pip install --upgrade pip
RUN pip install jupyterlab \
&&  pip install pandas \
&&  pip install numpy \
&&  pip install sklearn\
&&  pip install mlflow