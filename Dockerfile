FROM python:alpine 

LABEL maintainer="Masaru Watanabe"

RUN pip3 install flask

WORKDIR /app

COPY . /app

CMD ["python", "herro.py","--config-file", "configs/eval_real_world_testset.yaml"]