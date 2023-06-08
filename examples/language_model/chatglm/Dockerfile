ROM registry.baidubce.com/paddlepaddle/paddle:2.4.2-gpu-cuda11.2-cudnn8.2-trt8.0

RUN groupadd work && useradd -g work -u 1000 -s /bin/bash -m work
WORKDIR /home/work
COPY server.py /home/work/
RUN chown -R work:work server.py

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --no-cache-dir paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html \
    && pip install --no-cache-dir --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html

# USER work
EXPOSE 8086
CMD ["uvicorn", "server:app", "--reload", "--host", "0.0.0.0", "--port", "8086"]