FROM python:3.7.7

RUN mkdir src
WORKDIR src/

COPY . .

RUN pip install numpy
RUN pip install matplotlib
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install tensorflow
RUN pip install keras
RUN pip install openml
RUN pip install pyod
RUN pip install mxnet
RUN pip install autogluon
RUN pip install mxnet-mkl --pre


CMD [ "python", "experimental_setup_final.py" ]
