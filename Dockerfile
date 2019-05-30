FROM python:3

# ADD src /

ADD src/requirements.txt /
RUN pip3 install -r requirements.txt

RUN mkdir /microbnet
WORKDIR /microbnet

# CMD [ "ls" ]
CMD [ "bash", "create_graph.sh" ]

# docker run -w microbnet --name=python-microbiomenet -d -v src:/microbnet python-microbiomenet

# sudo docker run --rm -d -it --name python-microbiomenet -v "$(pwd)"/src:/microbnet -w "/microbnet" python-microbiomenet
