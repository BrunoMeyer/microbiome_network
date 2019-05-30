FROM python:3

ADD src/requirements.txt /
RUN pip3 install -r requirements.txt

RUN mkdir /microbnet
WORKDIR /microbnet

CMD [ "bash", "create_graph.sh" ]

# sudo docker run --rm -d -it --name python-microbiomenet -v "$(pwd)"/src:/microbnet -w "/microbnet" python-microbiomenet
