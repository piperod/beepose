FROM nvidia/cuda:9.0-base

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

#RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

ENTRYPOINT [ "/usr/bin/tini", "--" ]
 
RUN apt-get update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev build-essential libcap-dev

WORKDIR /home/beepose
COPY . /home/beepose

RUN cd /home/beepose/ && ls  && conda env create -f beepose.yml && echo "conda activate beepose" >> ~/.bashrc
ENV PATH /opt/conda/envs/beepose/bin:$PATH
RUN echo $PATH
# TODO: install python setup.py in beepose
#RUN /bin/bash -c "source ~/.bashrc && conda activate beepose && cd /home/beepose/ && python setup.py install"
#RUN cd /home/beepose/ && python setup.py install
RUN /bin/bash -c ". activate beepose &&  cd /home/beepose/ && python setup.py install"

CMD [ "/bin/bash" ]
