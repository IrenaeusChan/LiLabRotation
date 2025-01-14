FROM mosaicml/pytorch:1.13.1_cu117-python3.10-ubuntu20.04

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Install dependencies
RUN apt-get update --fix-missing && apt-get -y install gcc g++ less vim bcftools bedtools
RUN apt-get install -y libbz2-dev liblzma-dev zlib1g-dev
RUN apt-get install -y unzip wget curl git make cmake bzip2 ca-certificates
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/{apt,dpkg,cache,log}/

# Download and install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
RUN /opt/conda/bin/conda clean -tipy

# Create a non-root user and switch to it
RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
USER docker

# Set the default command to run when starting the container
CMD ["bash"]