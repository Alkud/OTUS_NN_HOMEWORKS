FROM ubuntu:18.04

LABEL Alexander KUDINOV <alexandr_kudinov@mail.ru>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update
RUN apt-get install -y apt-utils
RUN dpkg-reconfigure -u apt-utils

# Install system utilities
RUN apt-get install -y sudo git curl grep dpkg
# Install Anaconda prerequisites
RUN apt-get install -y libgl1-mesa-glx \
    libegl1-mesa libxrandr2 libxss1 \
    libxcursor1 libxcomposite1 libasound2 libxi6 \
    libxtst6

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install latest Miniconda version
RUN curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh

ENV PATH=/home/user/miniconda/bin:$PATH

# Install no CUDA pytorch
ENV NO_CUDA=1
RUN conda install -y -c pytorch pytorch-cpu \ 
    torchvision-cpu \
    && conda clean -ya

# Install ml and visualization libraries
RUN conda install -y ipython numpy pandas scikit-learn scikit-image notebook matplotlib seaborn

# Configure remote access to jupyter notebook
RUN jupyter notebook --generate-config --allow-root && \
	echo "c.NotebookApp.ip = '*'" >> ~/.jupyter/jupyter_notebook_config.py && \	
	echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = u'sha1:be2bd35a8d79:37fc43a72cd0de8befe88c602ac3d8f2039ac2eb'" >> ~/.jupyter/jupyter_notebook_config.py && \
	echo "c.NotebookApp.allow_remote_access = True" >> ~/.jupyter/jupyter_notebook_config.py && \
	echo "c.NotebookApp.port = 9910" >> ~/.jupyter/jupyter_notebook_config.py    

RUN mkdir -p /home/user/.local/share/jupyter
RUN cd /home/user && ls -lah
RUN sudo chown -R user:user /home/user/.local/share/jupyter

EXPOSE 9910

RUN mkdir /home/user/OTUS_NN
WORKDIR /home/user/OTUS_NN

CMD [ "/bin/bash" ]

COPY dataset.ipynb /home/user/OTUS_NN
COPY homework.ipynb /home/user/OTUS_NN
COPY pytorch.ipynb /home/user/OTUS_NN
COPY utils.py /home/user/OTUS_NN
COPY salt.py /home/user/OTUS_NN

ENTRYPOINT jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --port 9910


