# anaconda base, python3.8.8
FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

# build tools
RUN apt-get -y update
RUN apt-get -y install wget libssl-dev git

# install eigen, boost
RUN apt-get -y install libboost-all-dev libeigen3-dev

# clear cache
RUN rm -rf /var/lib/apt/lists/*

# install python packages
RUN conda install -y \
    jupyterlab==3.0.14 \
    matplotlib==3.3.4 \
    numba==0.53.1 \
    pandas==1.3.2 \
    seaborn==0.11.2 \
    sphinx==4.2.0 \
    sphinx_rtd_theme==0.4.3
RUN conda install -y -c conda-forge \
    coloredlogs==15.0 \
    timeout-decorator==0.5.0 \
    tensorboard==1.15.0
RUN conda install -y -c anaconda \
    joblib==0.17.0
RUN pip install \
    hydra-core==1.1.0 \
    hydra-joblib-launcher==1.1.5

# install dev env
RUN conda install -y -c conda-forge flake8 black pytest mypy isort line_profiler

# for python-fcl
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /tmp

# install cmake
RUN wget https://github.com/Kitware/CMake/archive/refs/tags/v3.20.2.tar.gz -O cmake.tar.gz
RUN tar zxvf cmake.tar.gz
RUN cd CMake-3.20.2 && ./bootstrap && make -j4 && make install && cd ..
RUN hash -r
RUN rm -rf CMake-3.20.2

# install libccd
RUN git clone -b v2.1 https://github.com/danfis/libccd.git libccd
RUN cd libccd && cmake -B build && make -j4 -C build && make -C build install && cd ..
RUN rm -rf libccd

# install octomap
RUN git clone -b v1.9.7 https://github.com/OctoMap/octomap.git octomap
RUN cd octomap && cmake -B build && make -j4 -C build && make -C build install && cd ..
RUN rm -rf octomap

# install fcl
RUN git clone -b 0.5.0 https://github.com/flexible-collision-library/fcl.git fcl
RUN cd fcl && cmake -B build && make -j4 -C build && make -C build install && cd ..
RUN rm -rf fcl

# install python-fcl, there is no ubuntu version for conda
RUN pip install python-fcl==0.0.12
RUN pip install scikit-image==0.18.1

# install ompl
RUN git clone -b 1.5.2 https://github.com/ompl/ompl.git ompl
RUN cd ompl && cmake -B build && make -C build && make -C build install && cd ..
RUN rm -rf ompl

# install pybind11
RUN git clone -b v2.6.2 https://github.com/pybind/pybind11.git pybind11
RUN cd pybind11 && cmake -B build && make -j4 -C build && make -C build install && cd ..
RUN rm -rf pybind11

# install spars_wrapper
COPY cpp_helper/spars_wrapper/ spars_wrapper
RUN pip install spars_wrapper/

# install cost_to_go_wrapper
COPY cpp_helper/cost_to_go_wrapper/ cost_to_go_wrapper
RUN pip install cost_to_go_wrapper/

# install sphere_collision_check_wrapper
COPY cpp_helper/sphere_collision_check_wrapper/ sphere_collision_check_wrapper
RUN pip install sphere_collision_check_wrapper/

WORKDIR /workspace

# install ctrm, just making path
COPY setup.py .
COPY src/ src/
RUN pip install -e .

# alias
RUN echo 'alias j="jupyter lab --allow-root --ip=0.0.0.0 --ContentsManager.allow_hidden=True"' >> ~/.bashrc
RUN echo 'shopt -s autocd' >> ~/.bashrc