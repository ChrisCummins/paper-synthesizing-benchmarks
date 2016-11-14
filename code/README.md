## Requirements

* Ubuntu 16.04 or OS X.

Install Python 3:
```sh
# On Ubuntu Linux:
$ sudo apt-get install -y python3 python-virtualenv python3-pip
# On OS X:
$ brew install python3
```

Update python tools:
```sh
$ pip3 install --upgrade pip setuptools wheel
```

Install the CLgen [requirements](http://chriscummins.cc/clgen/#requirements):
```sh
$ curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.1.1/install-deps.sh | bash
```

## Installation - Virtualenv

From this directory, create a virtualenv environment:
```sh
$ virtualenv -p python3 env
```

Activate the virtualenv environment:
```sh
$ source env/bin/activate
```

Install one of the [CLgen](http://chriscummins.cc/clgen/) releases:
```sh
# if you have CUDA:
(env)$ curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.1.1/install-cuda.sh | bash
# if you have OpenCL:
(env)$ curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.1.1/install-opencl.sh | bash
# CPU-only (this will disable some of the experiments):
(env)$ curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.1.1/install-cpu.sh | bash
```

Install the requirements for these experiments:
```sh
(env)$ make install
```

Launch the experiments:
```sh
(env)$ make run
```
