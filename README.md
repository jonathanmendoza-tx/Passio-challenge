# Passio Code Challenge

Implemented a version of the MobileNetV2 convolutional neural network for image classification. Used a toy data-set (cats and dogs) to show it is working.

![MIT](https://img.shields.io/packagist/l/doctrine/orm.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Tensorflow)
![Docker Automated build](https://img.shields.io/docker/automated/tensorflow/tensorflow)

## Project Overview

Used TensorFlow Keras pretrained MobileNetV2 for binary image classification. With a few tweaks, this model could be used for multiclass classification. MobileNetV2 was modified to allow for an output layer which is L2-normalized. After 10 Epochs, the model was about 98% accurate for the validation set.

### Tech Stack

- Python
    - Tensorflow/Keras
- Docker
    - Tensorflow image

### Model

MobileNetV2 is a compact convolutional neural network, created by google. Its compact nature makes it suitable for use on mobile devices.

You can find the official documentaion on tensorflow's [website](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)

### Data Sources

Image dataset (zip file):
    [Cats and dogs filtered](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip)

### Getting Started

Docker is suggested to run this application, but, model.py can be run without Docker, if necessary.

With Docker:
    - create a directory to bind your volume to
    - build docker image from Dockerfile
    - run container
    
    mkdir challenge

    docker build . -t challenge && docker run challenge -v ~/challenge:/model/


without Docker:
    - Must have tensorflow installed, Python 3.6+
    - run model.py
    

     python model.py



## Contributing

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owner of this repository before making a change.

Please note, there is a [code of conduct](./code_of_conduct.md.md). Please follow it in all your interactions with the project.

### Issue/Bug Request

 **If you are having an issue with the existing project code, please submit a bug report under the following guidelines:**
 - Check first to see if your issue has already been reported.
 - Check to see if the issue has recently been fixed by attempting to reproduce the issue using the latest master branch in the repository.
 - Create a live example of the problem.
 - Submit a detailed bug report including your environment & browser, steps to reproduce the issue, actual and expected outcomes,  where you believe the issue is originating from, and any potential solutions you have considered.

### Feature Requests

I would love to hear from you about new features which would improve this app and further the aims of the project. Please provide as much detail and information as possible to show why you think your new feature should be implemented.

### Pull Requests

If you have developed a patch, bug fix, or new feature that would improve this project, please submit a pull request. It is best to communicate your ideas with the developer first before investing a great deal of time into a pull request to ensure that it will mesh smoothly with the project.

Remember that this project is licensed under the MIT license, and by submitting a pull request, you agree that your work will be, too.

#### Pull Request Guidelines

- Ensure any install or build dependencies are removed before the end of the layer when doing a build.
- Update the README.md with details of changes to the interface, including new plist variables, exposed ports, useful file locations and container parameters.
- Ensure that your code conforms to our existing code conventions and test coverage.
- Include the relevant issue number, if applicable.

### Attribution

These contribution guidelines have been adapted from [this good-Contributing.md-template](https://gist.github.com/PurpleBooth/b24679402957c63ec426).
