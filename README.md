# Projet PCBS : Eye Tracker

The goal of this project is to create an eye tracker and a game based it.Eye tracking the user will allow him to improve his focus. In fact tracking his eyes allows us to be able to evaluate 
how focused he is, and permits him to improve over time whilst playing.
In the end the game should look like a race, where the user can control a ball with his eyes while trying to avoid being disturbed by some background noise.

[Website](https://dawarriorna.github.io/projetPCBS/) - _Generated with sphinx_

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need to get this on your local machine. For that you can use the following commands

```shell script
$ git clone https://github.com/Dawarriorna/projetPCBS.git
$ cd projetPCBS
```

### Installing

Then you need to install [Python3](https://www.python.org/downloads/) if you don't have it. After that you can install the requirements with the following command :

```shell script
$ python3 -m pip install --user --requirement requirements.txt
```

Then try to run the `main.py` file, if it runs, press q to exit
```shell script
$ python3 main.py # press q to exit the program
```

## Running the tests

### Running the tests with sphinx

You can easily run the test with sphinx like so:

```shell script
$ make doctest
```

If everything went ok you should have the following output :
```text
[...]
Document: main
--------------
1 items passed all tests:
  44 tests in default
44 tests in 1 items.
44 passed and 0 failed.
Test passed.

Doctest summary
===============
   44 tests
    0 failures in tests
    0 failures in setup code
    0 failures in cleanup code
build succeeded.
[...]

```

## Built With

* [Python](https://www.python.org/downloads/) - The language used
* [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html) - A nice module for face and eye detection
* [Numpy](https://numpy.org/doc/1.17/) - scientific computing with Python
* [Pygame](https://www.pygame.org/docs/) - The graphical package of python used
* [Sphinx](http://www.sphinx-doc.org/en/master/) - The module used to generate the documentation

## Authors

* **Dana Ladon** - *Founder* - [Dawarriorna](https://github.com/Dawarriorna)

## License

This project is licensed under the Creative Commons BY-ND 4.0 License

![CC](https://creativecommons.org/images/deed/cc_icon_white_x2.png) ![Attribution](https://creativecommons.org/images/deed/attribution_icon_white_x2.png) ![No derivatives](https://creativecommons.org/images/deed/nd_white_x2.png)

Click [here](https://creativecommons.org/licenses/by-nd/4.0/) for more details
