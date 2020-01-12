# Projet PCBS : Eye Tracker

The goal of this project is to create an eye tracker and a game based on it. In this objective I have a made a game with fairly simple rules. The screen is filled with a red circle. The goal is to look longly enough on the black dot to reduce the radius of the red circle to 0. Some things pop-up to distract the user. The score is based on the time you take to make disappear the red circle.

For more information with the documentation please go to the [website](https://dawarriorna.github.io/PCBS_EyeTracker/) _(Generated from the [readme](README.md))_

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need to get this on your local machine. For that you can use the following commands

```shell
$ git clone https://github.com/Dawarriorna/projetPCBS.git
$ cd projetPCBS
```

### Installing

Then you need to install [Python3](https://www.python.org/downloads/) if you don't have it. After that you can install the requirements with the following command :

```shell
$ python3 -m pip install --user --requirement requirements.txt
```

Then try to run the `main.py` file, if it runs, press q to exit
```shell
$ python3 main.py # press q to exit the program
```

### Usage

To play the game you have to start the main program with the following commmand:
```shell
$ python3 main.py
```
The first screen contains all the instructions about how to play. You can exit the program at any time
by pressing `escape` or `q`. Enjoy the game.

## Generating the documentation

You may want to generate yourself the documentation, as if you change the code it won't be automatically updated.

```shell
$ make html # This will produce an output in the folder build/
```

To open the documentation, open the folder build/html/index.html.

If you wish to clean the documentation you can use the following command:

```shell
$ make clean
```

## Running the tests

You can easily run the test with sphinx like so:

```shell
$ make doctest
```

If everything went ok you should have the following output :
```text
[...]
Document: main
--------------
1 items passed all tests:
  33 tests in default
33 tests in 1 items.
33 passed and 0 failed.
Test passed.

Doctest summary
===============
   33 tests
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

## Conclusion ~ To do next

I'm quite satisfied with what I have done. This project takes a lot of time especially the EyeTracker part with eye detection and doing the right threshold. Here are some improvement points that I could have made with more time:

* A better interface
* A more precise EyeTracker
* Different levels of difficulties

### Previous coding experience

I'm coming from a L2 in mathematics, computer science, and cognitive science. So I already used python to do some projects.

### What I learned

I had never used OpenCV nor an EyeTracker.It is new also for me to do a code which is using image analysis. I think that this experience will be helpful in my future career.

### Improvement points for the course

The structure of the course is great. Indeed we saw a large range of python possibilities but it is a little bit frustrating to review all of the "basic" stuff. It could be great to do "skill groups".

## Authors

* **Dana Ladon** - *Founder* - [Dawarriorna](https://github.com/Dawarriorna)

## License

This project is licensed under the Creative Commons BY-ND 4.0 License

![CC](https://creativecommons.org/images/deed/cc_icon_white_x2.png) ![Attribution](https://creativecommons.org/images/deed/attribution_icon_white_x2.png) ![No derivatives](https://creativecommons.org/images/deed/nd_white_x2.png)

Click [here](https://creativecommons.org/licenses/by-nd/4.0/) for more details
