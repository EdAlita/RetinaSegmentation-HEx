## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Authors](#authors)
* [Setup](#setup)
* [To Do's](#todo)
* [FAQ](#faq)

## General Info
This project focus on the Image analysis of retinal color fundus images. The
main focus of the segmentation will be to detected lesion on the early
detection of diabetic retinoplasthy, being the point of focus the Hard Exudus.

![alt text](https://pub.mdpi-res.com/sensors/sensors-21-03704/article_deploy/html/images/sensors-21-03704-g001.png?1623041606)

The segementation perfomance evaluation is done by area under the
precision-recall curve (AUPR).


## Technologies
Project is created with: 
* Python 3.10.6
* OpenCV 
* Machine Learning ucasML

## Authors
Colabolators in the project: 
* [Edwing Ulin](https://github.com/EdAlita)
* [Jaqueline Leal](https://github.com/JLealc)
* [Carmen]()
* [Taibur]()

## Setup
* Install sofware dependencies that are mention in the Technologies
    * Install the next libraries in python:
        * Glob
        * cv2
        * logging
        * time
* First create a file name main.cfg in the next route /code/main.cfg. The first line of the location of the test folder images in your local machine and the second is the trainning.
## TO DO's

- [ ] Vein extraction
- [ ] Filters of the Eye Image to enhance hard and soft Exodus
- [ ] Binarization of the Image and obtain area of the hard and soft Exodus
- [ ] Evaluate which MAchine Learning tool to use
- [ ] How to evaluate or model
- [ ] The documentation?

## FAQ

### How to log in our project?

The logging capabilty pf pur project is to get a runtime history of the code and help us on debuging code. The file should be cretaed in the next structure:

/code/03282023-124502.log

Each time that you run the main.py a file is created under the code folder. The name of the file has the next structure:  month+day+year-hour+minute+second.log

You don't need to install a library for this is a build in fucntion of vanilla python.

You can use the next function to log your functions

|Logging function                                                     |Use                                                                             |
|:---                                                                 |:---                                                                            |
|logging.debug("Debug logging test...")                               | Use this one to mark the flow of our program.                                  |
|logging.info("Program is working as expected")                       | Use when you make a change to the images, to see if they affect all the images |
|logging.warning("Warning, the program may not function properly")    | No use so far                                                                  |
|logging.error("The program encountered an error")                    | No use so far                                                                  |
|logging.critical("The program crashed")                              | No use so far                                                                  |

### How to make commits from terminal to this repository

The first stepp is to setup your ssh key with github follow this tutorial: 
    [SSH Tutorial](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

After that you need to setup your credentials in your local github instalation, ***Same ones as your github account***

* Change the email and username to the ones that you have in github with:
    1. Open your terminal and navigate to your git repository.
    2. Change Git user name by running: `git config --global user.name “Your Name”`
    3. Change Git user email by running: `git config --global user.email “name@email.com”`
    4. Check the config with:
        * git config --list

* After that you are ready for making your first commit, navigate to your local copy of this project: 
    1. Check the status of your changes with `git status`
        * if the files are in __red__ : are changes,but not staged
        * if the files are in __green__ : are changes in staged
        * you wnat the files in __green__ since only the changes in the staged will be commited to the repository.
    2. Add the files to the stagged are with `add [file name]`or add all files with `add *`
    3. Check again Status.
    4. Prepare the commit with `commit -m "[Add message]"`
        * Please be as ***specific*** with the changes, updates or what are you committing
    5. Pull lastest change to avoid merging problems with `git pull`
    6. push your changes with `git push`

> If you fell confortable on using branches, please remeber to merge them to the main one








