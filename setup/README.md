# Setup and Installation

This is a step-by-step tutorial that will guide you through the installation of the necessary libraries and how to setup your environment, in order to run the various code examples.

The code should run on any machine (i.e. Windows, macOS, Linux) that supports Python 3.

__Note:__ It is possible that the code might work with Python 2, either as is or with a few modifications. However, this repository does not support Python 2. Also, there is no guarantee that individual files will work on their own.

## Quick installation

If, for any reason, you want to spend the minimum amount of time for setting up your environment, then you can follow __all__ the steps below.

### 1. Install Python 3

In order to save a lot of trouble, download the [Anaconda Distribution](https://www.anaconda.com/distribution/), which is actively supported by the Data Science community.

1. Go to the [Download](https://www.anaconda.com/download/) page and choose the installation for either [Windows](https://www.anaconda.com/download/#windows), [macOS](https://www.anaconda.com/download/#macos) or [Linux](https://www.anaconda.com/download/#linux) accordingly.
2. Choose **Python 3.6** or later.

The Anaconda Distribution also includes the [SPYDER](https://www.spyder-ide.org/) IDE, which is the one that was used for writting the code examples in this repository. This IDE is considered to be ideal for Machine Learning, Data Science, Scientific Computing, Computational Science and Engineering.

### 2. Install the required packages

The code examples require several Python packages to be installed. These packages are listed in the 
[requirements.txt](requirements.txt) file of this folder. You do not need to edit the file for the quick installation.

To install the required Python packages and dependencies you have to run the following commands in a terminal:

    pip install -r requirements.txt
    conda install PyYAML==3.13 -y

Note that in order to run the commands above, you first have to change to the directory to the current one:

    cd <your directory path>/setup
    
__Note:__ For Windows, you should run all of the commands, including the above, at __Anaconda Prompt__ instead of the Terminal.

### 3. Download the datasets

You need to download the necessary data for each corresponding code example, as they are not inluded in this repository. To do this, follow the instructions in the [datasets](../datasets) folder.

    cd <your directory path>/datasets

## Recommended installation

TODO: write this section

## Troubleshoot (if you have errors or problems)

If you face any errors, then please try asking at __all__ of the available forums and websites online __first__. For example:
- [StackExchange](https://stackexchange.com/) and any of its sub-forums, such as [Data Science](https://datascience.stackexchange.com/)
- [Reddit](https://www.reddit.com/) and any of its sub-forums, such as [Python](https://www.reddit.com/r/Python/)
- any other website similar to the above

If you have asked at __all__ of the available forums and websites online, then in that case, you can copy-paste the URL links of your questions in this repository and ask your question here. We will try our best to resolve the issue(s).

