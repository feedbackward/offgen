# Learning as choosing a loss distribution

A visual "explainer" notebook for learning criteria as an additional degree of freedom in our machine learning methodology (presented at [VISxAI 2025](https://visxai.io/)) by Matthew J. Holland.

## Interactive notebook

Launch the interactive version in your browser (no install needed):

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/feedbackward/offgen/ddcf7c0f34ab4534160c5161adaeacb340137677?urlpath=lab%2Ftree%2Foffgen%2Fexplainer.ipynb)

In section 2, there are sliders which can be used to interactively change the data being generated; run the relevant cells at the above link to see the sliders.


## If you just want to read the notebook

Feel free to just read through [the notebook as-is on GitHub](https://github.com/feedbackward/offgen/blob/main/offgen/explainer-static.ipynb). Here is a brief introduction to it.

In machine learning, losses (or rewards) are the fundamental source of data-driven *feedback* that make learning algorithms work. Much of the research literature, both theoretical and applied, is centered around the design of *loss functions* (input: model candidate and data point, output: real value), with the tacit assumption that at training time, *individual losses will be summed* over to construct a final objective function to be optimized. This approach is the bedrock of traditional machine learning (also called empirical risk minimization), but the choice of summing or averaging over losses leads to tradeoffs (e.g., poor worst-case performance, weak high-probability guarantees, issues with fairness or privacy, etc.), meaning that naively averaging over losses might not lead to the outcomes we hope for at test time.


### What is this explainer about?

In our "explainer" notebook, rather than having loss function as our only degree of freedom, we take the loss function as *given*, and instead consider the transformation of a set of losses (or more generally, a loss distribution) to a real value as a key element of the learning algorithm design process. We call this transformation a "learning criterion", with the expected value or mean being the canonical choice of criterion, though we will go far beyond the mean in this explainer. Using simple, illustrated examples, our goal here is to show that paying attention to learning criteria can be fruitful, and that machine learning itself can be understood as selecting a loss distribution with desirable properties, noting that the criterion is tasked with encoding these properties. We introduce and visualize several classes of learning criteria, highlighting their traits under varying distributions, and discussing fundamental limitations that should be considered when designing such criteria.

The notebook is divided into three main sections.

1. Warmup: choosing a loss distribution
2. Quantifying and visualizing learning criteria
3. Fundamental limitations

The first section uses a simple one-variable linear regression task as an example to highlight the distinct nature of "loss function design" and "learning criterion design". In section 2, we then take detailed a look at important classes of learning criteria, with lots of visual aids. Section 3 is slightly more advanced, and highlights unavoidable tradeoffs and potential pitfalls that arise when designing criteria. All the code for generating figures is available in this repository, and can be easily customized and modified.

### Who should read this?

We hope that anyone with an interest in the methodology of modern machine learning can get something out of our explainer. We have done our best to make the terminology accessible and the examples simple and transparent, but a certain degree of basic literacy is needed to really understand what is going on. Anyone who is familiar with how machine learning problems are formally formulated should have no issue.


## For those who want to run the code themselves

We use just a few standard libraries, very easy to prepare. For completeness, here we give a typical flow for setting things up.

Assuming that the user has cloned this repository and thus has the `offgen` directory somewhere on disk, let's set up a virtual environment within our project.

```
cd [path]/offgen
mkdir venv
python3 -m venv ./venv/
```

Next, activate the virtual environment and add the usual software, namely [Jupyter](https://jupyter.org/install), [Matplotlib](https://matplotlib.org/stable/), [SciPy](https://scipy.org/install/), and [scikit-learn](https://scikit-learn.org/stable/install.html)

```
source ./venv/bin/activate
(venv) pip3 install jupyterlab
(venv) pip3 install matplotlib
(venv) pip3 install scipy
(venv) pip3 install -U scikit-learn
(venv) pip3 install ipywidgets
```

Once this is all in place, all that remains is to run the Jupyter notebook.

```
(venv) cd offgen
(venv) jupyter-lab
```

Virtually all the source code for our various helper functions is stored in `setup/utils.py`; please feel free to poke around and customize as desired. Happy learning!
