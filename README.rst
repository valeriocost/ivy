
.. _`Backend Handler`: https://lets-unify.ai/docs/ivy/overview/design/building_blocks.html#backend-handler
.. _`Backend Functional APIs`: https://lets-unify.ai/docs/ivy/overview/design/building_blocks.html#backend-functional-apis

.. _`Mechanics`: https://github.com/unifyai/mech
.. _`Computer Vision`: https://github.com/unifyai/vision
.. _`Robotics`: https://github.com/unifyai/robot
.. _`Reinforcement Learning Gym`: https://github.com/unifyai/gym
.. _`Memory`: https://github.com/unifyai/memory
.. _`Builder tools`: https://github.com/unifyai/builder
.. _`Models`: https://github.com/unifyai/models

.. _`Examples page`: https://lets-unify.ai/demos/
.. _`open tasks`: https://lets-unify.ai/docs/ivy/overview/contributing/open_tasks.html

.. _`Discord`: https://discord.gg/sXyFF8tDtm
.. _`Twitter`: https://twitter.com/letsunifyai


.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/logo_dark.png?raw=true#gh-dark-mode-only
   :width: 100%
   :class: only-dark

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/logo.png?raw=true#gh-light-mode-only
   :width: 100%
   :class: only-light

..

   ⚠️ **Warning**: The compiler and the transpiler are not publicly available yet, so certain parts of this README won't work as expected as of now!

.. raw:: html

    <br/>
    <div align="center">
    <a href="https://github.com/unifyai/ivy/issues">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/issues/unifyai/ivy">
    </a>
    <a href="https://github.com/unifyai/ivy/network/members">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/forks/unifyai/ivy">
    </a>
    <a href="https://github.com/unifyai/ivy/stargazers">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/stars/unifyai/ivy">
    </a>
    <a href="https://github.com/unifyai/ivy/pulls">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg">
    </a>
    <a href="https://pypi.org/project/ivy-core">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://badge.fury.io/py/ivy-core.svg">
    </a>
    <a href="https://github.com/unifyai/ivy/actions?query=workflow%3Adocs">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/unifyai/ivy/actions/workflows/docs.yml/badge.svg">
    </a>
    <a href="https://github.com/unifyai/ivy/actions?query=workflow%3Atest-ivy">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/unifyai/ivy/actions/workflows/test-ivy.yml/badge.svg">
    </a>
    <a href="https://discord.gg/sXyFF8tDtm">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/discord/799879767196958751?color=blue&label=%20&logo=discord&logoColor=white">
    </a>
    </div>
    <br clear="all" />

.. raw:: html

    <div style="display: block;" align="center">
    <b><a href="https://lets-unify.ai/">Website</a></b> | <b><a href="https://lets-unify.ai/docs/ivy/">Docs</a></b> | <b><a href="https://lets-unify.ai/demos/">Examples</a></b> | <b><a href="https://lets-unify.ai/docs/ivy/overview/design.html">Design</a></b> | <b><a href="https://lets-unify.ai/docs/ivy/overview/faq.html">FAQ</a></b><br><br>
    
    <b>All of AI, at your fingertips</b>
    
    </div>
    
    <br>
    
    <div style="display: block;" align="center">
        <div>
        <a href="https://jax.readthedocs.io">
            <img class="dark-light" width="10%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/jax_logo.png">
        </a>
        <img width="1%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://www.tensorflow.org">
            <img class="dark-light" width="10%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/tensorflow_logo.png">
        </a>
        <img width="1%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://pytorch.org">
            <img class="dark-light" width="10%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/pytorch_logo.png">
        </a>
        <img width="1%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://numpy.org">
            <img class="dark-light" width="10%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/numpy_logo.png">
        </a>
        </div>
    </div>
    
    <br clear="all" />

------------------------------------------------------

Ivy is both an ML transpiler and a framework, currently supporting JAX, TensorFlow, PyTorch and Numpy.

Ivy unifies all ML frameworks 💥 enabling you not only to **write code that can be used with any of these frameworks as the backend**, 
but also to **convert 🔄 any function, model or library written in any of them to your preferred framework!**

You can check out `Ivy as a transpiler`_ and `Ivy as a framework`_ to learn more about this, try out Ivy
straight away going through the `Setting up Ivy`_ section, or dive deep into Ivy's `Documentation`_ and `Examples`_!

If you would like to contribute, you can join our growing `Community`_ 🌍, check out our `Contributing`_ guide,
and take a look at the `open tasks`_ if you'd like to dive straight in 🧑‍💻 

`lets-unify.ai <https://lets-unify.ai>`_ **together 🦾**


Contents
--------

* `Ivy as a transpiler`_
* `Ivy as a framework`_
* `Setting up Ivy`_
* `Documentation`_
* `Examples`_
* `Contributing`_
* `Community`_
* `Citation`_

Ivy as a transpiler
-------------------

Ivy's transpiler allows you to use code from any other framework (or from any other version of the same framework!) in your own code, by just adding one line of code. Under the hood, Ivy traces a computational graph and leverages the frontends and backends to link one framework to another. 

This way, Ivy makes all ML-related projects available for you, independently of the framework you want to use to research, develop, or deploy systems. Feel free to head over to the docs for the full API reference, but the functions you'd most likely want to use are:

.. code-block:: python

    # Compiles a function into an efficient fully-functional graph, removing all wrapping and redundant code
    ivy.compile()

    # Converts framework-specific code to a different framework
    ivy.transpile()

    # Converts framework-specific code to Ivy
    ivy.unify()

These functions can be used eagerly or lazily. If you pass the necessary arguments for function tracing, the compilation/transpilation step will happen instantly (eagerly). Otherwise, the compilation/transpilation will happen only when the returned function is first invoked.

.. code-block:: python
    
    import ivy
    import jax
    ivy.set_backend("jax")

    # Simple JAX function to transpile
    def test_fn(x):
        return jax.numpy.sum(x)

    x1 = ivy.array([1., 2.])

.. code-block:: python
    
    # Arguments are available -> transpilation happens eagerly
    eager_graph = ivy.transpile(test_fn, source="jax", to="torch", args=(x1,))
    
    # eager_graph is now torch code and runs efficiently
    ret = eager_graph(x1)

.. code-block:: python
    
    # Arguments are not available -> transpilation happens lazily
    lazy_graph = ivy.transpile(test_fn, source="jax", to="torch")
    
    # The transpiled graph is initialized, transpilation will happen here
    ret = lazy_graph(x1)
    
    # lazy_graph is now torch code and runs efficiently
    ret = lazy_graph(x1)

If you want to learn more, you can find more information in the `Ivy as a transpiler section of the docs! <https://lets-unify.ai/docs/ivy/overview/design/ivy_as_a_transpiler.html>`_

When should I use Ivy as a transpiler?
######################################

If you want to use building blocks published in other frameworks (neural networks, layers, array computing libraries, training pipelines...), you want to integrate code developed in various frameworks, or maybe straight up move code from one framework to another, the transpiler is definitely the tool 🔧 for the job! As the output of transpilation is native code in the target framework, you can use the converted code just as if it was code originally developed in that framework, applying framework-specific optimizations or tools, instantly exposing your project to all of the unique perks of a different framework.

Ivy as a framework
-------------------

The Ivy framework is built on top of various essential components, mainly the `Backend Handler`_, which manages what framework is being used behind the scenes and the `Backend Functional APIs`_, which provide framework-specific implementations of the Ivy functions. Likewise, classes such as :code:`ivy.Container` or :code:`ivy.Array` are also available, facilitating the use of structured data and array-like objects (learn more about them `here! <https://lets-unify.ai/docs/ivy/overview/design/ivy_as_a_framework.html>`_). 

All of the functionalities in Ivy are exposed through the :code:`Ivy functional API` and the :code:`Ivy stateful API`. All functions in the `Functional API <https://lets-unify.ai/docs/ivy/overview/design/building_blocks.html#ivy-functional-api>`_ are **Framework Agnostic Functions**, which mean that we can use them like this:

.. code-block:: python

    import ivy
    import jax.numpy as jnp
    import tensorflow as tf
    import numpy as np
    import torch

    def mse_loss(y, target):
        return ivy.mean((y - target)**2)

    jax_mse   = mse_loss(jnp.ones((5,)), jnp.ones((5,)))
    tf_mse    = mse_loss(tf.ones((5,)), tf.ones((5,)))
    np_mse    = mse_loss(np.ones((5,)), np.ones((5,)))
    torch_mse = mse_loss(torch.ones((5,)), torch.ones((5,)))

In the example above we show how Ivy's functions are compatible with tensors from different frameworks.
This is the same for ALL Ivy functions. They can accept tensors from any framework and return the correct result.

The `Ivy Stateful API <https://lets-unify.ai/docs/ivy/overview/design/ivy_as_a_framework/ivy_stateful_api.html>`_, on the other hand, allows you to define trainable modules and layers, which you can use alone or as a part of any other framework code!

.. code-block:: python

    import ivy

    class MyModel(ivy.Module):
        def __init__(self):
            self.linear0 = ivy.Linear(3, 64)
            self.linear1 = ivy.Linear(64, 1)
            ivy.Module.__init__(self)

        def _forward(self, x):
            x = ivy.relu(self.linear0(x))
            return ivy.sigmoid(self.linear1(x))


If we put it all together, we'll have something like this. This example uses PyTorch as the backend,
but this can easily be changed to your favorite framework, such as TensorFlow, or JAX.

.. code-block:: python

    import ivy

    class MyModel(ivy.Module):
        def __init__(self):
            self.linear0 = ivy.Linear(3, 64)
            self.linear1 = ivy.Linear(64, 1)
            ivy.Module.__init__(self)

        def _forward(self, x):
            x = ivy.relu(self.linear0(x))
            return ivy.sigmoid(self.linear1(x))

    ivy.set_backend('torch')  # change to any backend!
    model = MyModel()
    optimizer = ivy.Adam(1e-4)
    x_in = ivy.array([1., 2., 3.])
    target = ivy.array([0.])

    def loss_fn(v):
        out = model(x_in, v=v)
        return ivy.mean((out - target)**2)

    for step in range(100):
        loss, grads = ivy.execute_with_gradients(loss_fn, model.v)
        model.v = optimizer.step(model.v, grads)
        print('Step: {} --- Loss: {}'.format(step, ivy.to_numpy(loss).item()))

    print('Finished training!')


Last but not least, we are also working on specific extension totally written in Ivy and therefore usable within any framework, 
covering topics like `Mechanics`_, `Computer Vision`_, `Robotics`_, a `Reinforcement Learning Gym`_, `Memory`_ and implementation of various `Models`_ or `Builder tools`_ with trainers, data loaders and more!

.. raw:: html

    <br/>
    <div align="center">
    <a href="https://github.com/unifyai/mech">
        <img width="12.5%" style="padding-right: 4px; padding-bottom: 4px;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_mech.png">
    </a>
    <a href="https://github.com/unifyai/vision">
        <img width="12.5%" style="padding-right: 4px; padding-bottom: 4px;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_vision.png">
    </a>
    <a href="https://github.com/unifyai/robot">
        <img width="12.5%" style="padding-right: 4px; padding-bottom: 4px;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_robot.png">
    </a>
    <a href="https://github.com/unifyai/gym">
        <img width="12.5%" style="padding-right: 4px; padding-bottom: 4px;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_gym.png">
    </a>
    <a href="https://github.com/unifyai/memory">
        <img width="12.5%" style="padding-right: 4px; padding-bottom: 4px;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_memory.png">
    </a>
    <a href="https://github.com/unifyai/builder">
        <img width="12.5%" style="padding-right: 4px; padding-bottom: 4px;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_builder.png">
    </a>
    <a href="https://github.com/unifyai/models">
        <img width="12.5%" style="padding-right: 4px; padding-bottom: 4px;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/ivy_models.png">
    </a>
    </div>
    <br clear="all" />

As always, you can find more information about `Ivy as a framework in the docs! <https://lets-unify.ai/docs/ivy/overview/design/ivy_as_a_framework.html>`_

When should I use Ivy as a framework?
######################################

As Ivy supports multiple backends, writing code in Ivy breaks you free from framework limitations. If you want to publish highly flexible code for everyone to use, independently of the framework they are using, or you plan to develop ML-related tools and want them to be interoperable with not only the already existing frameworks, but also with future frameworks, then Ivy is for you!

Setting up Ivy
--------------

There are various ways to use Ivy, depending on your preferred environment:

Installing using pip
####################

The easiest way to set up Ivy is to install it using pip with the following command:

.. code-block:: bash

    pip install ivy-core

or alternatively:

.. code-block:: bash

    python3 -m pip install ivy-core


Docker
######

If you prefer to use containers, we also have pre-built Docker images with all the supported frameworks and some relevant packages already installed, which you can pull from:

.. code-block:: bash

    docker pull unifyai/ivy:latest

If you are working on a GPU device, you can pull from:

.. code-block:: bash

    docker pull unifyai/ivy:latest-gpu

Installing from source
######################

You can also install Ivy from source if you want to take advantage of the latest changes, but we can't ensure everything will work as expected. :sweat_smile:

.. code-block:: bash

    git clone https://github.com/unifyai/ivy.git
    cd ivy 
    pip install --user -e .

or alternatively, for the last step:

.. code-block:: bash

    python3 -m pip install --user -e .

If you want to set up testing and various frameworks it's probably best to check out the `Contributing - Setting Up <https://lets-unify.ai/docs/ivy/overview/contributing/setting_up.html#setting-up>`_ page, where OS-specific and IDE-specific instructions and video tutorials to do so are available!


Using Ivy
#########

You can find quite a lot more examples in the corresponding section below, but using Ivy is as simple as:

.. raw:: html

   <h4>Multi-backend Support</h4>

.. code-block:: python

    import ivy
    import torch
    import jax

    ivy.set_backend("jax")

    x = jax.numpy.array([1, 2, 3])
    y = jax.numpy.array([3, 2, 1])
    z = ivy.add(x, y)

    ivy.set_backend('torch')

    x = torch.tensor([1, 2, 3])
    y = torch.tensor([3, 2, 1])
    z = ivy.add(x, y)

.. raw:: html

   <h4>Transpilation API</h4>

.. code-block:: python

   import ivy
   import torch
   import jax

   def jax_fn(x):
       a = jax.numpy.dot(x, x)
       b = jax.numpy.mean(x)
       return x * a + b

   jax_x = jax.numpy.array([1, 2, 3])
   torch_x = torch.tensor([1, 2, 3])
   torch_fn = ivy.transpile(jax_fn, source="jax", to="torch", args=(jax_x,))
   ret = torch_fn(torch_x)


Documentation
-------------

The `Ivy Docs page <https://lets-unify.ai/docs/ivy/>`_ holds all the relevant information about Ivy's and it's framework API reference. 

There, you will find the `Design <https://lets-unify.ai/docs/ivy/overview/design.html>`_ page, which is a user-focused guide about the architecture and the building blocks of Ivy. Likewise, you can take a look at the `Deep dive <https://lets-unify.ai/docs/ivy/overview/deep_dive.html>`_, which is oriented towards potential contributors of the code base and explains the nuances of Ivy in full detail 🔎

Another important sections of the docs is `Background <https://lets-unify.ai/docs/ivy/overview/background.html>`_, which contextualises the problem Ivy is trying to solve and the current `ML Explosion <https://lets-unify.ai/docs/ivy/overview/background/ml_explosion.html#ml-explosion>`_, explaining both (1) why is important `to solve this problem <https://lets-unify.ai/docs/ivy/overview/background/why_unify.html#why-unify>`_ and (2) how we are adhering to existing `standards <https://lets-unify.ai/docs/ivy/overview/background/standardization.html#standardization>`_ to make this happen.

Lastly, you can also find there the `Related Work <https://lets-unify.ai/docs/ivy/overview/related_work.html>`_ section, which paints a clear picture of the role Ivy plays in the ML stack, comparing it to other existing solutions in terms of functionalities and level.


Examples
--------

The `Examples page`_ features a wide range of demos and tutorials showcasing the functionalities of Ivy along with multiple use cases, but feel free to check out some shorter framework-specific examples here ⬇️

-- only:: html

    .. tab-set::

        .. tab-item:: PyTorch

            .. tab-set::

                .. tab-item:: Any model

                    .. tab-set::

                        .. tab-item:: From TensorFlow

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From JAX

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From NumPy

                            .. code-block:: python

                                print("Hello World")

                .. tab-item:: Any library

                    .. tab-set::

                        .. tab-item:: From TensorFlow

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From JAX

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From NumPy

                            .. code-block:: python

                                print("Hello World")

                .. tab-item:: Any function

                    .. tab-set::

                        .. tab-item:: From TensorFlow

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From JAX

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From NumPy

                            .. code-block:: python

                                print("Hello World")

        .. tab-item:: TensorFlow

            .. tab-set::

                .. tab-item:: Any model

                    .. tab-set::

                        .. tab-item:: From PyTorch

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From JAX

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From NumPy

                            .. code-block:: python

                                print("Hello World")

                .. tab-item:: Any library

                    .. tab-set::

                        .. tab-item:: From PyTorch

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From JAX

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From NumPy

                            .. code-block:: python

                                print("Hello World")

                .. tab-item:: Any function

                    .. tab-set::

                        .. tab-item:: From PyTorch

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From JAX

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From NumPy

                            .. code-block:: python

                                print("Hello World")

        .. tab-item:: Jax

            .. tab-set::

                .. tab-item:: Any model

                    .. tab-set::

                        .. tab-item:: From PyTorch

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From TensorFlow

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From NumPy

                            .. code-block:: python

                                print("Hello World")

                .. tab-item:: Any library

                    .. tab-set::

                        .. tab-item:: From PyTorch

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From TensorFlow

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From NumPy

                            .. code-block:: python

                                print("Hello World")

                .. tab-item:: Any function

                    .. tab-set::

                        .. tab-item:: From PyTorch

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From TensorFlow

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From NumPy

                            .. code-block:: python

                                print("Hello World")

        .. tab-item:: NumPy

            .. tab-set::

                .. tab-item:: Any library

                    .. tab-set::

                        .. tab-item:: From PyTorch

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From TensorFlow

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From JAX

                            .. code-block:: python

                                print("Hello World")

                .. tab-item:: Any function

                    .. tab-set::

                        .. tab-item:: From PyTorch

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From TensorFlow

                            .. code-block:: python

                                print("Hello World")

                        .. tab-item:: From JAX

                            .. code-block:: python

                                print("Hello World")

.. only:: rst

    Only on github readme

.. raw:: html

               </details>
            </blockquote>
        </details>
        
     </blockquote>
   </details>

   <h3>I'm using Ivy&ensp;<img class="dark-light" style="height: 1.75em; vertical-align:-40%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/ivy_logo_only_small.png"></h3>
   
Or you can use Ivy as a framework, breaking yourself (and your code) free from deciding which community to support, allowing anyone to run your code in their framework of choice!

.. code-block:: python

    import ivy

    # a simple image classification model
    class IvyNet(ivy.Module):
        def __init__(
            self,
            h_w=(32, 32),
            input_channels=3,
            output_channels=512,
            num_classes=2,
            data_format="NCHW",
            device="cpu",
        ):
            self.extractor = ivy.Sequential(
                ivy.Conv2D(input_channels, 6, [5, 5], 1, "SAME", data_format=data_format),
                ivy.GELU(),
                ivy.Conv2D(6, 16, [5, 5], 1, "SAME", data_format=data_format),
                ivy.GELU(),
                ivy.Conv2D(16, output_channels, [5, 5], 1, "SAME", data_format=data_format),
                ivy.GELU(),
            )

            self.classifier = ivy.Sequential(
                # since padding is "SAME", this would be image_height x image_width x output_channels
                ivy.Linear(h_w[0] * h_w[1] * output_channels, 512),
                ivy.GELU(),
                ivy.Linear(512, num_classes),
            )
            ivy.Module.__init__(self)

        def _forward(self, x):
            x = self.extractor(x)
            # flatten all dims except batch dim
            x = ivy.flatten(x, start_dim=1, end_dim=-1)
            logits = self.classifier(x)
            probs = ivy.softmax(logits)
            return logits, probs


After building your model in Ivy, you can set your favourite framework as the backend to use its operations under the hood!

.. code-block:: python

    ivy.set_backend("torch")
    model = IvyNet()
    x = torch.randn(1, 3, 32, 32)
    logits, probs = model(x)

.. code-block:: python

    ivy.set_backend("tensorflow")
    model = IvyNet()
    x = tf.random.uniform(shape=(1, 3, 32, 32))
    logits, probs = model(x)

.. code-block:: python

    ivy.set_backend("jax")
    model = IvyNet()
    x = jax.random.uniform(key, shape=(1, 3, 32, 32))
    logits, probs = model(x)

.. code-block:: python

    ivy.set_backend("numpy")
    model = IvyNet()
    x = np.random.uniform(size=(1, 3, 32, 32))
    logits, probs = model(x)

Last but not least, we can also build the training pipeline in pure ivy ⬇️

.. raw:: html

   <details>
   <summary><a>Let's define some helper functions first</a></summary>

.. code-block:: python

    # helper function for loading the dataset in batches
    def generate_batches(images, classes, dataset_size, batch_size=32):
        targets = {k: v for v, k in enumerate(np.unique(classes))}
        y_train = [targets[classes[i]] for i in range(len(classes))]
        if batch_size > dataset_size:
            raise ivy.utils.exceptions.IvyError("Use a smaller batch size")
        for idx in range(0, dataset_size, batch_size):
            yield ivy.stack(images[idx : min(idx + batch_size, dataset_size)]), ivy.array(
                y_train[idx : min(idx + batch_size, dataset_size)]
            )


    # helper function to get the number of current predictions
    def num_correct(preds, labels):
        return (preds.argmax() == labels).sum().to_numpy().item()


    # define a loss function
    def loss_fn(params):
        v, model, x, y = params
        y_pred, probs = model(x)
        return ivy.cross_entropy(y, probs), probs


.. raw:: html

   </details>

.. raw:: html

   <details>
   <summary><a>And train this model!</a></summary>

.. code-block:: python

    # train the model on gpu if it's available
    device = "cuda:0" if ivy.gpu_is_available() else "cpu"

    model = IvyNet(
        h_w=(28, 28),
        input_channels=1,
        output_channels=120,
        num_classes=num_classes,
        device=device,
    )
    model_name = type(model).__name__.lower()
    
    
    # training hyperparams
    optimizer= ivy.Adam(1e-4)
    batch_size = 64 
    num_epochs = 20
    num_classes = 10
    
    
    # training loop
    def train(images, classes, epochs, model, device, num_classes=10, batch_size=32):
        # training metrics
        epoch_loss = 0.0
        running_loss = 0.0
        fields = ["epoch", "epoch_loss", "training_accuracy"]
        metrics = []
        dataset_size = len(images)

        for epoch in range(epochs):
            train_loss, train_correct = 0, 0
            train_loop = tqdm(
                generate_batches(images, classes, len(images), batch_size=batch_size),
                total=dataset_size // batch_size,
                position=0,
                leave=True,
            )

            for xbatch, ybatch in train_loop:
                if device != "cpu":
                    xbatch, ybatch = xbatch.to_device("gpu:0"), ybatch.to_device("gpu:0")

                # since the cross entropy function expects the target classes to be in one-hot encoded format
                ybatch_encoded = ivy.one_hot(ybatch, num_classes)

                # update model params
                loss_probs, grads = ivy.execute_with_gradients(
                    loss_fn,
                    (model.v, model, xbatch, ybatch_encoded),
                    ret_grad_idxs=[[0]],
                    xs_grad_idxs=[[0]],
                )
                
                model.v = optimizer.step(model.v, grads["0"])

                batch_loss = ivy.to_numpy(loss_probs[0]).mean().item()  # batch mean loss
                epoch_loss += batch_loss * xbatch.shape[0]
                train_correct += num_correct(loss_probs[1], ybatch)

                train_loop.set_description(f"Epoch [{epoch + 1:2d}/{epochs}]")
                train_loop.set_postfix(
                    running_loss=batch_loss,
                    accuracy_percentage=(train_correct / dataset_size) * 100,
                )
            
            epoch_loss = epoch_loss / dataset_size
            training_accuracy = train_correct / dataset_size

            metrics.append([epoch, epoch_loss, training_accuracy])

            train_loop.write(
                f"\nAverage training loss: {epoch_loss:.6f}, Train Correct: {train_correct}",
                end="\n",
            )

        # write metrics for plotting
        with open(f"/{model_name}_train_summary.csv", "w") as f:
            f = csv.writer(f)
            f.writerow(fields)
            f.writerows(metrics)
            
            
    # assuming the dataset(images and classes) are already prepared in a folder      
    train(images, classes, num_epochs, model, device, num_classes = num_classes, batch_size = batch_size)


.. raw:: html

   </details>

|

Contributing
------------

We believe that everyone can contribute and make a difference. Whether it's writing code 💻, fixing bugs 🐛, 
or simply sharing feedback 💬, your contributions are definitely welcome and appreciated 🙌 

Check out all of our open tasks, and find out more info in our `Contributing guide <https://lets-unify.ai/docs/ivy/overview/contributing.html>`_ in the docs!

Join our amazing community as a code contributor, and help accelerate our journey to unify all ML frameworks!

.. raw:: html

   <a href="https://github.com/unifyai/ivy/graphs/contributors">
     <img src="https://contrib.rocks/image?repo=unifyai/ivy&anon=0&columns=20&max=100" />
   </a>

|

Community
------------

In order to achieve the ambitious goal of unifying AI we definitely need as many hands as possible on it! Whether you are a seasoned developer or just starting out, you'll find a place here! Join the Ivy community in our `Discord`_ 👾 server, which is the perfect place to ask questions, share ideas, and get help from both fellow developers and the Ivy Team directly!

Also! Feel free to follow us in `Twitter`_ 🐦 as well, we use it to share updates, sneak peeks, and all sorts of relevant news, certainly a great way to stay in the loop 😄

Can't wait to see you there!


Citation
--------

If you use Ivy for your work, please don't forget to give proper credit by including the accompanying `paper <https://arxiv.org/abs/2102.02886>`_ 📄 in your references. 
It's a small way to show appreciation and help to continue to support this and other open source projects 🙌

::

    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
