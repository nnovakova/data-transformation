+++
title="Python Type Checkers"
date=2020-11-07

[taxonomies]
categories = ["Programming"]
tags = ["python", "static typing"]

[extra]
toc = true

+++

As I am working with Java code, I also find static types in my project as a useful thing. They help me to reason about my code
and avoid simple bugs, for example if I pass incorrect argument to some method or function. 

<!-- more -->

[Python](https://en.wikipedia.org/wiki/Python_%28programming_language%29) is a dynamically-typed language. 
This property makes it faster to write code, but NOT necessarily easier to read. Type annotations and type checkers appeared in Python not so long time ago. Today, there are several type checkers in Python eco-system:

- [**Mypy**](http://mypy-lang.org/) - first static type checker for Python, initiated in 2012 and still in active development. Uses type annotations to check program correctness.
- [**Pyre**](https://pyre-check.org/docs/getting-started) - created by Facebook, uses type annotations. User can switch on the "strict mode" to raise errors on missing type annotations.
- [**Pytype**](https://google.github.io/pytype/) - created by Google. Uses type inference technique, instead of only using type annotations.

There are might be other type checkers, but I mainly looked into those. In this blog-post, I only discuss **Mypy** type checker. 

## Mypy

All type checkers in Python are based on the successful work being done in PEPs (Python Enhancement Proposal):
- [PEP 484](https://www.python.org/dev/peps/pep-0484/) - Type Hints
- [PEP 526](https://www.python.org/dev/peps/pep-0526/) - Syntax for Variable Annotations
- [PEP 612](https://www.python.org/dev/peps/pep-0612/) - Parameter Specification Variables

Type annotations can be used for variables, method arguments and return types. 
There are many other type features like generics, but I will skip that in this blog-post.

Mypy comes as command line tool plus several other tools for typing, library stubs and more. Let's install it and try on a simple program.

### Installation

Mypy requires Python version >= 3.5. You can install it using [pip](https://pip.pypa.io/en/stable/installing/):

```bash
python3 -m pip install mypy
```

### Usage

Once Mypy is in your environment, you can run it via command line interface to type check you Python source file, module or entire workspace.

```bash
mypy <myfile>.py
```

Below example has two lines where we have bugs:

```python
# pizza.py

class Pizza:
    def __init__(self, name: str, count: int):
        self.name = name
        self.count = count


def order_pizza(name: str, count: int) -> Pizza:
    return Pizza(count, name) # bug: count and name swapped

order_pizza(2, "spinacci") # bug: count and name swapped
```

Let's run Mypy to catch these bugs:

```bash
mypy pizza.py
pizza.py:8: error: Argument 1 to "Pizza" has incompatible type "int"; expected "str"
pizza.py:8: error: Argument 2 to "Pizza" has incompatible type "str"; expected "int"
pizza.py:11: error: Argument 1 to "order_pizza" has incompatible type "int"; expected "str"
pizza.py:11: error: Argument 2 to "order_pizza" has incompatible type "str"; expected "int"
Found 4 errors in 1 file (checked 1 source file)
```

As you can see, Mypy reports that we pass `int` value to a function where string value is expected and vice versa.
Let's fix these bugs and run Mypy again:

```python
def order_pizza(name: str, count: int) -> Pizza:
    return Pizza(name, count)

order_pizza("spinacci", 2)
```
``` bash
mypy pizza.py
Success: no issues found in 1 source file
```

If we miss function argument(s), Mypy reports it as well:

```python
def order_pizza(name: str, count: int) -> Pizza:
    return Pizza(name, count)

order_pizza("spinacci")
```

```bash
 mypy pizza.py
pizza.py:10: error: Too few arguments for "order_pizza"
Found 1 error in 1 file (checked 1 source file)
```

In case we put wrong return value, then we get an error as well:

```python
def order_pizza(name: str, count: int) -> Pizza:
    return "Pizza(name, count)"

order_pizza("spinacci", 2)
```

```bash
mypy pizza.py
pizza.py:8: error: Incompatible return value type (got "str", expected "Pizza")
Found 1 error in 1 file (checked 1 source file)
```

Mypy can also type check Python 2 code.

### 3rd party libraries

 Mypy provides type definitions via the [typeshed repository](https://github.com/python/typeshed), 
 which contains library stubs for the Python builtins, the standard library, and selected third-party packages.
 Type stub is python code that contains only class, function definitions with types. 
 Mypy can automatically find type stubs for a library if it has a type stub. 
 Otherwise, Mypy will report an error that types are missing for a specific module:

 ```python
 # test.py
 from pyspark.sql import DataFrame
 ```

 ```bash
 mypy test.py
test.py:1: error: Skipping analyzing 'pyspark': found module but no type hints or library stubs
test.py:1: note: See https://mypy.readthedocs.io/en/latest/running_mypy.html#missing-imports
Found 1 error in 1 file (checked 1 source file)
 ```

 As you can see above, `pyspark` package does not provide type stubs on its own. However, there is open-source package 
[pyspark-stubs](https://pypi.org/project/pyspark-stubs/), which we need to install to add Mypy type stubs for PySpark to our environment:

```bash
pip install pyspark-stubs
```
Once it is installed, we can successfully run Mypy and use PySpark types in our code.

```bash
mypy test.py
Success: no issues found in 1 source file
```

#### Configuration

Mypy can read user-defined configuration from `mypy.ini` file. One the convenient use case is to disable type checking for a specific library or
its module:

```ini
# mypy.ini
[mypy]

[mypy-pyspark.sql.*]
ignore_missing_imports = True
```

Above config sections ignore missing imports for `pyspark.sql` module of pyspark library.

#### Type Stubs for an external library

There are a lot of libraries which do not have type stubs as of today, neither in `typeshed` repository nor as separate `pip` package. 
In this case, one can create own type stubs using Mypy and keep in own project repository. 

There are a couple examples how to create your stubs and install them, so that Mypy can see your own stubs for specific library:

- [Mypy wiki](https://github.com/python/mypy/wiki/Creating-Stubs-For-Python-Modules)
- [PyCharm - Create a stub file](https://www.jetbrains.com/help/pycharm/stubs.html#create-stub-external)

Let's try to provide type stub for one of the [Plotly](https://plotly.com/python/getting-started/) module.

1. Create local hierarchy that mirrors python file naming convention:

```bash
|myproject
├── stubs
│   └── plotly
│       ├── __init__.pyi
│       └── graph_objs
│           └── __init__.pyi
└── test.py
```

2. We put only one class constuctor definition as an example:

```python
# stubs/plotly/graph_objs/__init__.pyi

from typing import Any

class Scatter:
    def __init__(self, x: Any, y: Any, name: str, text: Any): ...
```

3. In a caller script `test.py` we instantiate `Scatter` class:

```python
# test.py
import plotly.graph_objs as go

plot = go.Scatter(0, 1, "", 3)
```

4. Define environment variable to point `stubs` directory:

```bash
export MYPYPATH=~/myproject/stubs
```

5. Finally, we run type checker:

```bash
mypy test.pyt
Success: no issues found in 1 source file
```

`Scatter` constructor definition was provided from our own type stubs.

### Stubgen

Above example is based on manual type stub creation, however Mypy project also provides [stubgen](https://mypy.readthedocs.io/en/stable/stubgen.html) tool to create a stub draft automatically.
Auto-generated stub requires then manual updates to get rid of `Any` types, which are inferred in many cases.

## Conclusion

As a Data Scientist I do not really define new types, but rather using a lot of Python libraries to write a sequence of steps as a script.
I do not really write complex systems with a lot of own libraries. So Type Checkers are not really critical to me, however it is nice to
have such tool around. Type annotations are great, especially if the help IDEs to provide better development experience via auto-completion 
and type hints. Sometimes I can annotate my own functions, this serves me as a documentation. 
It is quite useful, when I am reading my code in a couple of months in future.
