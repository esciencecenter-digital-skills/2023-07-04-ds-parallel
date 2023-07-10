![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document day 2 (hello again!)

**2023-07-04 Parallel Python Workshop**

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

[toc]

----------------------------------------------------------------------------

This is the Document for today: [link](https://codimd.carpentries.org/pHojkc20QxisT7zC_9HDBg?both)

Collaborative Document day 1: [link](https://codimd.carpentries.org/pHojkc20QxisT7zC_9HDBg?both)

Collaborative Document day 2: [link](<https://codimd.carpentries.org/pHojkc20QxisT7zC_9HDBg?edit)

Collaborative Document day 3: [~~link~~](<url>)

Collaborative Document day 4: [~~link~~](<url>) 

## 👮Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a pink post-it note on your laptop lid. A helper will come to assist you as soon as possible.

## 🖥 Workshop website

[link](https://esciencecenter-digital-skills.github.io/2023-07-04-ds-parallel/)

🛠 Setup

Look up the [Setup section here](https://esciencecenter-digital-skills.github.io/2023-07-04-ds-parallel/)

<!--Download files

[link](<url>)-->

## 👩‍🏫👩‍💻🎓 Instructors

Johan Hidding, Leon Oostrum

## 🧑‍🙋 Helpers

 Giordano Lipari, Ewan Cahen  

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city
:::success
This document has been anonymised
:::

## 🧊 Icebreaker
Tell us your summer storm horror story!

## 🗓️ Agenda

<table class="table table-striped">
  <tr> <td>09:30</td>  <td>Welcome, icebreaker and recap</td> </tr>
  <tr> <td>09:45</td>  <td>Delayed evaluation</td> </tr>
  <tr> <td>10:30</td>  <td>Coffee break</td> </tr>
  <tr> <td>10:45</td>  <td>Data flow patterns</td></tr>
  <tr> <td>12:00</td>  <td>Tea break</td> </tr>
  <tr> <td>12:15</td>  <td>Coroutines and Asyncio</td> </tr>
  <tr> <td>13:00</td>  <td>Lunch</td></tr>      
  <tr> <td>14:00</td>  <td>Questions and Discussion</td> </tr>
  <tr> <td>14:45</td>  <td>Coffee break</td> </tr>
  <tr> <td>15:00</td>  <td>Computing fractals in parallel</td></tr>
  <tr> <td>15:45</td>  <td>Tea break</td> </tr>
  <tr> <td>16:30</td>  <td>Presentations of group work</td> </tr>
  <tr> <td>16:45</td>  <td>Post-workshop Survey</td> </tr>
  <tr> <td>17:00</td>  <td>Drinks</td> </tr>
</table>

![](https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Mandel_zoom_00_mandelbrot_set.jpg/1280px-Mandel_zoom_00_mandelbrot_set.jpg)

## 🏢 Location logistics
* Coffee and toilets are in the hallway, just outside of the classroom.
* If you leave the building, 
  be sure to be accompanied by someone from the escience center to let you back in through the groundfloor door
* For access to this floor you might need to ring the doorbell so someone can let you in
* In case of an emergency, you can exit our floor using the main staircase.
  Or follow green light signs at the ceiling to the emergency staircase.
* **Wifi**: Eduroam should work. Otherwise use the 'matrixbuilding' network, password should be printed out and available somewhere in the room.
* Logistics of the day:

## 🎓 Certificate of attendance
If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .

---
## 🔧 A. Exercises

### A1 Challenge: Run the workflows
:::info
Given this workflow:
```python
x_p = add(1, 2)
y_p = add(x_p, 3)
z_p = add(x_p, -3)
```
Visualize and compute `y_p` and `z_p` separately, how many times is `x_p` evaluated?

Now change the workflow:

```python
x_p = add(1, 2)
y_p = add(x_p, 3)
z_p = add(x_p, y_p)
z_p.visualize(rankdir="LR")
```

We pass the not-yet-computed promise `x_p`to both `y_p`and `z_p`. Now, only compute `z_p`, how many times do you expect `x_p` to be evaluated? Run the workflow to check your answer.

**Please do not share the answers in the collaborative document for this exercise**
:::


### A.2 Challenge: Understand our gather function
:::info
```python
@delayed
def gather(*arg):
    return list(arg)
```
Can you describe what the gather function does in terms of lists and promises?   
Hint: Suppose I have a list of promises, what does `gather` allow me to do?

**Please write down your answer below**
:::

:::success
1. : does it store the array into memory (so making a list) just after computing?
2. : execute the list of promises?
3. It maybe allows you to group all promises in a variable and call any of the promises by using the variable and an index
4. I think it allows you to turn a list of promises into a list of regular variables.
5. **It may allow you to group all promises so you minimize the number of evaluations needed by bringing them all under one workflow** 
6. put the promises in a list so you can index them easily
7. The gather function takes an arbitrary number of arguments (\*args) and returns a list of those arguments. By using the list() function, the tuple of arguments (args) is converted into a list.
8. Collects all promises together? One compute rules them all?
:::

## A.3 Challenge: generate all even numbers

:::info
Can you write a generator that generates all even numbers? Try to reuse `integers()`.  
Extra: Can you generate the Fibonacci sequence?
:::

:::success
```python=
def evens():
    for i in islice(integers(), 1, None, 2) if i < 25:
        print(i)
        if i > 25:
            break
```

```python=
def evens():
    a = 2
    while True:
        yield a
        a += 2

list(islice(evens(), 2, 5))
```

```python=
def even_numbers():
    int_gen = integers()
    while True:
        num = next(int_gen)
        if num % 2 == 0:
            yield num
            
def fib():
    # Initializing a,b
    a, b = 0, 1
    int_gen = integers()
    while True:
        yield a
        a, b = b, a + b
        next(int_gen)
```

```python=
def fibonacci():
    a, b = 0, 1
    while True:
        yield b
        tmp = b
        b = a + b
        a = tmp
```
:::

## A.4 Challenge: line numbers
:::info
Change `printer` to add line numbers to the output.
Use f-strings (format strings).
:::

## A.5 Gather multiple outcomes
:::info
We've seen that we can gather multiple coroutines using `asyncio.gather`.  
Now gather two `calc_pi` computations, and time them.
:::

:::success
| Name       | Single task [s] | Two tasks [s] |
|:---------- |:--------------- | -------------:|
|  | 0.023 s         |               |
|        | 0.083 s         |       0.119 s |
|    | 0.09 s          |        0.08 s |
|      | 0.112 s         |       0.117 s |
|        | 0.115 s         |       0.119 s |
|     | 0.141 s         |       0.158 s |
|     | 0.180 s         |               |
:::

## A.6 Mandelbrot

:::warning
### Complex numbers
Complex numbers are a special representation of rotations and scalings in the two-dimensional plane. Multiplying two complex numbers is the same as taking a point, rotate it by an angle $\phi$ and scale it by the absolute value. Multiplying with a number $z \in \mathbb{C}$ by 1 preserves $z$. Multiplying a point at $i = (0, 1)$ (having a positive angle of 90 degrees and absolute value 1), rotates it anti-clockwise by 90 degrees. Then you might see that $i^2 = (-1, 0)$. The funny thing is, that we can treat $i$ as any ordinary number, and all our algebra still works out. This is actually nothing short of a miracle! We can write a complex number

$$z = x + iy,$$

remember that $i^2 = -1$ and act as if everything is normal!
:::

:::info
This exercise uses Numpy and Matplotlib.

```python
from matplotlib import pyplot as plt
import numpy as np
```

We will be computing the famous [Mandelbrot fractal](https://en.wikipedia.org/wiki/Mandelbrot_fractal).


The Mandelbrot set is the set of complex numbers $$c \in \mathbb{C}$$ for which the iteration,

$$z_{n+1} = z_n^2 + c,$$

converges, starting iteration at $z_0 = 0$. We can visualize the Mandelbrot set by plotting the number of iterations needed for the absolute value $|z_n|$ to exceed 2 (for which it can be shown
that the iteration always diverges).

![The whole Mandelbrot set](https://esciencecenter-digital-skills.github.io/parallel-python-workbench/fig/mandelbrot-all.png)

We may compute the Mandelbrot as follows:

```python
max_iter = 256
width = 256
height = 256
center = -0.8 + 0.0j
extent = 3.0 + 3.0j
scale = max((extent / width).real, (extent / height).imag)

result = np.zeros((height, width), int)
for j in range(height):
    for i in range(width):
        c = center + (i - width // 2 + (j - height // 2)*1j) * scale
        z = 0
        for k in range(max_iter):
            z = z**2 + c
            if (z * z.conjugate()).real > 4.0:
                break
        result[j, i] = k
```

Then we can plot with the following code:

```python
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plot_extent = (width + 1j * height) * scale
z1 = center - plot_extent / 2
z2 = z1 + plot_extent
ax.imshow(result**(1/3), origin='lower', extent=(z1.real, z2.real, z1.imag, z2.imag))
ax.set_xlabel("$\Re(c)$")
ax.set_ylabel("$\Im(c)$")
```

Things become really loads of fun when we start to zoom in. We can play around with the `center` and `extent` values (and necessarily `max_iter`) to control our window.

```python
max_iter = 1024
center = -1.1195 + 0.2718j
extent = 0.005 + 0.005j
```

When we zoom in on the Mandelbrot fractal, we get smaller copies of the larger set!

![Zoom in on Mandelbrot set](https://esciencecenter-digital-skills.github.io/parallel-python-workbench/fig/mandelbrot-1.png)

**Exercise**
Make this into an efficient parallel program.     What kind of speed-ups do you get?
:::

:::info
- Best practice in Numba teaches us that arrays should be allocated outside Numba optimised functions. So a better pattern is (in quasi-python):

```python=
@numba.jit(nopython=True, nogil=True)
def compute_my_stuff(output, ... other args):
    width, height = output.shape
    for i in range(width):
        for j in range(height):
            ...

if __name__ == "__main__":
    result = np.zeros((width, height), np.uint32)
    compute_my_stuff(result, ... other args)
```
:::

:::info
Don't be afraid of trying features that we didn't discuss! Check out the documentation of Numba: https://numba.pydata.org/numba-doc/latest/index.html
See if you can find features that may be useful to you and try them out.
:::

---
## 🧠 B. Collaborative Notes

:::warning
**Full disclosure**
Updated as promptly as possible, wi-fi permitting.  
Please bear with us if the notes lag a bit behind.
:::

### 1. Delayed evaluation

```python
from dask import delayed

@delayed
def add(a, b):
    result = a + b
    print(f"{a} + {b} = {result}")  # f-strings = use variables inside a string
    return result

# a delayed function promises a value to be computed later
x_p = add(1, 2)
type(x_p)  # it is just an object
x_p.compute()  # ...like dask.array
```

```python
# demo of 'promises'
x_p = add(1, 2)
y_p = add(x_p, 3)
z_p = add(x_p, y_p)
z_p.visualize(rankdir='LR')
```

:::info
The **exercise** is in the section A, above in this document, at line https://codimd.carpentries.org/pHojkc20QxisT7zC_9HDBg?view#A1-Challenge-Run-the-workflow

First pass:
```python
x_p = add(1, 2)
y_p = add(x_p, 3)
z_p = add(x_p, -3)

# visualize the workflows
y_p.visualize()
z_p.visualize()

# evaluate the functions
y_p.compute()
z_p.compute()
```

Second pass:
```python
x_p = add(1, 2)
y_p = add(x_p, 3)
z_p = add(x_p, y_p)
z_p.visualize()
z_p.compute()  # dask reuses results as much as possible
```
:::

```python
def mult(a ,b):
    return a * b

mult(2, 3)
x_p = delayed(mult)(2,3)  # no decorator? mind the parentheses
type(x_p)
```

```python
# variadic arguments (those starting with star~~t~~)

def add(*args):  # any number of arguments will do
    return sum(args)

print(add(1, 2))
print(add(1, 2, 3, 4))

# another *-suffixed feature: tuple-unpacking

numbers = [1, 2, 3, 4] # a list, obviously
add(*numbers)  # list is converted to list of arguments
```

```python
@delayed
def gather(*arg):
    return list(arg)
```

:::info
The **exercise** is in the section A in this document, at line https://codimd.carpentries.org/pHojkc20QxisT7zC_9HDBg?view#A2-Challenge-Understand-gather

```python
# shorthand for gather(add(1,1), add(2,2), ...)
def add(*args):  # any number of arguments will do
    return sum(args)

x_p = gather(*(add(n,n) for n in range(10)))
x_p.visualize()
# dask.delayed is able to recognize there is parallelism


x_p.compute()
# ...although the speed up may not be visible
```
:::

:::warning
**Warning**
`dask.delayed` uses threads by default, so speed-up depends on having the GIL lifted (see yesterday)
:::

```python
# delay the vanilla calc_pi
import random

@delayed
def calc_pi(N):
    """docstrings definitely late"""
    M =0
    for i in range(N):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 < 1.0:
            M += 1
    return 4 * M / N

@delayed
def mean(*args):
    return sum(args)/len(args)
   
pi_p = mean(*(calc_pi(10**6) for i in range(4)))

pi_p.visualize()
pi_p.compute()
# use Numba to lift the GIL 
```

:::warning
**Note of caution**
To combine the `numba.jit` and `dask.delayed` decorators, you need to JIT the function first and only then make it delayed. So:

```python=
# the @delayed decorator would go in front here,
# but it is better in this case to keep it separate
# after all, we may want to reuse calc_pi_nb
# @delayed

@numba.jit(nopython=True, nogil=True)
def calc_pi_nb(N):
    """Compute the value of pi by drawing random numbers
    from a uniform distribution."""
    M =0
    for i in range(N):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 < 1.0:
            M += 1
    return 4 * M / N

calc_pi_delayed = delayed(calc_pi_nb)

@delayed
def mean(*args):
    return sum(args) / len(args)

def calc_pi_parallel(N, repeat):
    return mean(*(calc_pi_delayed(N) for _ in range(repeat)))
```

Now you can compute $\pi$ in parallel using

```python
calc_pi_parallel(10*6, 100).compute()
```
Time this!
:::

:::info
#### We resumed at 10:55
:::

#### Digression on `itertools`, tools for iterators


`Itertools` is a library of the Python standard libraries:

```python
[x**2 for x in range(4)]  # list comprehension, classic

result = (x**2 for x in count(40))  # generator
next(result)  # spans the items in the generator one at at time > providing memory savings for intensive project

result = (x**2 for x in count())  # generator over an infinite sequence

```

`dask.bag` provides scaled-up capabilities of the iterators, gauged for dataset sizes in the order of TB


```python
# python code coming
```
---

### 2. ASyncIO (tricky but not horrible!)

#### 2.1 Generalities and preparations

Illustration of conventional functions and asyncio coroutines.

```python
from itertools import count
x = count()
next(x)

def integers():
    a = 1
    while True:
        yield
        a += 1
        
for i in integers():
    print(i)
    if i > 10:
        break
```

```python
from itertools import islice

lst = list(range(1, 10))
lst[2:5]
islice(integers(), 2, 5)  # not a computation
list(islice(integers(), 2, 5))

```

:::info
The **exercise** is in the section A in this document, at line https://codimd.carpentries.org/pHojkc20QxisT7zC_9HDBg?view#A3-Challenge-generate-all-even-numbers
```python
def evens1():
    a = 2
    while True:
        yield a
        a +=2
        
def evens2():
    return(i for i in integers() if i % 2 ==0)

def evens3()
    return filter(lambda i: i % 2 == 0, count(1))

def evens4():
    for i in integers():
        if i % 2 == -:
            yield i

def fib(a, b):
    while True:
        yield a
        a, b = b, a + b  # implements trick to swap numbers

# test functions (target True)
print(list(islice(integers(), 2, 5)) == [6, 8, 10])
print(list(islice(fib(1,1), 10)) == [1, 1, 2, 3, FILL])
```
:::


```python
# this is not a function, it's a coroutine
def printer():
    while True:
        x = yield  # this receives values
        print(x)
        
p = printer()
next(p)

p.send("Mercury")
p.send("Venus")
p.send("Earth")
```

:::info
The **exercise** is in the section A in this document, at line https://codimd.carpentries.org/pHojkc20QxisT7zC_9HDBg?view#A4-Challenge-line-numbers
```python
def printer():
    lineno = 1
    while True:
        x = yield 
        print(f"{lineno:03} {x}")
        lineno += 1

p = printer()
next(p)

p.send("Mercury")
p.send("Venus")
p.send("Earth")

```
:::

#### 2.2 Action with ASyncIO

```python
import asyncio

# asynchronous code routine
async def counter(name):
    """print some silly output"""
    for i in range(5):
        print(f"{name:<10} {i:03}")
        await asyncio.sleep(0.2) # collaborative multitasking = I give away control to concurrent task, but come back to me for a check
        
await counter("Venus")
```

:::warning
await in jupyter lab is a kind of magic
:::

```python

counter("Mars")  # Juoyter returns the object type

# serial
await counter("Venus")
await counter("Earth")

# alternation that mimicks parallelism
await asyncio.gather(counter("Earth"), counter("Venus"))
```

```python
# uneven asyncio tasks gathered

import asyncio

# asynchronous code routine
async def counter_time(name, time):
    """print some silly output"""
    for i in range(5):
        print(f"{name:<10} {i:03}")
        await asyncio.sleep(time) 
        
await asyncio.garther(counter_timed("Earth", 0.5), counter_timed("Moon", 0.1))
```

```python
# contagiousness of ASyncIO: all has to be structured in asyncio-speak
import asyncio

async def main():
    ...
    
if __name__ == "__main__":
    asyncio.run(main())

```

The following throws an error because two magics interact badly:
```python
%%timeit  # a well-known magic
await counter("Mars")  # this is a magic in disguise
```

:::info
#### We resumed at 12:20
:::

Snippet 'free of charge':

```python
from dataclasses import dataclass
from typing import Optional
from time import perf_counter
from contextlib import asynccontextmanager


@dataclass
class Elapsed:
    time: Optional[float] = None

@asynccontextmanager
async def timer():
    e = Elapsed()
    t = perf_counter()
    yield e
    e.time = perf_counter() - t
```


```python
# our baseline
import random
import numba

@nb.jit(nopython=True, nogil=True)
def calc_pi_numba_nogil(N):
    M = 0
    # print('Start')
    for i in range(N):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 < 1:
            M += 1
    # print('End')
    return 4 * M / N

async with timer() as t:
    await asyncio.to thread(calc_pi, 10**7)
print(f"That took {t.time} seconds")
    
```

:::info
The **exercise** is in the section A in this document, at line https://codimd.carpentries.org/pHojkc20QxisT7zC_9HDBg?view#A5-Gather-multiple-outcomes

```python
async with timer() as t:
    await asyncio.gather(
        asyncio.to thread(calc_pi, 10**7)
        asyncio.to thread(calc_pi, 10**7)
    )
print(f"Those took {t.time} seconds")
```
:::

```python
# create a series of threads with *-notation
async def calc_pi_parallel(N, repeat):
    results = await asyncio.gather(
        *(asyncio.to_thread(calc_pi,N) for _ in range(repeat))
    return sum(result) / repeat
        
# 1 run of size 10^8
async with timer() as t:
    pi = await asyncio.to_thread(calc_pi, 10**8)
    print(f"Value of pi: {pi}")
print(f"That took {t.time} seconds")

# 10 runs of size 10^7
async with timer() as t:
    pi = await calc_pi_paralel(10**7, 10)
    print(f"Value of pi: {pi}")
print(f"Those took {t.time} seconds")
```
#### Johan's script with all we did

To see if/how perfomance in Python differs from JupyterLab:

```python
import asyncio
import random
import numba
from dataclasses import dataclass
from typing import Optional
from time import perf_counter
from contextlib import asynccontextmanager

@dataclass
class Elapsed:
    time: Optional[float] = None

@asynccontextmanager
async def timer():
    e = Elapsed()
    t = perf_counter()
    yield e
    e.time = perf_counter() - t

@numba.jit(nopython=True, nogil=True)
def calc_pi(N):
    M = 0
    for i in range(N):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if x**2 + y**2 < 1.0:
            M += 1
    return 4 * M / N

async def calc_pi_parallel(N, repeat):
    results = await asyncio.gather(
        *(asyncio.to_thread(calc_pi, N) for _ in range(repeat)))
    return sum(results) / repeat

async def main():
    async with timer() as t:
        pi = await calc_pi_parallel(10**7, 10)
        print(f"Value of pi: {pi}")
    print(f"that took {t.time} seconds")
    async with timer() as t:
        pi = await calc_pi_parallel(10**7, 10)
        print(f"Value of pi: {pi}")
    print(f"that took {t.time} seconds")
    async with timer() as t:
        pi = await calc_pi_parallel(10**7, 10)
        print(f"Value of pi: {pi}")
    print(f"that took {t.time} seconds")


if __name__ == "__main__":
    asyncio.run(main())
```

:::info
#### We resumed at 14:00
:::

### 3. The Mandelbrot (almond-bread?) fractal

```python
(5 + 3j) * (3 + 1j)
j**2

# demo of Mandelbrot sequence
# start with c and z reals, then imaginary
c = 1j
z = 0

for _ in range(10):
    z = z**2 + c
    print(z)
    
# Leon shows nice visualisations with his own function
```

:::info
The **exercise** is in the section A in this document, at line https://codimd.carpentries.org/pHojkc20QxisT7zC_9HDBg?view#A6-Mandelbrot
:::


---
## 📚 Resources
* [Post-workshop survey](https://www.surveymonkey.com/r/T3XDWPB)
* [eScience Center workshops and other events](https://www.esciencecenter.nl/events/)
* [eScience Center newsletter signup link](http://eepurl.com/dtjzwP)
* [Deploying dask on a (e.g. slurm) cluster](https://docs.dask.org/en/stable/deploying.html)

