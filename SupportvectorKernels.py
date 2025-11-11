import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
X=np.linspace(-5, 5, 100)
y=np.sqrt(10**2-X**2)
y=np.hstack([y,-y])
x=np.hstack([x,-x])
