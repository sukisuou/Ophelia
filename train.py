import numpy as np
import tensorflow as tf
from model import build_transformer

model = build_transformer()
model.summary()