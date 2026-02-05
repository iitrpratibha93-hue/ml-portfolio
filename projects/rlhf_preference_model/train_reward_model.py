#!/usr/bin/env python3
"""
Train a preference (reward) model on (prompt, chosen, rejected) triples.
Loss: -log σ(r_chosen - r_rejected). Saves best checkpoint by eval loss.
"""

import argparse
import json
import logging
import os
import random

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import PreferenceDataset
from model import RewardModel
