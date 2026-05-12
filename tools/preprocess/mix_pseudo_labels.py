"""
Given 2 sets of pseudo labels, for each gtxy, we have 2 pseudo labels with confidence scores.
Load the ground truth pseudo labels, calc the IoU between the 2 pseudo labels,
and keep the one with higher score:
  score = IoU * confidence
"""
