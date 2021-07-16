#!/usr/bin/env python3
import csv, math
from numpy import array, arctan2

class Sentiment:
    """A simple class to hold word sentiments and operations on sentiments"""
    def __init__(self, word, valence=None, arousal=None, dominance=None):
        self.word = word
        self.valence = valence
        self.arousal = arousal
        self.dominance = dominance
        self._vector = self.as_array()

    def update(self, *, valence=None, arousal=None, dominance=None):
        """update valence, arousal, and/or dominance. keywords required"""
        self._vector = None
        if valence:
            self.valence = valence
        if arousal:
            self.arousal = arousal
        if dominance:
            self.dominance = dominance

    @property
    def angle(self):
        return arctan2(self.arousal, self.valence) * 180/math.pi

    def __str__(self):
        return f"Sentiment: {self.word}(valence={self.valence}, arousal={self.arousal}, dominance={self.dominance})"

    def __repr__(self):
        return f"Sentiment({self.word},{self.valence},{self.arousal},{self.dominance})"

    @property
    def vector(self):
        """Return the vector"""
        if self._vector is None:
            self._vector = self.as_array()
        return self._vector

    def as_array(self):
        """Create an array representation of this sentiment"""
        return array([self.valence, self.arousal, self.dominance])


def load_sentiments(lang_tsv):
    """Given a language tsv file for sentiments, load them into a dictionary
    lexicon = nrc.load_sentiments("NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt")
    """
    lexicon = {}
    with open(lang_tsv) as data:
        dataset = csv.reader(data, delimiter="\t", quotechar='"')
        next(dataset) # skip the header column
        for row in dataset:
            lexicon[row[0]] = Sentiment(row[0], *[2*float(v) - 1 for v in row[1:]])
    return lexicon
