#!/usr/bin/env python3

import onnxruntime as rt
import numpy as np
import scipy.spatial.distance as dist


class EncoderModel:
    def __init__(self):
        self.sess = rt.InferenceSession('encoder/encoder_model.onnx', providers=rt.get_available_providers())

        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def embed(self, acts):
        traj_raw = np.hstack([np.zeros(150 - len(acts[-150:])), acts[-150:]])
        traj = np.broadcast_to(np.arange(18), (1, 150, 18)) == np.broadcast_to(traj_raw, (1, 18, 150)).swapaxes(1, 2)
        traj = traj.astype(np.int64)

        return self.sess.run([self.out_name], {self.in_name: traj})[0][0]

    def logit_dist(self, act_embed, lang_embed):
        d = dist.cosine(act_embed[0], lang_embed)
        # logistic regression fit on values, formatted as (related, unrelated) logit pair
        return (1.511 - 4.35 * d, 0)

    def predict(self, acts, langs):
        acts = acts[0]
        langs = langs[0]

        return [self.logit_dist(self.embed(acts), langs)]
