import atexit
from queue import Empty, Queue
from threading import Thread

import numpy as np
import tensorflow as tf

from rllab.sampler.utils import rollout
import pickle




__all__ = ['init_plot', 'update_plot']

thread = None
queue = None


class Plotter(Thread):
    def __init__(self, sess):
        super(Plotter, self).__init__()
        self.sess = sess

    def run(self):
        env = None
        policy = None
        max_length = None
        try:
            while True:
                msgs = {}
                # Only fetch the last message of each type
                with self.sess.as_default():
                    while True:
                        try:
                            msg = queue.get_nowait()
                            msgs[msg[0]] = msg[1:]
                        except Empty:
                            break
                    if 'stop' in msgs:
                        break
                    elif 'update' in msgs:
                        env, policy = msgs['update']
                    elif 'demo' in msgs:
                        param_values, max_length = msgs['demo']
                        policy.set_param_values(param_values)
                        if not self.sess._closed:
                            rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)
                    else:
                        if max_length:
                            if not self.sess._closed:
                                rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)
        except KeyboardInterrupt:
            pass


def _shutdown_worker():
    if thread:
        queue.put(['stop'])
        queue.task_done()
        queue.join()
        thread.join()


def _init_worker(sess=None):
    global queue, thread
    if queue is None:
        queue = Queue()
    if sess is None:
        sess = tf.get_default_session()
        thread = Plotter(sess)
        thread.daemon = True
        thread.start()
        atexit.register(_shutdown_worker)


def init_plot(env, policy,sess=None):
    _init_worker(sess)
    queue.put(['update', env, policy])
    queue.task_done()


def update_plot(policy, max_length=np.inf):
    queue.put(['demo', policy.get_param_values(), max_length])
    queue.task_done()
