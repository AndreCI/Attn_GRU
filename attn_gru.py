#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:27:27 2017

@author: andre
"""
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import RNNCell, _linear
import tensorflow as tf

class AttnGRUCell(RNNCell):
    
    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(AttnGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation=activation or math_ops.tanh
        self._kernel_initializer=kernel_initializer
        self._bias_initializer=bias_initializer
        self._u = None
        self._us = None
        self._updated = False
        self._idx = None
        
    @property
    def state_size(self):
        return self._num_units
    
    @property
    def output_size(self):
        return self._num_units
    
    def update_attention_gate(self, new_attention):
        """
        Feed the next attention weight to the cell.
        """
        if(self._updated):
            raise Warning("The attention has already been updated. You should call the cell before feeding a new attention.")
        if(self._us is not None):
            raise ValueError("The next attention has already been fed. Use multiple calls of update_attention_gate or one call fo update_all_attention_gate.")
        self._u = new_attention
        self._updated = True
        
    def update_all_attention_gate(self, new_attentions):
        """
        Feed all the attention weight to the cell
        input: new_attentions [batch_size, max_time] for scalar attentions or [batch_size, max_time, hidden_size] (Not implemented yet!)
        """
        if(self._u is not None):
            raise ValueError("The next attention has already been fed. Use multiple calls of update_attention_gate or one call fo update_all_attention_gate.")
        self._us = new_attentions
        self._updated = True
        self._idx = 0
        
    def call(self, inputs, state):
        """
        Gated recurrent unit with attention gate (AttnGRU) with nunits cells.
        """
        if(self._u is None and self._us is None):
            raise ValueError("No value for the attention gate has been specified. You need to call update_attention_gate(new_attention) before calling an AttnGRUCell.")
        if(self._updated == False and self._us is None):
            raise Warning("The attention gate has not been updated. Before each call, you are supposed to call update_attention_gate(new_attention).")
        if(self._us is None):
            self._updated = False
            u = self._u
        else:
            u = tf.slice(self._us, [0,self._idx], [-1, self._idx + 1])
        
        with vs.variable_scope("gates"):
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                dtype = [a.dtype for a in [inputs, state]][0]
                bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
            r = math_ops.sigmoid(_linear([inputs, state], self._num_units, True, bias_ones, self._kernel_initializer))
        with vs.variable_scope("candidate"):
            c = self._activation(_linear([inputs, r * state], self._num_units, True, self._bias_initializer, self._kernel_initializer))
        new_h = u * state + (1 - u) * c
        return new_h, new_h
            
            
            
            
            




