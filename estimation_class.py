#!/usr/bin/env python

#####################################################################################
#
# Copyright 2022 Quantinuum
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
#####################################################################################

"""Base estimation class."""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from utils import convert_error_dict


class QVEstimate(object):
    """
    Estimate QVT heavy output probability over set of qubits, error magnitudes.
    
    """
    def __init__(self,
                 qubit_list: list,
                 optimization_lvl: list,
                 error_name: str,
                 error_list: list):
        """
        Returns probability of success for range of QV experiments.
        
        Args:
            qubit_list: list of qubits to test 
            optimization_lvl:  preset optimization level
            error_name: name of error
            error_list: list of error magnitudes
              
        """
        self.qubit_list = qubit_list
        self.optimization_lvl = optimization_lvl
        self.error_name = error_name
        self.error_dict = {
            e: convert_error_dict(e, self.error_name) 
            for e in error_list
        }
        self.ideal_success = {}
        self.act_success = {}
        
    def table(self,
              yvals: str):
        """Returns pandas DataFrame table of results."""
        if yvals == 'act':
            yvals = self.act_success
            
        elif yvals == 'ideals':
            yvals = self.ideal_success
            
        else:
            print(f'No data for {yvals}')
            yvals = None

        out = pd.DataFrame(
            {n: {e: np.mean(yvals[n, e])
             for e in self.error_dict}
             for n in self.qubit_list}
        )

        return out

    def plot(self,
             yvals: str):
        """
        Plot results.
        
        Args:
            yvals: which value to plot
        
        """
        if yvals == 'act':
            yvals = self.act_success
            
        elif yvals == 'ideals':
            yvals = self.ideal_success
            
        else:
            print(f'No data for {yvals}')
            yvals = None
                 
        data = np.array(
            [[np.mean(yvals[n, e])
              for e in self.error_dict]
             for n in self.qubit_list]
        )

        if min(self.error_dict) < 0.001:
            error_labels = [f'{e:.2e}' for e in self.error_dict]
            
        else:
            error_labels = [f'{e:.3f}' for e in self.error_dict]
            
        
        fig, ax = plt.subplots()
        CS = ax.imshow(
            data.T,
            aspect='equal', 
            interpolation=None,
            origin='upper', 
            cmap=plt.get_cmap('plasma'), 
            vmin=0.45, 
            vmax=1
        )
       
        cbar = fig.colorbar(CS, extend='both')
        cbar.ax.tick_params(labelsize=14)
        
        ax.set_xticks(list(range(len(self.qubit_list))))
        ax.set_yticks(list(range(len(error_labels))))
        
        ax.set_xticklabels(self.qubit_list)
        ax.set_yticklabels(error_labels)
        
        ax.set_xlabel('qubits')
        ax.set_ylabel('error magnitude')
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        cbar.fontsize = 14
        ax.labelsize = 16

    def save(self,
             name: str):
        """Save important information."""

        with open(name + '.pkl', 'wb') as f:
            pickle.dump(self.ideal_success, f)
            pickle.dump(self.act_success, f)
            pickle.dump(self.qubit_list, f)
            pickle.dump(self.error_name, f)
            pickle.dump(self.error_dict, f)
            pickle.dump(self.optimization_lvl, f)
            
    @staticmethod
    def open(name: str):
        """Load object."""

        with open(name + '.pkl', 'rb') as f:
            ideal_success = pickle.load(f)
            act_success = pickle.load(f)
            qubit_list = pickle.load(f)
            error_name = pickle.load(f)
            error_dict = pickle.load(f)
            optimization_lvl = pickle.load(f)
            
        m = QVEstimate(
            qubit_list,
            optimization_lvl,
            error_name,
            [],
        )

        m.error_dict = error_dict
        m.ideal_success = ideal_success
        m.act_success = act_success
        
        return m
