import tempfile
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import subprocess
from pymatgen.core import Composition, Element

class experiment():
    def __init__(self):
        self.n_experiment = 0
        self.target_bandgap = 0
        self.bandgaps = []
        self.crystals = []
        self.learning_curve = []
        self.db_path = "./database/db_true.csv"
        self.prior_path = './database/prior.csv'
        self.db_df = pd.read_csv(self.db_path, index_col=0)
        self.losses = self.db_df['indirect_bandgap (eV)']
        
    def set_target(self, target_bandgap):
        self.target_bandgap = target_bandgap
        self.losses = np.abs(self.db_df['indirect_bandgap (eV)'] - target_bandgap)
        self.n_experiment = 0
        self.bandgaps = []
        self.crystals = []
        self.learning_curve = []
        print("We've set the target value as "+str(target_bandgap)+"[eV]. Now let's begin exploration!")
    
    def prior_information(self, target_value):
        df_prior = pd.DataFrame([])
        indice = np.array(range(len(self.db_df)))
        np.random.shuffle(indice)
        count = 0
        for idx in indice:
            bandgap = self.db_df.iloc[idx, 1]
            if (bandgap > (target_value * 1.2)) or (bandgap < (target_value * 0.8)):
                count += 1
                df_prior = pd.concat([df_prior, self.db_df.iloc[idx, :]], axis=1)
                if count > 99:
                    break
        df_prior = df_prior.T
        df_prior = df_prior[df_prior.columns[::-1]]
        df_prior.to_csv('prior.csv')
        display(df_prior)
    
    def set_search_table(self):
        df_prior = pd.read_csv(self.prior_path, index_col=0)
        display(df_prior)
        return df_prior
    
    def search_atoms(self, inclusive_query, exclusive_query):
        df_query = pd.concat([self.db_df.iloc[:,0], self.db_df.iloc[:,2:8]], axis=1)
        if len(inclusive_query) > 0:
            for atom in inclusive_query:
                indice = []
                for idx, crystal_name in enumerate(list(df_query['crystal_name'])):
                    comp = Composition(crystal_name)
                    if atom in [str(e) for e in comp.elements]:
                        indice.append(idx)
                df_query = df_query.iloc[indice, :]

        if len(exclusive_query) > 0:
            for atom in exclusive_query:
                indice = []
                for idx, crystal_name in enumerate(list(df_query['crystal_name'])):
                    comp = Composition(crystal_name)
                    if not atom in [str(e) for e in comp.elements]:
                        indice.append(idx)
                df_query = df_query.iloc[indice, :]

        print(str(len(df_query))+' data was found in the database to be explored.')
        display(df_query)
        return df_query

    def search_by_electronegatvity(self, df, sort_by=True):
        electronegs = []
        names = list(df['crystal_name'])
        for crystal_name in names:
            comp = Composition(crystal_name)
            electronegs.append(comp.average_electroneg)

        df = pd.DataFrame([names, electronegs], index=['crystal_name', 'electronegativity']).T
        df = df.sort_values(by=['electronegativity'], ascending=sort_by)
        display(df)
        
    def search_by_atomin_fraction(self, element, df, sort_by=True):
        atomic_fractions = []
        names = list(df['crystal_name'])
        for crystal_name in names:
            comp = Composition(crystal_name)
            atomic_fractions.append(comp.get_atomic_fraction(Element("Li")))

        df = pd.DataFrame([names, atomic_fractions], index=['crystal_name', 'atomic_fraction']).T
        df = df.sort_values(by=['atomic_fraction'], ascending=sort_by)
        display(df)
    
    def show_details(self, crystal_name):
        comp = Composition(crystal_name)
        print('chemical formula: ' + comp.reduced_formula)
        print('valence: ' + str(comp.add_charges_from_oxi_state_guesses().as_dict()))
        print('average electronegativity: '+str(comp.average_electroneg))
        print('atomic fraction:')
        for e in comp.elements:
            print(str(e) + ': '+ str(comp.get_atomic_fraction(Element(e))))

    def show_answer(self, crystal_name):
        if crystal_name in list(self.db_df['crystal_name']):
            file = self.db_path
            bandgap = self.db_df[
                self.db_df['crystal_name'].isin([crystal_name])
            ]['indirect_bandgap (eV)'].item()
            loss = np.abs(bandgap - self.target_bandgap)

            self.n_experiment += 1
            self.bandgaps.append(bandgap)
            self.crystals.append(crystal_name)
            self.learning_curve.append(loss)
            print('The number of trials: '+str(self.n_experiment))
            print('The chosen material：'+str(crystal_name)+' and its bandgap is '+str(bandgap)+' [eV]')
            
            n_better = (self.losses < loss).sum()
            
            arr = np.array([self.crystals, self.bandgaps, self.learning_curve]).T
            pd.DataFrame(arr, columns=['crystal', 'bandgap', 'loss']).to_csv('trial_history.csv')
        else:
            print("None in the database. Check the material name you've typed.")

    def show_history(self):
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_axes((0.1, 0.4, 0.8, 0.5))

        tick_l = [i for i in range(1, self.n_experiment+1)]
        label_l = self.crystals

        ax.set(
            title='trial history',
            xlabel='experiments',
            ylabel='band gap [eV]',
            xticks=tick_l,
        )

        ax.set_xticklabels(label_l, rotation=90, ha='right')
        ax.plot(self.bandgaps, marker="D", markersize=12)
        plt.show()

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_axes((0.1, 0.4, 0.8, 0.5))
        ax.set(
            title='learning curve',
            xlabel='experiments',
            ylabel='Absolute Error',
            xticks=tick_l,
        )
        ax.set_xticklabels(label_l, rotation=90, ha='right')
        ax.plot(self.learning_curve, marker="D", markersize=12)
        plt.show()
        
        display(pd.DataFrame([self.crystals, self.bandgaps, self.learning_curve], index=['crystal_name', 'bandgap [eV]', 'losses']).T)