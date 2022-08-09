#!/usr/bin/env python3

from types import SimpleNamespace

def whichPC(username):
    if username == 'matt office':
        account = 'joliens'
        seedhemi = 'left'
    elif username == 'jolien office':
        account = 'jschuurmans'
        seedhemi = 'right'
    elif username == 'pony':
        account = 'joliens'
        seedhemi = 'both'
    return account, seedhemi


def path_names(username):
    account, seedhemi = whichPC(username)
    base_path = f'/home/{account}/Documents/02_recurrentSF_3T/data-bids/derivatives/'
    stat_path = f'{base_path}firstlevel-ffa/'
    ppi_path = f'{base_path}ppi/'
    plot_path = f'{ppi_path}plots/'
    name_addon = ''

    # get a dictionary of local variable names and values
    loc = locals()

    # make them less clunky to access
    paths = SimpleNamespace(**loc)

    return paths


def sub_cond_roi_names():
    subject_names = ['01','02','03','04','05','06','07','08',
                     '09','10','11','12','13','14','15','18']

    condition_names = {
        '1': 'pos 50 HSF',
        '3': 'pos 83 HSF',
        '5': 'pos 100 HSF',
        '7': 'pos 150 HSF',
        '2': 'pos 50 LSF',
        '4': 'pos 83 LSF',
        '6': 'pos 100 LSF',
        '8': 'pos 150 LSF',
        '9': 'neg 50 HSF',
        '11': 'neg 83 HSF',
        '13': 'neg 100 HSF',
        '15': 'neg 150 HSF',
        '10': 'neg 50 LSF',
        '12': 'neg 83 LSF',
        '14': 'neg 100 LSF',
        '16': 'neg 150 LSF',
    }

    roi_names={'14':'InferiorTemporalGyrusAnterior',
               '15':'InferiorTemporalGyrusPosterior',
               '16':'InferiorTemporalGyrusTemporooccipital',
               '23':'LateralOccipitalCortexInferior',
               '37':'TemporalFusiformCortexAnterior',
               '38':'TemporalFusiformCortexPosterior',
               '39':'TemporalOccipitalFusiformCortex',
               '40':'OccipitalFusiformGyrus'}

    return subject_names, condition_names, roi_names
