# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 21:01:49 2024

@author: Christos P. Feidakis
"""

""" Helper functions for parsing AHoJ DB """

import pandas as pd
import numpy as np
import sys
from scipy.stats import sem, t

from run_types import APO_LABELS, HOLO_LABELS   # the pooled label sets (AHOJ-001, AHOJ-020)



def unique_uniprots(series):
    uniprots = set(up for row in series for up in row.split(' ') if up)
    #uniprots = set(up for row in series for up in row.split('-') if up)
    return len(uniprots)


def unique_values(series):
    return series.nunique()


def unique_struct_chains(qstruct_series, chains3_series):
    struct_chains = set()
    for struct, chains in zip(qstruct_series, chains3_series):
        # Ensure both struct and chains are treated as strings
        struct_str = str(struct) if pd.notna(struct) else ''
        chains_str = str(chains) if pd.notna(chains) else ''

        for chain in chains_str.split('-'):
            struct_chains.add(struct_str + chain)
    return len(struct_chains)


def parse_query_summaries1(ddf):
    # Group by unique ligands
    grouped_ddf = ddf.groupby('qlig')

    ligand_info_in = grouped_ddf.apply(lambda x: pd.Series({
        'Ligand_ID': x['qlig'].iloc[0],
        'UniProt': unique_uniprots(x['qUNPs']),
        'structures': unique_values(x['qstruct']),
        'chains': unique_struct_chains(x['qstruct'], x['qchains3']),
        'sites': unique_values(x['ahoj_query'])
    }))

    # Extract paired entries only (num_apo_pockets >= 1)
    ddf_paired = ddf[ddf['num_apo_pockets'] >= 1]
    
    # Group by unique ligands for paired entries
    grouped_ddf_paired = ddf_paired.groupby('qlig')
    
    ligand_info_paired = grouped_ddf_paired.apply(lambda x: pd.Series({
        'Ligand_ID': x['qlig'].iloc[0],
        'UniProt': unique_uniprots(x['qUNPs']),
        'structures': unique_values(x['qstruct']),
        'chains': unique_struct_chains(x['qstruct'], x['qchains3']),
        'sites': unique_values(x['ahoj_query'])
    }))

    # Calculate percentages in new df and handle missing ligands
    ligand_info_ratio = ligand_info_in.copy()
    columns_to_calculate = ['UniProt', 'structures', 'chains', 'sites']
    for column in columns_to_calculate:
        percentage_column = '%' + column
        ligand_info_ratio[percentage_column] = (ligand_info_paired[column + '_pair'] / ligand_info_in[column] * 100).fillna(0)
        ligand_info_ratio[percentage_column] = ligand_info_ratio[percentage_column].round(decimals=0)

    # Rename columns in ligand_info_paired with '_pair' suffix
    paired_column_names = {col: col + '_pair' for col in ligand_info_paired.columns if col != 'Ligand_ID'}
    ligand_info_paired = ligand_info_paired.rename(columns=paired_column_names)

    # Merge 'ligand_info_paired' with 'ligand_info_ratio'
    ligand_info_ratio = pd.concat([ligand_info_ratio, ligand_info_paired.drop(columns='Ligand_ID')], axis=1)

    # Reorder columns in 'ligand_info_ratio'
    column_order = ['Ligand_ID'] + ligand_info_in.columns[1:].tolist() + ligand_info_paired.drop(columns='Ligand_ID').columns.tolist() + ['%' + column for column in columns_to_calculate]
    ligand_info_ratio = ligand_info_ratio[column_order]

    # Sort by #sites
    ligand_info_ratio = ligand_info_ratio.sort_values(by='sites', ascending=False)
    
    # Calculate ratio of paired queries out of all queries
    paired_query_ratio = round((len(ddf_paired) / len(ddf)) * 100, 2)

    return ligand_info_ratio


def parse_query_summaries(ddf): # Can be also used locally after run is complete
#def parse_query_summaries(query_summaries_df):
    # Make tables to show differences between Ligand types and apo holo ratios
    # Same with UniProts, show which proteins don't have apos

    # Create ligand representation table (how many sites, chains, structures, UNPs)
    # Ligand, #UniProt, #structures, #chains, #sites

    # Group by unique ligands and calculate the desired information for entire ddf (input)
    ligand_info_in = ddf.groupby('qlig').agg(
        Ligand_ID=('qlig', 'first'),
        # Count the number of unique values in the 'UNP' column for each unique ligand
        UniProt=('qUNPs', unique_uniprots),  # Use the custom function here
        #UniProt=('qUNPs', lambda x: x.nunique()),
        # Count the number of unique values in the 'qr_struct' column for each unique ligand
        structures=('qstruct', lambda x: x.nunique()),
        # Count the number of unique values obtained by joining the 'qr_struct' and 'qr_chain' columns
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf.loc[x.index, 'qchains3'])),
        #chains=('qstructchains3', lambda x: x.nunique()),
        # Count the number of unique values in the 'query' column for each unique ligand
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Extract paired entries only (num_apo_pockets >= 1)
    ddf_paired = ddf[ddf['num_apo_pockets'] >= 1]

    # Extract entries with unobserved pockets (fully or 50%)
    ddf_dis = ddf[(ddf['num_unobs100'] >= 1) | (ddf['num_unobs30'] >= 1)] # entries with 100% or 30% unobserved pockets (includes 50% unobserved)
    #ddf_dis = ddf[(ddf['num_unobs100'] >= 1) | (ddf['num_unobs50'] >= 1)] # entries with 100% or 50% unobserved pockets
    #ddf_dis = ddf[ddf['num_unobs100'] >= 1] # Only 100% unobserved pockets

    # Extract entries with unobserved pockets and no observed apo pockets (num_unobs100 > 0, num_apo_pockets = 0)
    #ddf_dis2 = ddf[((ddf['num_unobs100'] > 0) | (ddf['num_unobs50'] >= 1)) & (ddf['num_apo_pockets'] == 0)] # entries with 100% or 50% unobserved pockets
    #ddf_dis2 = ddf[(ddf['num_unobs100'] > 0) & (ddf['num_apo_pockets'] == 0)] # Only 100% unobserved pockets

    # Extract entries with unobserved pockets and specific conditions:
    # - For num_unobs100 > 0: num_apo_pockets == 0
    # - For num_unobs50 >= 1: num_apo_pockets > num_unobs50
    condition_100 = (ddf['num_unobs100'] > 0) & (ddf['num_apo_pockets'] == 0)
    condition_50 = (ddf['num_unobs50'] >= 1) & (ddf['num_apo_pockets'] <= ddf['num_unobs50'])
    condition_30 = (ddf['num_unobs30'] >= 1) & (ddf['num_apo_pockets'] <= ddf['num_unobs30'])

    ddf_dis2 = ddf[condition_100 | condition_30] # Should include 50% unobserved
    #ddf_dis2 = ddf[condition_100 | condition_50]

    # Do the same for the paired entries (at least one apo)
    ligand_info_paired = ddf_paired.groupby('qlig').agg(
        Ligand_ID=('qlig', 'first'),
        # Count the number of unique values in the 'UNP' column for each unique ligand
        UniProt=('qUNPs', unique_uniprots),  # Use the custom function here
        #UniProt=('qUNPs', lambda x: x.nunique()),
        # Count the number of unique values in the 'qr_struct' column for each unique ligand
        structures=('qstruct', lambda x: x.nunique()),
        # Count the number of unique values obtained by joining the 'qr_struct' and 'qr_chain' columns
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf.loc[x.index, 'qchains3'])),
        #chains=('qstructchains3', lambda x: x.nunique()),
        # Count the number of unique values in the 'query' column for each unique ligand
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Calculate percentages in new df and handle missing ligands
    ligand_info_ratio = ligand_info_in.copy()
    columns_to_calculate = ['UniProt', 'structures', 'chains', 'sites']
    for column in columns_to_calculate:
        percentage_column = '%' + column
        ligand_info_ratio[percentage_column] = (ligand_info_paired[column] / ligand_info_in[column] * 100).fillna(np.nan)
        ligand_info_ratio[percentage_column] = ligand_info_ratio[percentage_column].round(decimals=0)
        #ligand_info_ratio[percentage_column] = (ligand_info_paired[column] / ligand_info_in[column] * 100).replace(np.inf, np.nan)
        #ligand_info_ratio[percentage_column] = ligand_info_ratio[percentage_column].round(decimals=0).astype(str)

    # Rename columns in ligand_info_paired with '_pair' suffix
    paired_column_names = {col: col + '_pair' for col in ligand_info_paired.columns if col != 'Ligand_ID'}
    ligand_info_paired = ligand_info_paired.rename(columns=paired_column_names)

    # Merge 'ligand_info_paired' with 'ligand_info_ratio'
    ligand_info_ratio = pd.concat([ligand_info_ratio, ligand_info_paired.drop(columns='Ligand_ID')], axis=1)

    # Reorder columns in 'ligand_info_ratio'
    column_order = ['Ligand_ID'] + ligand_info_in.columns[1:].tolist() + ligand_info_paired.drop(columns='Ligand_ID').columns.tolist() + ['%' + column for column in columns_to_calculate]
    ligand_info_ratio = ligand_info_ratio[column_order]

    # Sort by #sites
    ligand_info_ratio = ligand_info_ratio.sort_values(by='sites', ascending=False)
    #ligand_info_in = ligand_info_in.sort_values(by='sites', ascending=False)
    #ligand_info_paired = ligand_info_paired.sort_values(by='sites', ascending=False)
    
    # Extract unique UniProts from both dataframes
    #unique_uniprots_in = set(up for row in ddf['qUNPs'] if isinstance(row, str) for up in row.split(' ') if up)
    #unique_uniprots_paired = set(up for row in ddf_paired['qUNPs'] if isinstance(row, str) for up in row.split(' ') if up)
    unique_uniprots_in = set(up for row in ddf['qUNPs'] for up in row.split(' ') if up) # changed from "-"
    unique_uniprots_paired = set(up for row in ddf_paired['qUNPs'] for up in row.split(' ') if up) # changed from "-"


    # Calculate ratio of paired queries out of all queries
    paired_query_ratio = round((len(ddf_paired) / len(ddf)) * 100, 2)

    return ligand_info_ratio, len(ddf), len(ddf_paired), paired_query_ratio, unique_uniprots_in, unique_uniprots_paired, ddf_dis, ddf_dis2


def parse_query_summaries3(ddf): # Can be also used locally after run is complete
    # Make tables to show differences between Ligand types and apo holo ratios
    # Same with UniProts, show which proteins don't have apos

    # Create ligand representation table (how many sites, chains, structures, UNPs)
    # Ligand, #UniProt, #structures, #chains, #sites

    # Group by unique ligands and calculate the desired information for entire ddf (input)
    ligand_info_in = ddf.groupby('qlig').agg(
        Ligand_ID=('qlig', 'first'),
        # Count the number of unique values in the 'UNP' column for each unique ligand
        UniProt=('qUNPs', unique_uniprots),  # Use the custom function here
        #UniProt=('qUNPs', lambda x: x.nunique()),
        # Count the number of unique values in the 'qr_struct' column for each unique ligand
        structures=('qstruct', lambda x: x.nunique()),
        # Count the number of unique values obtained by joining the 'qr_struct' and 'qr_chain' columns
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf.loc[x.index, 'qchains3'])),
        #chains=('qstructchains3', lambda x: x.nunique()),
        # Count the number of unique values in the 'query' column for each unique ligand
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Extract paired entries only (num_apo_pockets >= 1)
    ddf_paired = ddf[ddf['num_apo_pockets'] >= 1]

    # Do the same for the paired entries (at least one apo)
    ligand_info_paired = ddf_paired.groupby('qlig').agg(
        Ligand_ID=('qlig', 'first'),
        # Count the number of unique values in the 'UNP' column for each unique ligand
        UniProt=('qUNPs', unique_uniprots),  # Use the custom function here
        #UniProt=('qUNPs', lambda x: x.nunique()),
        # Count the number of unique values in the 'qr_struct' column for each unique ligand
        structures=('qstruct', lambda x: x.nunique()),
        # Count the number of unique values obtained by joining the 'qr_struct' and 'qr_chain' columns
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf.loc[x.index, 'qchains3'])),
        #chains=('qstructchains3', lambda x: x.nunique()),
        # Count the number of unique values in the 'query' column for each unique ligand
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Extract entries with unobserved pockets
    ddf_dis = ddf[ddf['num_unobs100'] >= 1]

    # Extract entries with unobserved pockets and no observed apo pockets (num_unobs100 > 0, num_apo_pockets = 0)
    ddf_dis2 = ddf[(ddf['num_unobs100'] > 0) & (ddf['num_apo_pockets'] == 0)]
    
    # Do the same for the unobserved entries (at least one unobserved)
    ligand_info_unobs = ddf_dis.groupby('qlig').agg(
        Ligand_ID=('qlig', 'first'),
        # Count the number of unique values in the 'UNP' column for each unique ligand
        UniProt=('qUNPs', unique_uniprots),  # Use the custom function here
        #UniProt=('qUNPs', lambda x: x.nunique()),
        # Count the number of unique values in the 'qr_struct' column for each unique ligand
        structures=('qstruct', lambda x: x.nunique()),
        # Count the number of unique values obtained by joining the 'qr_struct' and 'qr_chain' columns
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf.loc[x.index, 'qchains3'])),
        #chains=('qstructchains3', lambda x: x.nunique()),
        # Count the number of unique values in the 'query' column for each unique ligand
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Calculate percentages in new df and handle missing ligands
    ligand_info_ratio = ligand_info_in.copy()
    columns_to_calculate = ['UniProt', 'structures', 'chains', 'sites']
    for column in columns_to_calculate:
        percentage_column = '%' + column
        ligand_info_ratio[percentage_column] = (ligand_info_paired[column] / ligand_info_in[column] * 100).fillna(np.nan)
        ligand_info_ratio[percentage_column] = ligand_info_ratio[percentage_column].round(decimals=0)
        percentage_dis_col = '%' + column + '_dis'
        ligand_info_ratio[percentage_dis_col] = (ligand_info_unobs[column] / ligand_info_in[column] * 100).fillna(np.nan)
        ligand_info_ratio[percentage_dis_col] = ligand_info_ratio[percentage_dis_col].round(decimals=0)

    # Rename columns in ligand_info_paired with '_pair' suffix
    paired_column_names = {col: col + '_pair' for col in ligand_info_paired.columns if col != 'Ligand_ID'}
    ligand_info_paired = ligand_info_paired.rename(columns=paired_column_names)

    # Rename columns in ligand_info_unobs with '_dis' suffix
    paired_column_names = {col: col + '_dis' for col in ligand_info_unobs.columns if col != 'Ligand_ID'}
    ligand_info_unobs = ligand_info_unobs.rename(columns=paired_column_names)

    # Merge 'ligand_info_paired' with 'ligand_info_ratio'
    ligand_info_ratio = pd.concat([ligand_info_ratio, ligand_info_paired.drop(columns='Ligand_ID')], axis=1)

    # Merge 'ligand_info_unobs' with 'ligand_info_ratio'
    ligand_info_ratio = pd.concat([ligand_info_ratio, ligand_info_unobs.drop(columns='Ligand_ID')], axis=1)

    # Reorder columns in 'ligand_info_ratio'
    column_order = ['Ligand_ID'] + ligand_info_in.columns[1:].tolist() + ligand_info_paired.drop(columns='Ligand_ID').columns.tolist() + ['%' + column for column in columns_to_calculate]
    ligand_info_ratio = ligand_info_ratio[column_order]

    # Sort by #sites
    ligand_info_ratio = ligand_info_ratio.sort_values(by='sites', ascending=False)

    # Extract unique UniProts from both dataframes
    unique_uniprots_in = set(up for row in ddf['qUNPs'] for up in row.split(' ') if up)
    unique_uniprots_paired = set(up for row in ddf_paired['qUNPs'] for up in row.split(' ') if up)
    unique_uniprots_unobs = set(up for row in ddf_dis['qUNPs'] for up in row.split(' ') if up)

    # Calculate ratio of paired queries out of all queries
    paired_query_ratio = round((len(ddf_paired) / len(ddf)) * 100, 2)
    unobs_query_ratio = round((len(ddf_dis) / len(ddf)) * 100, 2)

    return ligand_info_ratio, len(ddf), len(ddf_paired), paired_query_ratio, unique_uniprots_in, unique_uniprots_paired, ddf_dis, ddf_dis2, ligand_info_unobs


def parse_query_summaries4(ddf):

    # Group by unique ligands and calculate the desired information for entire ddf (input)
    ligand_info_in = ddf.groupby('qlig').agg(
        Ligand_ID=('qlig', 'first'),
        UniProt=('qUNPs', unique_uniprots),  # Use the custom function here
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Extract paired entries only (num_apo_pockets >= 1)
    ddf_paired = ddf[ddf['num_apo_pockets'] >= 1]

    # Extract entries with unobserved pockets
    ddf_dis = ddf[(ddf['num_unobs100'] >= 1) | (ddf['num_unobs30'] >= 1)] # entries with 100% or 30% unobserved pockets
    #ddf_dis = ddf[(ddf['num_unobs100'] >= 1) | (ddf['num_unobs50'] >= 1)] # entries with 100% or 50% unobserved pockets
    #ddf_dis = ddf[ddf['num_unobs100'] >= 1] # Only 100% unobserved pockets

    # Extract entries with unobserved pockets and no observed apo pockets (num_unobs100 > 0, num_apo_pockets = 0)
    #ddf_dis2 = ddf[((ddf['num_unobs100'] > 0) | (ddf['num_unobs50'] >= 1)) & (ddf['num_apo_pockets'] == 0)] # entries with 100% or 50% unobserved pockets
    #ddf_dis2 = ddf[(ddf['num_unobs100'] > 0) & (ddf['num_apo_pockets'] == 0)] # Only 100% unobserved pockets

    # Extract entries with unobserved pockets and specific conditions:
    # - For num_unobs100 > 0: num_apo_pockets == 0
    # - For num_unobs50 >= 1: num_apo_pockets > num_unobs50
    condition_100 = (ddf['num_unobs100'] > 0) & (ddf['num_apo_pockets'] == 0)
    #condition_50 = (ddf['num_unobs50'] >= 1) & (ddf['num_apo_pockets'] <= ddf['num_unobs50'])
    condition_30 = (ddf['num_unobs30'] >= 1) & (ddf['num_apo_pockets'] <= ddf['num_unobs30'])

    ddf_dis2 = ddf[condition_100 | condition_30]
    #ddf_dis2 = ddf[condition_100 | condition_50]

    # Extract entries with unobserved pockets and no observed apo pockets (num_unobs100 > 0, num_apo_pockets == 0)
    #ddf_dis2 = ddf[(ddf['num_unobs100'] > 0) & (ddf['num_apo_pockets'] == 0)]

    # Do the same for the paired entries (at least one apo)
    ligand_info_paired = ddf_paired.groupby('qlig').agg(
        Ligand_ID=('qlig', 'first'),
        UniProt=('qUNPs', unique_uniprots),  # Use the custom function here
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Do the same for the entries with unobserved pockets
    ligand_info_unobs = ddf_dis.groupby('qlig').agg(
        Ligand_ID=('qlig', 'first'),
        UniProt=('qUNPs', unique_uniprots),  # Use the custom function here
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Calculate percentages in new df and handle missing ligands
    ligand_info_ratio = ligand_info_in.copy()
    columns_to_calculate = ['UniProt', 'structures', 'chains', 'sites']

    for column in columns_to_calculate:
        # Calculate percentage for paired entries
        percentage_column_paired = '%' + column + '_pair'
        ligand_info_ratio[percentage_column_paired] = (ligand_info_paired[column] / ligand_info_in[column] * 100).fillna(np.nan)
        ligand_info_ratio[percentage_column_paired] = ligand_info_ratio[percentage_column_paired].round(decimals=0)

        # Calculate percentage for unobserved pockets
        percentage_column_unobs = '%' + column + '_unobs'
        ligand_info_ratio[percentage_column_unobs] = (ligand_info_unobs[column] / ligand_info_in[column] * 100).fillna(np.nan)
        ligand_info_ratio[percentage_column_unobs] = ligand_info_ratio[percentage_column_unobs].round(decimals=0)

    # Rename columns in ligand_info_paired and ligand_info_unobs with '_pair' and '_unobs' suffixes respectively
    paired_column_names = {col: col + '_pair' for col in ligand_info_paired.columns if col != 'Ligand_ID'}
    ligand_info_paired = ligand_info_paired.rename(columns=paired_column_names)

    unobs_column_names = {col: col + '_unobs' for col in ligand_info_unobs.columns if col != 'Ligand_ID'}
    ligand_info_unobs = ligand_info_unobs.rename(columns=unobs_column_names)

    # Merge 'ligand_info_paired' and 'ligand_info_unobs' with 'ligand_info_ratio' by Ligand_ID
    ligand_info_ratio = pd.concat([ligand_info_ratio, ligand_info_paired.drop(columns='Ligand_ID'), ligand_info_unobs.drop(columns='Ligand_ID')], axis=1)

    # Reorder columns in 'ligand_info_ratio'
    column_order = ['Ligand_ID'] + ligand_info_in.columns[1:].tolist() + \
                   [col for col in ligand_info_paired.columns if col != 'Ligand_ID'] + \
                   ['%' + column + '_pair' for column in columns_to_calculate] + \
                   [col for col in ligand_info_unobs.columns if col != 'Ligand_ID'] + \
                   ['%' + column + '_unobs' for column in columns_to_calculate]
    ligand_info_ratio = ligand_info_ratio[column_order]

    # Sort by #sites
    ligand_info_ratio = ligand_info_ratio.sort_values(by='sites', ascending=False)

    # Extract unique UniProts from both dataframes
    unique_uniprots_in = set(up for row in ddf['qUNPs'] for up in row.split(' ') if up)
    unique_uniprots_paired = set(up for row in ddf_paired['qUNPs'] for up in row.split(' ') if up)

    # Calculate ratio of paired queries out of all queries
    paired_query_ratio = round((len(ddf_paired) / len(ddf)) * 100, 2)

    return ligand_info_ratio, len(ddf), len(ddf_paired), paired_query_ratio, unique_uniprots_in, unique_uniprots_paired, ddf_dis, ddf_dis2, ligand_info_unobs


def parse_query_summaries5(ddf): # Add cryptic entries, get all data from single input df

    # Extract entries based on selection criteria
    # APO paired entries only (observed)
    ddf_paired_A = ddf[ddf['num_apo_0unobs'] > 0]
    #ddf_paired_A = ddf[ddf['num_apo_pockets'] > 0]
    # HOLO paired entries (observed)
    ddf_paired_H = ddf[ddf['num_holo_0unobs'] > 0]
    #ddf_paired_H = ddf[ddf['num_holo_pockets'] > 0]

    # Disorder (out of SEQRES results) Note: seqres_ddf replaced with ddf
    # Extract entries with apo or 100unobs pockets (as reference to unobserved %)
    ddf_seqres_unobs_reference = ddf[(ddf['num_apo_pockets'] > 0) | (ddf['num_unobs100'] > 0)] # Paired or unobserved entries
    # Create categories of unobserved pockets based on specific conditions
    condition_100 = (ddf['num_unobs100'] > 0) & (ddf['num_apo_pockets'] == 0)
    #condition_50 = (seqres_ddf['num_unobs50'] >= 1) & (seqres_ddf['num_apo_pockets'] <= seqres_ddf['num_unobs50'])
    condition_30 = (ddf['num_unobs30'] > 0) & (ddf['num_apo_pockets'] <= ddf['num_unobs30'])
    condition_apo30percent_plus = ddf['ratio_apo_to_unobs30+'] >= 30 # entries where (num_unobs30 + num_unobs100) make up 30% or more of the total apo results (num_apo_pockets + num_unobs100)
    # Extract each category of unobserved pockets
    ddf_dis = ddf[(ddf['num_unobs100'] > 0) | (ddf['num_unobs30'] > 0)] # entries with at least one 100% or 30% unobserved pocket
    ddf_dis2 = ddf[condition_apo30percent_plus]
    ddf_dis3 = ddf[condition_100 | condition_30]

    # Crypticity (out of observed results)
    ddf_crypt_A = ddf[ddf['crypt_Ao'] > 0] # ddf_crypt_A = ddf[ddf['crypt_A'] > 0] # ddf[(ddf['crypt_A'] > 0) | (ddf['crypt_H'] > 0)]
    ddf_crypt_H = ddf[ddf['crypt_Ho'] > 0] # ddf_crypt_H = ddf[ddf['crypt_H'] > 0] # ddf[(ddf['crypt_A'] > 0) | (ddf['crypt_H'] > 0)]


    # Create groups for each extracted category (Group by unique ligands and calculate the desired information for entire data frame)

    # Input observed
    ligand_info_in = ddf.groupby('qlig').agg(
        Ligand_ID=('qlig', 'first'),
        UniProt=('qUNPs', unique_uniprots),  # Use the custom function here
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Apo-paired observed
    ligand_info_paired_A = ddf_paired_A.groupby('qlig').agg(
        Ligand_ID=('qlig', 'first'),
        UniProt=('qUNPs', unique_uniprots),  # Use the custom function here
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf_paired_A.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Holo-paired observed
    ligand_info_paired_H = ddf_paired_H.groupby('qlig').agg(
        Ligand_ID=('qlig', 'first'),
        UniProt=('qUNPs', unique_uniprots),  # Use the custom function here
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf_paired_H.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Apo-paired SEQRES
    ligand_info_seqres = ddf_seqres_unobs_reference.groupby('qlig').agg(
        Ligand_ID=('qlig', 'first'),
        UniProt=('qUNPs', unique_uniprots),  # Use the custom function here
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf_seqres_unobs_reference.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # dis1 SEQRES
    ligand_info_unobs = ddf_dis.groupby('qlig').agg( #
        Ligand_ID=('qlig', 'first'),
        UniProt=('qUNPs', unique_uniprots),  # Use the custom function here
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf_dis.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # dis2 SEQRES
    ligand_info_unobs2 = ddf_dis2.groupby('qlig').agg( #
        Ligand_ID=('qlig', 'first'),
        UniProt=('qUNPs', unique_uniprots),  # Use the custom function here
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf_dis2.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # dis3 SEQRES
    ligand_info_unobs3 = ddf_dis3.groupby('qlig').agg( #
        Ligand_ID=('qlig', 'first'),
        UniProt=('qUNPs', unique_uniprots),  # Use the custom function here
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf_dis3.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Cryptic Apo observed
    ligand_info_crypt_A = ddf_crypt_A.groupby('qlig').agg(
        Ligand_ID=('qlig', 'first'),
        UniProt=('qUNPs', unique_uniprots),  # Use the custom function here
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf_crypt_A.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Cryptic Holo observed
    ligand_info_crypt_H = ddf_crypt_H.groupby('qlig').agg(
        Ligand_ID=('qlig', 'first'),
        UniProt=('qUNPs', unique_uniprots),  # Use the custom function here
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf_crypt_H.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Calculate percentages in new df and handle missing ligands
    ligand_info_ratio = ligand_info_in.copy()
    columns_to_calculate = ['UniProt', 'structures', 'chains', 'sites']

    for column in columns_to_calculate:
        # Calculate percentage for paired entries (out of all input pockets)
        percentage_column_paired = '%' + column + '_pair'
        ligand_info_ratio[percentage_column_paired] = (ligand_info_paired_A[column] / ligand_info_in[column] * 100).fillna(np.nan)
        ligand_info_ratio[percentage_column_paired] = ligand_info_ratio[percentage_column_paired].round(decimals=0)

        # Calculate percentage for unobserved pockets (out of SEQRES results)
        percentage_column_unobs = '%' + column + '_unobs'
        ligand_info_ratio[percentage_column_unobs] = (ligand_info_unobs[column] / ligand_info_seqres[column] * 100).fillna(np.nan)
        ligand_info_ratio[percentage_column_unobs] = ligand_info_ratio[percentage_column_unobs].round(decimals=0)

        # Calculate percentage for unobserved pockets #2 (out of SEQRES results)
        percentage_column_unobs2 = '%' + column + '_unobs2'
        ligand_info_ratio[percentage_column_unobs2] = (ligand_info_unobs2[column] / ligand_info_seqres[column] * 100).fillna(np.nan)
        ligand_info_ratio[percentage_column_unobs2] = ligand_info_ratio[percentage_column_unobs2].round(decimals=0)

        # Calculate percentage for unobserved pockets #3 (out of SEQRES results)
        percentage_column_unobs3 = '%' + column + '_unobs3'
        ligand_info_ratio[percentage_column_unobs3] = (ligand_info_unobs3[column] / ligand_info_seqres[column] * 100).fillna(np.nan)
        ligand_info_ratio[percentage_column_unobs3] = ligand_info_ratio[percentage_column_unobs3].round(decimals=0)

        # Calculate percentage for Apo cryptic pockets (out of APO-paired pockets)
        percentage_column_crypt_A = '%' + column + '_cryptA'
        ligand_info_ratio[percentage_column_crypt_A] = (ligand_info_crypt_A[column] / ligand_info_paired_A[column] * 100).fillna(np.nan) # ligand_info_in
        ligand_info_ratio[percentage_column_crypt_A] = ligand_info_ratio[percentage_column_crypt_A].round(decimals=0)
        
        # Calculate percentage for Holo cryptic pockets (out of HOLO-paired pockets)
        percentage_column_crypt_H = '%' + column + '_cryptH'
        ligand_info_ratio[percentage_column_crypt_H] = (ligand_info_crypt_H[column] / ligand_info_paired_H[column] * 100).fillna(np.nan) # ligand_info_in
        ligand_info_ratio[percentage_column_crypt_H] = ligand_info_ratio[percentage_column_crypt_H].round(decimals=0)


    # Rename columns in ligand_info_paired and ligand_info_unobs with '_pair' and '_unobs' suffixes respectively
    paired_column_names = {col: col + '_pair' for col in ligand_info_paired_A.columns if col != 'Ligand_ID'}
    ligand_info_paired_A = ligand_info_paired_A.rename(columns=paired_column_names)

    unobs_column_names = {col: col + '_unobs' for col in ligand_info_unobs.columns if col != 'Ligand_ID'}
    ligand_info_unobs = ligand_info_unobs.rename(columns=unobs_column_names)

    unobs_column_names2 = {col: col + '_unobs2' for col in ligand_info_unobs2.columns if col != 'Ligand_ID'}
    ligand_info_unobs2 = ligand_info_unobs2.rename(columns=unobs_column_names2)

    unobs_column_names3 = {col: col + '_unobs3' for col in ligand_info_unobs3.columns if col != 'Ligand_ID'}
    ligand_info_unobs3 = ligand_info_unobs3.rename(columns=unobs_column_names3)

    cryptA_column_names = {col: col + '_cryptA' for col in ligand_info_crypt_A.columns if col != 'Ligand_ID'}
    ligand_info_crypt_A = ligand_info_crypt_A.rename(columns=cryptA_column_names)
    
    cryptH_column_names = {col: col + '_cryptH' for col in ligand_info_crypt_H.columns if col != 'Ligand_ID'}
    ligand_info_crypt_H = ligand_info_crypt_H.rename(columns=cryptH_column_names)

    # Merge 'ligand_info_paired', 'ligand_info_unobs', and 'ligand_info_crypt' with 'ligand_info_ratio' by Ligand_ID
    ligand_info_ratio = pd.concat([ligand_info_ratio, 
                                   ligand_info_paired_A.drop(columns='Ligand_ID'), 
                                   ligand_info_unobs.drop(columns='Ligand_ID'), 
                                   ligand_info_unobs2.drop(columns='Ligand_ID'), 
                                   ligand_info_unobs3.drop(columns='Ligand_ID'), 
                                   ligand_info_crypt_A.drop(columns='Ligand_ID'),
                                   ligand_info_crypt_H.drop(columns='Ligand_ID')], axis=1)

    # Reorder columns in 'ligand_info_ratio'
    column_order = ['Ligand_ID'] + ligand_info_in.columns[1:].tolist() + \
                   [col for col in ligand_info_paired_A.columns if col != 'Ligand_ID'] + \
                   ['%' + column + '_pair' for column in columns_to_calculate] + \
                   [col for col in ligand_info_unobs.columns if col != 'Ligand_ID'] + \
                   ['%' + column + '_unobs' for column in columns_to_calculate] + \
                   [col for col in ligand_info_unobs2.columns if col != 'Ligand_ID'] + \
                   ['%' + column + '_unobs2' for column in columns_to_calculate] + \
                   [col for col in ligand_info_unobs3.columns if col != 'Ligand_ID'] + \
                   ['%' + column + '_unobs3' for column in columns_to_calculate] + \
                   [col for col in ligand_info_crypt_A.columns if col != 'Ligand_ID'] + \
                   ['%' + column + '_cryptA' for column in columns_to_calculate] + \
                   [col for col in ligand_info_crypt_H.columns if col != 'Ligand_ID'] + \
                   ['%' + column + '_cryptH' for column in columns_to_calculate]

    ligand_info_ratio = ligand_info_ratio[column_order]

    # Sort by #sites
    ligand_info_ratio = ligand_info_ratio.sort_values(by='sites', ascending=False)

    # Extract unique UniProts from both dataframes
    unique_uniprots_in = set(up for row in ddf['qUNPs'] for up in row.split(' ') if up)
    unique_uniprots_paired = set(up for row in ddf_paired_A['qUNPs'] for up in row.split(' ') if up)

    # Calculate ratio of paired queries out of all queries
    paired_query_ratio = round((len(ddf_paired_A) / len(ddf)) * 100, 2)

    return ligand_info_ratio, len(ddf), len(ddf_paired_A), paired_query_ratio, unique_uniprots_in, unique_uniprots_paired, ddf_dis, ddf_dis2, ddf_dis3, ligand_info_unobs


def parse_query_summaries_by_uniprot(ddf): 
    # First, split the qUNPs field to extract individual UniProt IDs
    ddf = ddf.assign(qUNPs_split=ddf['qUNPs'].str.split())

    # Explode the qUNPs_split column so that each UniProt ID has its own row
    ddf = ddf.explode('qUNPs_split')

    # Now group by UniProt IDs (qUNPs_split) and calculate the desired information
    uniprot_info_in = ddf.groupby('qUNPs_split').agg(
        UniProt_ID=('qUNPs_split', 'first'),
        Ligand=('qlig', lambda x: ' '.join(x.dropna().astype(str).unique())),  # Convert to string and handle NaN
        unique_ligand_count=('qlig', lambda x: x.dropna().nunique()),  # Count unique ligands
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Extract paired entries only (num_apo_pockets >= 1)
    ddf_paired = ddf[ddf['num_apo_pockets'] >= 1]

    # Extract entries with cryptic pockets
    ddf_crypt = ddf[(ddf['crypt_A'] > 0) | (ddf['crypt_H'] > 0)]

    # Extract entries with unobserved pockets
    ddf_dis = ddf[(ddf['num_unobs100'] >= 1) | (ddf['num_unobs30'] >= 1)]

    condition_100 = (ddf['num_unobs100'] > 0) & (ddf['num_apo_pockets'] == 0)
    condition_30 = (ddf['num_unobs30'] >= 1) & (ddf['num_apo_pockets'] <= ddf['num_unobs30'])
    ddf_dis2 = ddf[condition_100 | condition_30]

    # Group the paired entries by UniProt ID
    uniprot_info_paired = ddf_paired.groupby('qUNPs_split').agg(
        UniProt_ID=('qUNPs_split', 'first'),
        Ligand=('qlig', lambda x: ' '.join(x.dropna().astype(str).unique())),  # Convert to string and handle NaN
        unique_ligand_count=('qlig', lambda x: x.dropna().nunique()),  # Count unique ligands
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Group the unobserved entries by UniProt ID
    uniprot_info_unobs = ddf_dis.groupby('qUNPs_split').agg(
        UniProt_ID=('qUNPs_split', 'first'),
        Ligand=('qlig', lambda x: ' '.join(x.dropna().astype(str).unique())),  # Convert to string and handle NaN
        unique_ligand_count=('qlig', lambda x: x.dropna().nunique()),  # Count unique ligands
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Group the cryptic entries by UniProt ID
    uniprot_info_crypt = ddf_crypt.groupby('qUNPs_split').agg(
        UniProt_ID=('qUNPs_split', 'first'),
        Ligand=('qlig', lambda x: ' '.join(x.dropna().astype(str).unique())),  # Convert to string and handle NaN
        unique_ligand_count=('qlig', lambda x: x.dropna().nunique()),  # Count unique ligands
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Calculate percentages in new df and handle missing UniProt IDs
    uniprot_info_ratio = uniprot_info_in.copy()
    columns_to_calculate = ['structures', 'chains', 'sites']

    for column in columns_to_calculate:
        # Calculate percentage for paired entries
        percentage_column_paired = '%' + column + '_pair'
        uniprot_info_ratio[percentage_column_paired] = (uniprot_info_paired[column] / uniprot_info_in[column] * 100).fillna(np.nan)
        uniprot_info_ratio[percentage_column_paired] = uniprot_info_ratio[percentage_column_paired].round(decimals=0)

        # Calculate percentage for unobserved pockets
        percentage_column_unobs = '%' + column + '_unobs'
        uniprot_info_ratio[percentage_column_unobs] = (uniprot_info_unobs[column] / uniprot_info_in[column] * 100).fillna(np.nan)
        uniprot_info_ratio[percentage_column_unobs] = uniprot_info_ratio[percentage_column_unobs].round(decimals=0)

        # Calculate percentage for cryptic pockets
        percentage_column_crypt = '%' + column + '_crypt'
        uniprot_info_ratio[percentage_column_crypt] = (uniprot_info_crypt[column] / uniprot_info_in[column] * 100).fillna(np.nan)
        uniprot_info_ratio[percentage_column_crypt] = uniprot_info_ratio[percentage_column_crypt].round(decimals=0)

    # Rename columns in uniprot_info_paired and uniprot_info_unobs with '_pair' and '_unobs' suffixes respectively
    paired_column_names = {col: col + '_pair' for col in uniprot_info_paired.columns if col != 'UniProt_ID'}
    uniprot_info_paired = uniprot_info_paired.rename(columns=paired_column_names)

    unobs_column_names = {col: col + '_unobs' for col in uniprot_info_unobs.columns if col != 'UniProt_ID'}
    uniprot_info_unobs = uniprot_info_unobs.rename(columns=unobs_column_names)

    crypt_column_names = {col: col + '_crypt' for col in uniprot_info_crypt.columns if col != 'UniProt_ID'}
    uniprot_info_crypt = uniprot_info_crypt.rename(columns=crypt_column_names)

    # Merge 'uniprot_info_paired', 'uniprot_info_unobs', and 'uniprot_info_crypt' with 'uniprot_info_ratio' by UniProt_ID
    uniprot_info_ratio = pd.concat([uniprot_info_ratio, 
                                    uniprot_info_paired.drop(columns='UniProt_ID'), 
                                    uniprot_info_unobs.drop(columns='UniProt_ID'), 
                                    uniprot_info_crypt.drop(columns='UniProt_ID')], axis=1)

    # Reorder columns
    column_order = ['UniProt_ID'] + uniprot_info_in.columns[1:].tolist() + \
                   [col for col in uniprot_info_paired.columns if col != 'UniProt_ID'] + \
                   ['%' + column + '_pair' for column in columns_to_calculate] + \
                   [col for col in uniprot_info_unobs.columns if col != 'UniProt_ID'] + \
                   ['%' + column + '_unobs' for column in columns_to_calculate] + \
                   [col for col in uniprot_info_crypt.columns if col != 'UniProt_ID'] + \
                   ['%' + column + '_crypt' for column in columns_to_calculate]
    uniprot_info_ratio = uniprot_info_ratio[column_order]

    # Sort by #sites
    uniprot_info_ratio = uniprot_info_ratio.sort_values(by='sites', ascending=False)

    return uniprot_info_ratio


def parse_query_summaries_by_uniprot5(ddf): # Add cryptic entries, get all data from single input df

    # Split qUNPs to handle multiple UniProt IDs and explode the dataframe
    ddf = ddf.assign(qUNPs_split=ddf['qUNPs'].str.split())
    ddf = ddf.explode('qUNPs_split')

    # Extract entries based on selection criteria
    # APO paired entries only (observed)
    ddf_paired_A = ddf[ddf['num_apo_0unobs'] > 0]
    # HOLO paired entries (observed)
    ddf_paired_H = ddf[ddf['num_holo_0unobs'] > 0]

    # Disorder (out of SEQRES results)
    ddf_seqres_unobs_reference = ddf[(ddf['num_apo_pockets'] > 0) | (ddf['num_unobs100'] > 0)]
    condition_100 = (ddf['num_unobs100'] > 0) & (ddf['num_apo_pockets'] == 0)
    condition_30 = (ddf['num_unobs30'] > 0) & (ddf['num_apo_pockets'] <= ddf['num_unobs30'])
    condition_apo30percent_plus = ddf['ratio_apo_to_unobs30+'] >= 30
    ddf_dis = ddf[(ddf['num_unobs100'] > 0) | (ddf['num_unobs30'] > 0)]
    ddf_dis2 = ddf[condition_apo30percent_plus]
    ddf_dis3 = ddf[condition_100 | condition_30]

    # Crypticity (out of observed results)
    ddf_crypt_A = ddf[ddf['crypt_Ao'] > 0]
    ddf_crypt_H = ddf[ddf['crypt_Ho'] > 0]

    # Create groups for each extracted category (Group by unique UniProt IDs)
    # Input observed
    uniprot_info_in = ddf.groupby('qUNPs_split').agg(
        UniProt_ID=('qUNPs_split', 'first'),
        Ligands=('qlig', lambda x: ' '.join(x.dropna().unique())),
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Apo-paired observed
    uniprot_info_paired_A = ddf_paired_A.groupby('qUNPs_split').agg(
        UniProt_ID=('qUNPs_split', 'first'),
        Ligands=('qlig', lambda x: ' '.join(x.dropna().unique())),
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf_paired_A.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Holo-paired observed
    uniprot_info_paired_H = ddf_paired_H.groupby('qUNPs_split').agg(
        UniProt_ID=('qUNPs_split', 'first'),
        Ligands=('qlig', lambda x: ' '.join(x.dropna().unique())),
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf_paired_H.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Apo-paired SEQRES
    uniprot_info_seqres = ddf_seqres_unobs_reference.groupby('qUNPs_split').agg(
        UniProt_ID=('qUNPs_split', 'first'),
        Ligands=('qlig', lambda x: ' '.join(x.dropna().unique())),
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf_seqres_unobs_reference.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # dis1 SEQRES
    uniprot_info_unobs = ddf_dis.groupby('qUNPs_split').agg(
        UniProt_ID=('qUNPs_split', 'first'),
        Ligands=('qlig', lambda x: ' '.join(x.dropna().unique())),
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf_dis.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # dis2 SEQRES
    uniprot_info_unobs2 = ddf_dis2.groupby('qUNPs_split').agg(
        UniProt_ID=('qUNPs_split', 'first'),
        Ligands=('qlig', lambda x: ' '.join(x.dropna().unique())),
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf_dis2.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # dis3 SEQRES
    uniprot_info_unobs3 = ddf_dis3.groupby('qUNPs_split').agg(
        UniProt_ID=('qUNPs_split', 'first'),
        Ligands=('qlig', lambda x: ' '.join(x.dropna().unique())),
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf_dis3.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Cryptic Apo observed
    uniprot_info_crypt_A = ddf_crypt_A.groupby('qUNPs_split').agg(
        UniProt_ID=('qUNPs_split', 'first'),
        Ligands=('qlig', lambda x: ' '.join(x.dropna().unique())),
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf_crypt_A.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Cryptic Holo observed
    uniprot_info_crypt_H = ddf_crypt_H.groupby('qUNPs_split').agg(
        UniProt_ID=('qUNPs_split', 'first'),
        Ligands=('qlig', lambda x: ' '.join(x.dropna().unique())),
        structures=('qstruct', lambda x: x.nunique()),
        chains=('qstruct', lambda x: unique_struct_chains(x, ddf_crypt_H.loc[x.index, 'qchains3'])),
        sites=('ahoj_query', lambda x: x.nunique())
    )

    # Calculate percentages in new df and handle missing UniProt IDs
    uniprot_info_ratio = uniprot_info_in.copy()
    columns_to_calculate = ['structures', 'chains', 'sites']

    for column in columns_to_calculate:
        # Calculate percentage for paired entries (out of all input pockets)
        percentage_column_paired = '%' + column + '_pair'
        uniprot_info_ratio[percentage_column_paired] = (uniprot_info_paired_A[column] / uniprot_info_in[column] * 100).fillna(np.nan)
        uniprot_info_ratio[percentage_column_paired] = uniprot_info_ratio[percentage_column_paired].round(decimals=0)

        # Calculate percentage for unobserved pockets (out of SEQRES results)
        percentage_column_unobs = '%' + column + '_unobs'
        uniprot_info_ratio[percentage_column_unobs] = (uniprot_info_unobs[column] / uniprot_info_seqres[column] * 100).fillna(np.nan)
        uniprot_info_ratio[percentage_column_unobs] = uniprot_info_ratio[percentage_column_unobs].round(decimals=0)

        # Calculate percentage for unobserved pockets #2 (out of SEQRES results)
        percentage_column_unobs2 = '%' + column + '_unobs2'
        uniprot_info_ratio[percentage_column_unobs2] = (uniprot_info_unobs2[column] / uniprot_info_seqres[column] * 100).fillna(np.nan)
        uniprot_info_ratio[percentage_column_unobs2] = uniprot_info_ratio[percentage_column_unobs2].round(decimals=0)

        # Calculate percentage for unobserved pockets #3 (out of SEQRES results)
        percentage_column_unobs3 = '%' + column + '_unobs3'
        uniprot_info_ratio[percentage_column_unobs3] = (uniprot_info_unobs3[column] / uniprot_info_seqres[column] * 100).fillna(np.nan)
        uniprot_info_ratio[percentage_column_unobs3] = uniprot_info_ratio[percentage_column_unobs3].round(decimals=0)

        percentage_column_crypt_A = '%' + column + '_cryptA'
        uniprot_info_ratio[percentage_column_crypt_A] = (uniprot_info_crypt_A[column] / uniprot_info_paired_A[column] * 100).fillna(np.nan)
        uniprot_info_ratio[percentage_column_crypt_A] = uniprot_info_ratio[percentage_column_crypt_A].round(decimals=0)

        percentage_column_crypt_H = '%' + column + '_cryptH'
        uniprot_info_ratio[percentage_column_crypt_H] = (uniprot_info_crypt_H[column] / uniprot_info_paired_H[column] * 100).fillna(np.nan)
        uniprot_info_ratio[percentage_column_crypt_H] = uniprot_info_ratio[percentage_column_crypt_H].round(decimals=0)

    # Rename columns and prepare for merging
    paired_column_names = {col: col + '_pair' for col in uniprot_info_paired_A.columns if col != 'UniProt_ID'}
    uniprot_info_paired_A = uniprot_info_paired_A.rename(columns=paired_column_names)

    unobs_column_names = {col: col + '_unobs' for col in uniprot_info_unobs.columns if col != 'UniProt_ID'}
    uniprot_info_unobs = uniprot_info_unobs.rename(columns=unobs_column_names)

    unobs_column_names2 = {col: col + '_unobs2' for col in uniprot_info_unobs2.columns if col != 'UniProt_ID'}
    uniprot_info_unobs2 = uniprot_info_unobs2.rename(columns=unobs_column_names2)

    unobs_column_names3 = {col: col + '_unobs3' for col in uniprot_info_unobs3.columns if col != 'UniProt_ID'}
    uniprot_info_unobs3 = uniprot_info_unobs3.rename(columns=unobs_column_names3)

    cryptA_column_names = {col: col + '_cryptA' for col in uniprot_info_crypt_A.columns if col != 'UniProt_ID'}
    uniprot_info_crypt_A = uniprot_info_crypt_A.rename(columns=cryptA_column_names)

    cryptH_column_names = {col: col + '_cryptH' for col in uniprot_info_crypt_H.columns if col != 'UniProt_ID'}
    uniprot_info_crypt_H = uniprot_info_crypt_H.rename(columns=cryptH_column_names)

    # Merge all info with 'uniprot_info_ratio'
    uniprot_info_ratio = pd.concat([uniprot_info_ratio,
                                    uniprot_info_paired_A.drop(columns='UniProt_ID'),
                                    uniprot_info_unobs.drop(columns='UniProt_ID'),
                                    uniprot_info_unobs2.drop(columns='UniProt_ID'),
                                    uniprot_info_unobs3.drop(columns='UniProt_ID'),
                                    uniprot_info_crypt_A.drop(columns='UniProt_ID'),
                                    uniprot_info_crypt_H.drop(columns='UniProt_ID')], axis=1)

    # Reorder columns
    column_order = ['UniProt_ID'] + uniprot_info_in.columns[1:].tolist() + \
                   [col for col in uniprot_info_paired_A.columns if col != 'UniProt_ID'] + \
                   ['%' + column + '_pair' for column in columns_to_calculate] + \
                   [col for col in uniprot_info_unobs.columns if col != 'UniProt_ID'] + \
                   ['%' + column + '_unobs' for column in columns_to_calculate] + \
                   [col for col in uniprot_info_unobs2.columns if col != 'UniProt_ID'] + \
                   ['%' + column + '_unobs2' for column in columns_to_calculate] + \
                   [col for col in uniprot_info_unobs3.columns if col != 'UniProt_ID'] + \
                   ['%' + column + '_unobs3' for column in columns_to_calculate] + \
                   [col for col in uniprot_info_crypt_A.columns if col != 'UniProt_ID'] + \
                   ['%' + column + '_cryptA' for column in columns_to_calculate] + \
                   [col for col in uniprot_info_crypt_H.columns if col != 'UniProt_ID'] + \
                   ['%' + column + '_cryptH' for column in columns_to_calculate]

    uniprot_info_ratio = uniprot_info_ratio[column_order]

    # Sort by number of sites
    uniprot_info_ratio = uniprot_info_ratio.sort_values(by='sites', ascending=False)

    # Extract unique UniProts from both dataframes
    unique_uniprots_in = set(up for row in ddf['qUNPs'] for up in row.split(' ') if up)
    unique_uniprots_paired = set(up for row in ddf_paired_A['qUNPs'] for up in row.split(' ') if up)

    # Calculate ratio of paired queries out of all queries
    paired_query_ratio = round((len(ddf_paired_A) / len(ddf)) * 100, 2)

    return uniprot_info_ratio, len(ddf), len(ddf_paired_A), paired_query_ratio, unique_uniprots_in, unique_uniprots_paired, ddf_dis, ddf_dis2, ddf_dis3, uniprot_info_unobs


def calculate_averages_from_summaries(apo_summaries, holo_summaries, paired=None):

    # Separate the job_id column if it exists
    apo_job_id = apo_summaries['job_id'] if 'job_id' in apo_summaries.columns else None
    holo_job_id = holo_summaries['job_id'] if 'job_id' in holo_summaries.columns else None

    # Filter based on paired argument if provided
    if paired is not None:
        if 'paired_entry' in apo_summaries.columns:
            apo_summaries = apo_summaries[apo_summaries['paired_entry'] == paired]
        if 'paired_entry' in holo_summaries.columns:
            holo_summaries = holo_summaries[holo_summaries['paired_entry'] == paired]

    # Drop the 'paired_entry' column if it exists
    if 'paired_entry' in apo_summaries.columns:
        apo_summaries = apo_summaries.drop(columns=['paired_entry'])
    if 'paired_entry' in holo_summaries.columns:
        holo_summaries = holo_summaries.drop(columns=['paired_entry'])

    # Drop job_id column from summaries before converting to numeric
    if apo_job_id is not None:
        apo_summaries = apo_summaries.drop(columns=['job_id'])
    if holo_job_id is not None:
        holo_summaries = holo_summaries.drop(columns=['job_id'])

    # Calculate the average of each metric for apo_summaries and holo_summaries (naive approach)
    #apo_averages = apo_summaries.mean(skipna=True)
    #holo_averages = holo_summaries.mean(skipna=True)

    #print('\nApo summaries')
    #print(apo_summaries.to_string(index=False))
    #print(apo_summaries.T.to_string(index=True))
    #print('\nHolo summaries')
    #print(holo_summaries.to_string(index=False))
    #print(holo_summaries.T.to_string(index=True))

    # Convert all columns in apo_summaries to numeric, coercing errors to NaN
    apo_summaries_numeric = apo_summaries.apply(pd.to_numeric, errors='coerce')
    # Calculate the mean for numeric columns in apo_summaries
    apo_averages = apo_summaries_numeric.mean(skipna=True)
    # Do the same for holo_summaries
    holo_summaries_numeric = holo_summaries.apply(pd.to_numeric, errors='coerce')
    holo_averages = holo_summaries_numeric.mean(skipna=True)
    # Add totals for pockets to apo and holo dfs
    apo_pockets_total = apo_summaries['pockets'].sum()
    apo_averages['pockets_sum'] = apo_pockets_total
    holo_pockets_total = holo_summaries['pockets'].sum()
    holo_averages['pockets_sum'] = holo_pockets_total


    # Convert Series to DataFrame
    apo_averages_df = apo_averages.to_frame(name='Average').reset_index()
    holo_averages_df = holo_averages.to_frame(name='Average').reset_index()

    # Optionally, rename the 'index' column to something more descriptive like 'Metric'
    apo_averages_df.rename(columns={'index': 'Metric'}, inplace=True)
    holo_averages_df.rename(columns={'index': 'Metric'}, inplace=True)

    #print('\nGrand average apo')
    #print(apo_averages_df.to_string(index=True))
    #print('\nGrand average holo')
    #print(holo_averages_df.to_string(index=True))

    # Merge the two average DataFrames
    combined_averages_df = pd.merge(apo_averages_df, holo_averages_df, on='Metric', suffixes=('_apo', '_holo'))

    # Rename the columns for better clarity
    combined_averages_df.rename(columns={'Average_apo': 'Apo Average', 'Average_holo': 'Holo Average'}, inplace=True)

    #print('Combined apo and holo averages')
    #print(combined_averages_df.to_string(index=True))

    # Add % Difference calculations
    combined_averages_df['% Difference'] = abs((combined_averages_df['Apo Average'] - combined_averages_df['Holo Average']) /
                                           ((combined_averages_df['Apo Average'] + combined_averages_df['Holo Average']) / 2)) * 100
    combined_averages_df['% Difference from Apo'] = ((combined_averages_df['Holo Average'] - combined_averages_df['Apo Average']) /
                                                     combined_averages_df['Apo Average']) * 100
    combined_averages_df['% Difference from Holo'] = ((combined_averages_df['Apo Average'] - combined_averages_df['Holo Average']) /
                                                     combined_averages_df['Holo Average']) * 100
    combined_averages_df['Max % Difference'] = combined_averages_df[['% Difference from Apo', '% Difference from Holo']].abs().max(axis=1)

    # Round the values in the DataFrame
    combined_averages_df = combined_averages_df.round(2)

    # Set 'Metric' as the index
    combined_averages_df.set_index('Metric', inplace=True)

    # Sort the DataFrame by index (metric names)
    combined_averages_df.sort_index(inplace=True)
    # Sorting the columns of the DataFrame alphabetically
    #sorted_combined_averages_df = combined_averages_df.sort_index(axis=1)

    # Round apo and holo summaries before returning
    # Convert columns to numeric, coercing errors to NaN
    #apo_summaries_numeric = apo_summaries.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    #apo_summaries_rounded = apo_summaries_numeric.round(2)
    #holo_summaries_numeric = holo_summaries.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    #holo_summaries_rounded = holo_summaries_numeric.round(2)

    # Reattach the job_id column after rounding, if it was present
    #if apo_job_id is not None:        apo_summaries_rounded['job_id'] = apo_job_id
    #if holo_job_id is not None:        holo_summaries_rounded['job_id'] = holo_job_id

    return combined_averages_df#, apo_summaries_rounded, holo_summaries_rounded


class DualOutput:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        self.file.close()


def safe_mean(series):
    """Calculate the mean and count non-NaN values, handling errors and non-numeric values."""
    non_null_series = pd.to_numeric(series, errors='coerce').dropna()
    count = non_null_series.count()
    if count == 0:
        return np.nan
    return non_null_series.mean()


def safe_weighted_mean(series):
    """Calculate the mean and count non-NaN values, handling errors and non-numeric values."""
    non_null_series = pd.to_numeric(series, errors='coerce').dropna()
    count = non_null_series.count()
    if count == 0:
        return np.nan, 0
    return non_null_series.mean(), count


def safe_min_max(series):
    # Filter out non-numeric values and NaNs
    numeric_series = pd.to_numeric(series, errors='coerce').dropna()

    # Count the number of valid (non-NaN) values
    valid_count = numeric_series.count()

    # If the series is empty after filtering, return NaN for both min and max
    if valid_count == 0:
        return (np.nan, np.nan), 0

    # Otherwise, return the min and max values along with the count
    return (numeric_series.min(), numeric_series.max()), valid_count


def unique_count_split(series, sep='-'):
    """Count unique elements in a Series after splitting by a separator."""
    return series.str.split(sep).explode().nunique()


def safe_mean0(series):
    """Calculate the mean, handling errors and non-numeric values."""
    return pd.to_numeric(series, errors='coerce').mean()


def create_summary_df(query_df, subset_df, class_label, paired_entry=None, job_id=None, round_df=1):

    if subset_df.empty:
        return pd.DataFrame()

    # Initialize dictionaries for metric names, values, and weights
    metric_names = {'Metric': 'Metric Names'}
    metric_values = {'Metric': 'Values'}
    metric_weights = {'Metric': 'Weights'}

    # Add number of entries (pockets) as a special case
    metric_names['pockets'] = 'pockets'
    metric_values['pockets'] = len(subset_df)
    metric_weights['pockets'] = 'N/A'  # No weight for 'pockets'

    # Add class label as a special case
    metric_names['class'] = 'Class'
    metric_values['class'] = class_label
    metric_weights['class'] = 'N/A'  # No weight for 'pockets'

    # Add paired_entry if provided
    if paired_entry is not None:
        metric_names['paired_entry'] = 'paired_entry'
        metric_values['paired_entry'] = paired_entry
        metric_weights['paired_entry'] = 'N/A'

    # Add job_id if provided
    if job_id is not None:
        metric_names['job_id'] = 'job_id'
        metric_values['job_id'] = job_id
        metric_weights['job_id'] = 'N/A'

    # Define metrics to calculate
    avg_metrics = ['#resis5', '#resis3', 'unobserved_rsd', '%mapped_sqr', '%unobserved_rsd', 'RoG', 'bfac_t', 
                   'bfac_avg', 'sasa', 'sasa_avg', 'asa', 'asa_avg', 'volume', '#solvent', 'resolution', 'tm_score', 
                   'tm_score_i', 'rmsd', 'aln_len', 'pocket_rms', 'pocket_len', 'pocket_dist']
    min_max_metrics = ['#resis5', '#resis3', 'unobserved_rsd', '%mapped_sqr', '%unobserved_rsd', 'RoG', 'bfac_t', 
                       'bfac_avg', 'sasa', 'sasa_avg', 'asa', 'asa_avg', 'volume', 
                       'resolution', 'tm_score', 'tm_score_i', 'rmsd', 'aln_len', 'pocket_rms', 'pocket_len', 'pocket_dist']
    count_unique_metrics = ['UNPs', 'query_ligs', 'nonquery_ligs', 'pocket_ptm', 'pocket_nuc', 'allostery_lig', 'allostery_ptm', 'allostery_nuc']
    delta_metrics = ['sasa', 'sasa_avg', 'asa', 'asa_avg', 'volume', 'bfac_avg']

    try:
        # Calculate averages and weights
        for metric in avg_metrics:
            mean_val, count_val = safe_weighted_mean(subset_df[metric])
            metric_names[metric + '_avg'] = metric + '_avg'
            metric_values[metric + '_avg'] = mean_val
            metric_weights[metric + '_avg'] = count_val

        # Calculate min, max, and weights
        for metric in min_max_metrics:
            (min_val, max_val), count_val = safe_min_max(subset_df[metric])
            # Add min, max, and weight for each metric
            metric_names[metric + '_min'] = metric + '_min'
            metric_values[metric + '_min'] = min_val
            metric_weights[metric + '_min'] = count_val

            metric_names[metric + '_max'] = metric + '_max'
            metric_values[metric + '_max'] = max_val
            metric_weights[metric + '_max'] = count_val

        # Handle count unique metrics
        for metric in count_unique_metrics:
            unq_count_val = unique_count_split(subset_df[metric].astype(str))
            metric_names[metric + '_count'] = metric + '_count'
            metric_values[metric + '_count'] = unq_count_val
            metric_weights[metric + '_count'] = metric_values[metric + '_count'] #count_val

        # Calculate delta for each metric
        for metric in delta_metrics:
            if metric in query_df.columns and metric in subset_df.columns:

                # Calculate raw delta for each row (result - query)
                delta_values = subset_df[metric] - query_df[metric].values[0]

                # Calculate the mean absolute delta (abs before average)
                mean_delta = delta_values.abs().mean()

                # Add to the summary data
                metric_names[metric + '_delta'] = metric + '_delta'
                metric_values[metric + '_delta'] = mean_delta
                metric_weights[metric + '_delta'] = len(delta_values)  # Use count as weight

        # Handle special metrics
        metric_names['unique_pdbs'] = 'unique_pdbs'
        metric_values['unique_pdbs'] = len(subset_df['structure'].astype(str).unique())
        metric_weights['unique_pdbs'] = 'N/A'  # No weight  metric_values['unique_pdbs']

        metric_names['exp_method_combined'] = 'exp_method_combined'
        metric_values['exp_method_combined'] = '-'.join(subset_df['exp_method'].astype(str).str[:5].unique())
        metric_weights['exp_method_combined'] = 'N/A'  # No weight  metric_values['unique_pdbs']

        metric_names['unique_chains3'] = 'unique_chains3'
        metric_values['unique_chains3'] = len((subset_df['structure'].astype(str) + subset_df['chains3'].astype(str)).unique())
        metric_weights['unique_chains3'] = 'N/A'  # No weight

        #unpseq_overlap
        metric_names['unpseq_overlap'] = 'unpseq_overlap'
        metric_values['unpseq_overlap'] = np.nan
        metric_weights['unpseq_overlap'] = 'N/A'  # No weight

    except Exception as e:
        print(f"Error in summarizing data: {e}")

    # Create a summary DataFrame
    summary_df = pd.DataFrame([metric_names, metric_values, metric_weights])

    # Round all numerical columns to two decimal places
    if round_df == 1:
        # Since metric_values is the second row, select it for rounding (using .iloc[1] to get the second row)
        summary_df.iloc[1] = summary_df.iloc[1].apply(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)
        #numeric_cols = summary_df.select_dtypes(include=['number']).columns
        #summary_df[numeric_cols] = summary_df[numeric_cols].round(2)

        # 2nd approach: Convert columns to numeric, coercing errors to NaN (this will replace non-numeric values/columns with NaN)
        #summary_df_numeric = summary_df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        #summary_df_rounded = summary_df_numeric.round(2)

    #print(summary_df.to_string())
    return summary_df


def sort_dataframe_v2c(df, sorting=0, full_pockets_aligned=None, bndgrsds_prcnt_seqres=None, bndgrsds_prcnt_obs=None,
                       resolution_soft=None, rog_diff_threshold_deltaQ=None, rog_diff_threshold_mean=None,
                       rog_sorting=None, query_rog=None, pocket_distance=None, pocket_len_resis3=None,
                       drop_no_resolution=None):
    """
    Filter a dataframe like global_results.csv, based on qualitative arguments.
    Filtering operations are restricted to candidate rows (role = 'C').

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    sorting : TYPE, optional
        DESCRIPTION. The default is 0.
    full_pockets_aligned : TYPE, optional
        DESCRIPTION. The default is None.
    bndgrsds_prcnt_seqres : TYPE, optional
        DESCRIPTION. The default is None.
    bndgrsds_prcnt_obs : TYPE, optional
        DESCRIPTION. The default is None.
    resolution_soft : TYPE, optional
        DESCRIPTION. The default is None.
    rog_diff_threshold_deltaQ : TYPE, optional
        DESCRIPTION. The default is None.
    rog_diff_threshold_mean : TYPE, optional
        DESCRIPTION. The default is None.
    rog_sorting : TYPE, optional
        DESCRIPTION. The default is None.
    query_rog : TYPE, optional
        DESCRIPTION. The default is None.
    pocket_distance : TYPE, optional
        DESCRIPTION. The default is None.
    pocket_len_resis3 : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    # Return an empty dataframe if the input is empty
    if df.empty or (df.replace('', np.nan).isnull().all(axis=None)): return df.iloc[0:0]
    # Replace empty strings with NaN, drop rows that are all NaN
    #df_check = df.replace('', np.nan)
    #df_nonempty = df_check.dropna(how='all')
    #if df_nonempty.empty: return df.iloc[0:0]
    #if df.empty: return df

    # Convert columns to numeric, if applicable
    # AHOJ-006: '%mapped_obs' and 'pocket_dist' are filtered below but were previously not coerced
    #   (only '%mapped_sqr' was). They are float by design, but a stray non-numeric token would
    #   make the column object dtype and raise on the '>=' / '>' comparison -> coerce defensively.
    for col in ['#resis3', 'pocket_len', '%mapped_sqr', '%mapped_obs', 'resolution', 'RoG', 'pocket_dist']:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

    # Apply full_pockets_aligned filtering
    if full_pockets_aligned == 1:
        # Find the #resis3 value for the query entry (role 'Q')
        if 'Q' in df['role'].values:
            query_resis3 = df.loc[df['role'] == 'Q', '#resis3'].values[0]

            # Filter out candidate entries (role 'C') if their pocket_len doesn't match the query's #resis3 value
            df = df[(df['role'] != 'C') | (df['pocket_len'] == query_resis3)]
            #df = df[(df['role'] != 'C') | ((df['pocket_len'] == query_resis3) & (df['pocket_len'] == df['#resis3']))]

    # Added: filter out when pocket_len and #resis3 of the candidate are not the same ()
    if pocket_len_resis3 == 1:
        df = df[(df['role'] != 'C') | (df['pocket_len'] == df['#resis3'])]

    # Hardcoded filter for %mapped_sqr to reject rows over 100.0
    # AHOJ-002: restrict to candidate rows (role 'C') so query/model reference rows are never dropped.
    df = df.drop(df[(df['role'] == 'C') & (df['%mapped_sqr'] > 100.0)].index)
    #df.drop(df[df['%mapped_sqr'] > 100.0].index, inplace=True)   # df = df[df['%mapped_sqr'] <= 100.0] # SettingWithCopyWarning

    # Discard pockets that are too far from query pocket after global superposition
    # AHOJ-002: restrict to candidate rows (role 'C') so query/model reference rows are never dropped.
    # AHOJ-003: pocket_dist is measured AFTER global chain superposition, so it conflates global
    #   conformational change (low tm_score) with pocket dissimilarity and over-filtered locally-valid
    #   pockets. Default raised to 20 A after a full-DB scan; may later be removed or made
    #   tm_score-conditional. See KNOWN_ISSUES.md (AHOJ-003) and analysis/AHOJ-003_pocket_dist_scan/.
    if pocket_distance:
        df = df.drop(df[(df['role'] == 'C') & (df['pocket_dist'] > pocket_distance)].index)

    ## Handle RoG filtering
    if rog_diff_threshold_deltaQ or rog_sorting:

        # Get query rog if it's not defined
        if not query_rog:
            # Retrieve the RoG of the query entry (role 'Q')
            if 'Q' in df['role'].values:
                query_rog = df.loc[df['role'] == 'Q', 'RoG'].values[0]    # Apply RoG difference threshold from query RoG if specified (safer -only removes candidates)
            #else:
            #    print('Could not get query RoG for filtering')

        # Filter candidates by their RoG difference between the query
        if rog_diff_threshold_deltaQ:

            # Calculate lower and upper bounds based on rog_diff_threshold_deltaQ
            lower_bound = query_rog * (1 - rog_diff_threshold_deltaQ)
            upper_bound = query_rog * (1 + rog_diff_threshold_deltaQ)
            # Filter out candidate entries (role 'C') with RoG outside the tolerance range
            df = df[(df['role'] != 'C') | ((df['RoG'] >= lower_bound) & (df['RoG'] <= upper_bound))]

        # Calculate RoG "distance" if rog_sorting & query_rog is provided and round to two decimal places
        if rog_sorting:
            df.loc[:, 'RoG_distance'] = abs(df['RoG'] - query_rog).round(2)

    # Filter candidates by their RoG difference between the mean RoG of all entries
    if rog_diff_threshold_mean is not None:

        mean_rog = df['RoG'].mean()
        # Apply the filter only to candidate entries (role = 'C')
        df = df[(df['role'] != 'C') | (abs(df['RoG'] - mean_rog) <= mean_rog * rog_diff_threshold_mean)]
        #df = df[abs(df['RoG'] - mean_rog) <= mean_rog * rog_diff_threshold_mean]

    # Apply other thresholds for filtering
    # AHOJ-002: restrict to candidate rows (role 'C') so query/model reference rows are never dropped.
    if bndgrsds_prcnt_seqres is not None and '%mapped_sqr' in df.columns:
        df = df[(df['role'] != 'C') | (df['%mapped_sqr'] >= bndgrsds_prcnt_seqres)]
    if bndgrsds_prcnt_obs is not None and '%mapped_obs' in df.columns:
        df = df[(df['role'] != 'C') | (df['%mapped_obs'] >= bndgrsds_prcnt_obs)]

    # Resolution handling (candidate rows only).
    # AHOJ-005: the previous one-liner `df[df['resolution'].notna() & (df['resolution'] <= soft)]`
    #   was NOT role-guarded (it could drop Q/M reference rows) and conflated two concerns:
    #   (a) "resolution above cutoff" and (b) "resolution missing". Missing/NMR resolution is
    #   stored as '-' -> coerced to NaN above (confirmed: NOT 0/-1), so it is detectable as NaN.
    #   The two filters are now independent and restricted to role == 'C'.
    if drop_no_resolution and 'resolution' in df.columns:
        # Drop candidates with no resolution (e.g. SOLUTION_NMR). Known to be desirable: NMR
        #   query pockets selected across conformers can inflate residue counts downstream.
        df = df[(df['role'] != 'C') | df['resolution'].notna()]
    if resolution_soft is not None and 'resolution' in df.columns:
        # Soft max-resolution cutoff on candidates. Candidates with NaN resolution are KEPT here
        #   (use drop_no_resolution to remove them) so the two filters stay orthogonal.
        df = df[(df['role'] != 'C') | df['resolution'].isna() | (df['resolution'] <= resolution_soft)]

    # Check if DataFrame is empty after filtering
    if df.empty:    return df

    # Optionally sort the DataFrame
    if sorting == 1:
        try:
            # Define columns to sort and their order
            columns_to_sort = ['%mapped_sqr', '#resis3']
            if rog_sorting is not None:
                columns_to_sort.append('RoG_distance')  # Add RoG_distance as an additional criterion
            columns_to_sort.append('resolution')

            sort_order = [False, False] + ([True] if rog_sorting is not None else []) + [True]

            sorted_df = df.sort_values(by=columns_to_sort, ascending=sort_order, na_position='last')
            print(f"After sorting: {len(sorted_df)} rows in sorted DataFrame.")
        except Exception as e:
            print(f"Error during sorting dataframe: {e}")
            return df

    return df


def split_dataframe(df):
    # Filter out only the candidates (rows where 'role' is 'C')
    candidates_df = df[df['role'] == 'C']

    # Create apo_df with rows where 'apoholo_assignment' is 'A', 'A-', 'A*' or 'A**'
    # AHOJ-001: 'A**' (holo->apo single-atom-ligand relabel) counts as apo. See KNOWN_ISSUES.md.
    # AHOJ-020: so does AHoJ's 'A-' (a gate-demoted query-ligand hit).
    apo_df = candidates_df[candidates_df['apoholo_assignment'].isin(APO_LABELS)]

    # Create holo_df with rows where 'apoholo_assignment' is 'H' or 'H*'
    holo_df = candidates_df[candidates_df['apoholo_assignment'].isin(HOLO_LABELS)]

    # Create disorder df with rows where 'apoholo_assignment' is 'U'
    dis_df = candidates_df[candidates_df['apoholo_assignment'].isin(['U'])]

    # Create model df with rows where 'role' is 'M'
    #model_df = df[df['role'] == 'M']

    return apo_df, holo_df, dis_df#, model_df


    """
    Mean and Standard Deviation: Calculated for the column to understand the central tendency and dispersion.
    Confidence Interval: Uses a t-distribution to construct a 95% confidence interval around the mean.
    Interquartile Range (IQR): Calculates the spread of the middle 50% of values, giving insight into the variability without the influence of outliers.
    Coefficient of Variation: Shows the standard deviation relative to the mean, which is useful for comparing variability across columns of different scales.
    """
    """
    Confidence Intervals (high/low): Construct confidence intervals around the mean of pocket_rms for each job. 
        A 95% confidence interval would give you a range where you can be reasonably sure the true mean 
        pocket_rms lies, helping assess the consistency of pocket_rms within each job.

    Interquartile Range (IQR): Calculate the 25th percentile (Q1) and 75th percentile (Q3) of 
        pocket_rms values to measure the spread between the "middle" 50% of values. This method can 
        be particularly useful if your pocket_rms data is skewed, as IQR is less sensitive to outliers.

    Variation Coefficient: Divide the standard deviation by the mean of pocket_rms for each 
        job to get a coefficient of variation (CV). This would indicate how much pocket_rms varies 
        relative to its average value, providing a sense of variability independent of the magnitude 
        of the mean.
    """

def compute_metrics(df, df_q, deltas=0, columns=['pocket_rms', 'volume', 'sasa', 'RoG', 'tm_score', 'tm_score_i', 'rmsd'], alpha=0.05,
                    anchor_query=True, keep_stats=None):
    """
    CHANGELOG
    add deltas options
    AHOJ-007: add `anchor_query`. When True (legacy default) a synthetic "query" row is injected
      into the candidate population before computing the descriptive stats, anchoring *_min at the
      theoretical 0 for deltas/pocket_rms (and at the query value for tm_score/rmsd). This is a
      deliberate "query as population member" choice, but it biases mean/std/CV/flexibility by an
      amount that DEPENDS ON candidate count (large for sparse entries, negligible for dense ones).
      Call with anchor_query=False to get clean candidate-only stats (no synthetic point).
      See KNOWN_ISSUES.md.
    AHOJ-015: add `keep_stats`. None (default) emits all 11 per-metric stats (backward compatible:
      the legacy full-analytics parser relies on this). Pass a subset (e.g. ['min','max','mean',
      'std_dev','iqr']) to emit only those columns and keep the per-entry KDE summary lean — the
      full 11-stat suite was over-wide for the classification summary. CV is always computed
      internally (the flexibility score needs it) but only emitted when requested. local/global
      flexibility are always emitted.
    """

    columns = list(columns)
    keep = None if keep_stats is None else set(keep_stats)
    metrics_data = {}
    flexibility_score = {'local_flexibility': 0, 'global_flexibility': 0}
    local_metrics = ['pocket_rms', 'sasa', 'volume', 'RoG']
    global_metrics = ['tm_score', 'tm_score_i', 'rmsd']
    delta_metrics = ['sasa', 'volume', 'RoG']
    local_cv_sum, global_cv_sum = 0, 0
    valid_local_metrics, valid_global_metrics = 0, 0

    # Compute deltas if specified
    if deltas > 0:
        for col in delta_metrics:
            if col in df.columns and col in df_q.columns:

                query_value = df_q[col].values[0]  # Assuming df_q contains a single query row
                df = df.copy()
                df[f'{col}_delta'] = abs(df[col] - query_value)

        # Adjust columns list based on deltas option
        if deltas == 1:  # Additionally process deltas
            columns += [f'{col}_delta' for col in delta_metrics]
        elif deltas == 2:  # Exclusively replace specified columns with their deltas
            columns = [f'{col}_delta' if col in delta_metrics else col for col in columns]


    # Prepare synthetic datapoints (AHOJ-007: only when anchor_query, see docstring)
    if anchor_query:
        synthetic_rows = {}
        for col in columns:
            if col in df.columns:
                synthetic_value = 0 if col.endswith('_delta') or col == 'pocket_rms' else df_q[col].values[0]
                synthetic_rows[col] = [synthetic_value]

        # Append synthetic rows to the DataFrame in one operation
        synthetic_df = pd.DataFrame(synthetic_rows)
        df = pd.concat([df, synthetic_df], ignore_index=True)


    for col in columns:
        # Filter out NaN values for accurate calculations
        col_data = df[col].dropna()

        # Calculate metrics only if there is data
        if not col_data.empty:
            min_val = round(col_data.min(), 2)
            max_val = round(col_data.max(), 2)
            mean_val = round(col_data.mean(), 3)
            std_dev = round(col_data.std(), 3)
            n = len(col_data)

            confidence_interval = None
            if n > 1:
                confidence_margin = sem(col_data) * t.ppf((1 + (1 - alpha)) / 2., n - 1)
                confidence_interval = (round(mean_val - confidence_margin, 3), round(mean_val + confidence_margin, 3))

            q1_val = round(col_data.quantile(0.25), 3)
            q3_val = round(col_data.quantile(0.75), 3)
            iqr_val = round(q3_val - q1_val, 3)
            range_val = round(max_val - min_val, 3)
            cv_val = round(std_dev / mean_val, 3) if mean_val != 0 else np.nan

            # Store metrics in a flat dictionary, with column-specific prefixes. Only the stats in
            # `keep` are emitted (keep=None -> all). CV is computed above regardless (flexibility
            # needs it) but emitted only if requested. AHOJ-015.
            _stat_values = {
                'min': min_val, 'max': max_val, 'range': range_val, 'mean': mean_val,
                'std_dev': std_dev, 'q1': q1_val, 'q3': q3_val, 'iqr': iqr_val,
                'confidence_interval_low': confidence_interval[0] if confidence_interval else np.nan,
                'confidence_interval_high': confidence_interval[1] if confidence_interval else np.nan,
                'coefficient_of_variation': cv_val,
            }
            for _stat, _val in _stat_values.items():
                if keep is None or _stat in keep:
                    metrics_data[f'{col}_{_stat}'] = _val

            # Aggregate valid coefficients of variation for flexibility scores
            base_col = col.removesuffix('_delta')
            if not np.isnan(cv_val):
                if base_col in local_metrics:
                    local_cv_sum += cv_val
                    valid_local_metrics += 1
                elif base_col in global_metrics:
                    global_cv_sum += cv_val
                    valid_global_metrics += 1

    # Calculate flexibility scores, averaging over available metrics
    flexibility_score['local_flexibility'] = round(local_cv_sum / valid_local_metrics, 3) if valid_local_metrics > 0 else np.nan
    flexibility_score['global_flexibility'] = round(global_cv_sum / valid_global_metrics, 3) if valid_global_metrics > 0 else np.nan

    # Combine metric data and flexibility scores into a single DataFrame row, placing flexibility scores first
    metrics_data = {**flexibility_score, **metrics_data}

    return pd.DataFrame([metrics_data])


def plot_flexibility(df, save_dir=None):
    """ Given a data frame with AHoJ results, e.g., apo subset, holo subset or all entry results, plot selected metrics and optionally save the plot """
    


def compute_metric0s(df, columns=['pocket_rms', 'volume', 'sasa', 'RoG', 'tm_score', 'tm_score_i', 'rmsd'], alpha=0.05):
    metrics_data = {}
    flexibility_score = {'local_flexibility': 0, 'global_flexibility': 0}
    local_metrics = ['pocket_rms', 'sasa', 'volume', 'RoG']
    global_metrics = ['tm_score', 'tm_score_i', 'rmsd']
    local_cv_sum, global_cv_sum = 0, 0

    for col in columns:
        # Filter out NaN values for accurate calculations
        col_data = df[col].dropna()

        # Calculate metrics
        mean_val = round(col_data.mean(), 3)
        std_dev = round(col_data.std(), 3)
        n = len(col_data)

        confidence_interval = None
        if n > 1:
            confidence_margin = sem(col_data) * t.ppf((1 + (1 - alpha)) / 2., n - 1)
            confidence_interval = (round(mean_val - confidence_margin, 3), round(mean_val + confidence_margin, 3))

        iqr_val = round(col_data.quantile(0.75) - col_data.quantile(0.25), 3)
        cv_val = round(std_dev / mean_val, 3) if mean_val != 0 else np.nan

        # Store metrics in a flat dictionary, with column-specific prefixes
        metrics_data[f'{col}_mean'] = mean_val
        metrics_data[f'{col}_std_dev'] = std_dev
        metrics_data[f'{col}_confidence_interval_low'] = confidence_interval[0] if confidence_interval else np.nan
        metrics_data[f'{col}_confidence_interval_high'] = confidence_interval[1] if confidence_interval else np.nan
        metrics_data[f'{col}_iqr'] = iqr_val
        metrics_data[f'{col}_coefficient_of_variation'] = cv_val

        # Aggregate coefficient of variation for flexibility scores
        if col in local_metrics and cv_val is not np.nan:
            local_cv_sum += cv_val
        elif col in global_metrics and cv_val is not np.nan:
            global_cv_sum += cv_val

    # Calculate flexibility scores as average of CVs across local and global metrics
    flexibility_score['local_flexibility'] = round(local_cv_sum / len(local_metrics), 3) if local_metrics else np.nan
    flexibility_score['global_flexibility'] = round(global_cv_sum / len(global_metrics), 3) if global_metrics else np.nan

    # Combine metric data and flexibility scores into a single DataFrame row, place flexibility scores at the beginning if computed
    metrics_data = {**flexibility_score, **metrics_data}

    return pd.DataFrame([metrics_data])


def classify_model_wE(model, apo_stats, holo_stats, metrics):
    # Extract mean and std_dev for each metric from apo and holo statistics
    apo_means = apo_stats[[f"{metric}_mean" for metric in metrics]].values.flatten()
    apo_stds = apo_stats[[f"{metric}_std_dev" for metric in metrics]].values.flatten()
    holo_means = holo_stats[[f"{metric}_mean" for metric in metrics]].values.flatten()
    holo_stds = holo_stats[[f"{metric}_std_dev" for metric in metrics]].values.flatten()
    
    # Extract model values
    model_values = model[metrics].values.flatten()

    # Calculate weighted Euclidean distances, weighing each metric by the variability
    apo_distance = np.sqrt(np.sum(((model_values - apo_means) / apo_stds) ** 2))
    holo_distance = np.sqrt(np.sum(((model_values - holo_means) / holo_stds) ** 2))

    # Classify based on the smaller distance
    classification = "apo" if apo_distance < holo_distance else "holo"

    # Confidence based on distance comparison
    confidence = 1 - (min(apo_distance, holo_distance) / (apo_distance + holo_distance))
    return classification, round(confidence, 3)


def classify_model_M(model, apo_stats, holo_stats, apo_df, holo_df, metrics):
    """ Use mahalanobis distance instead of Euclidean"""

    from scipy.spatial.distance import mahalanobis

    # Extract mean values for each metric
    apo_means = apo_stats[[f"{metric}_mean" for metric in metrics]].values.flatten()
    holo_means = holo_stats[[f"{metric}_mean" for metric in metrics]].values.flatten()

    # Extract model values
    model_values = model[metrics].values.flatten()

    # Calculate the covariance matrix for each group
    apo_cov_matrix = np.cov(apo_df[metrics].dropna().values.T)
    holo_cov_matrix = np.cov(holo_df[metrics].dropna().values.T)

    # Invert the covariance matrices
    apo_cov_inv = np.linalg.inv(apo_cov_matrix)
    holo_cov_inv = np.linalg.inv(holo_cov_matrix)

    # Calculate Mahalanobis distances
    apo_distance = mahalanobis(model_values, apo_means, apo_cov_inv)
    holo_distance = mahalanobis(model_values, holo_means, holo_cov_inv)

    # Classify based on the smaller Mahalanobis distance
    classification = "apo" if apo_distance < holo_distance else "holo"

    # Confidence based on distance comparison
    confidence = 1 - (min(apo_distance, holo_distance) / (apo_distance + holo_distance))
    return classification, round(confidence, 3)


# Hard-coded list of excluded single-atom ligands (heavy atoms or total atoms)
EXCLUDED_LIGANDS_TOTAL_LE1 = [
'0BE','3CO','3NI','4MO','4PU','4TI','6MO','AG','AL','AM','AR','ARS','AU','AU3','BA','BR','BRO','BS3','CA','CD','CE','CF','CL','CLO','CO','CR','CS',
'CU','CU1','CU3','D8U','DUM','DY','ER3','EU','EU3','F','FE','FE2','FLO','GA','GD','GD3','H','HG','HO','HO3','IDO','IN','IOD','IR','IR3','K','KR',
'LA','LI','LU','MG','MN','MN3','MO','NA','ND','NGN','NI','O','OS','OS4','OX','OXO','PB','PD','PR','PT','PT4','QTR','RB','RE','RH','RH3','RHF',
'RU','S','SB','SE','SM','SR','TA0','TB','TE','TH','TL','U1','UNX','V','W','XE','Y1','YB','YB2','YT3','ZCM','ZN','ZN2','ZR','ZTM']

EXCLUDED_LIGANDS_HEAVY_LE1 = [
'0BE','3CO','3NI','4MO','4PU','4TI','6MO','74C','AG','AL','AM','AR','ARS','AU','AU3','BA','BR','BRO','BS3','CA','CD','CE','CF','CH2','CH3','CL','CLO',
'CO','CR','CS','CU','CU1','CU3','D3O','D8U','DIS','DOD','DUM','DY','ER3','EU','EU3','F','FE','FE2','FLO','GA','GD','GD3','GTE','H','H2S','HG','HO',
'HO3','HOH','HYD','IDO','IN','IOD','IR','IR3','K','KR','LA','LI','LU','MG','MN','MN3','MO','MTO','NA','ND','ND4','NGN','NH','NH2','NH3','NH4','NI',
'O','OH','OS','OS4','OX','OXO','PB','PD','PR','PT','PT4','QTR','RB','RE','RH','RH3','RHF','RU','S','SB','SE','SM','SR','TA0','TB','TE','TH',
'TL','U1','UNX','V','W','XE','Y1','YB','YB2','YT3','ZCM','ZN','ZN2','ZR','ZTM']

base_ion_set = {"ZN","MG","MN","CA","FE","CU","CO","NI","CD","NA","K","CL","BR","I","SR","RB","BA","AG","LI","AL","PT","AU","PB","GD","EU","HO","SM","CS"}
single_atom_ligands = [
'AG','AL','AR','AS','AU','B','BA','BE','BR','CA','CD','CE','CL','CO','CR','CS','CU','EU','F','FE','GA','GD','HE','HG','I','IN','IR','K','KR','LA',
'LI','LU','MG','MN','MO','NA','ND','NE','NI','OS','PB','PD','PM','PR','PT','RB','RE','RH','RN','RU','SB','SC','SE','SM','SN','SR','TB','TC','TE','TH',
'TI','TL','V','W','XE','Y','YB','ZN','ZR']


import os
import re
from typing import Iterable, Optional, Set, Tuple

def relabel_apoholo_by_ions(
    df: pd.DataFrame,
    excluded_ligands: Optional[Iterable[str]] = None,   # <- external list (optional)
    *,
    # choose which built-in list to use when no external list is given
    builtin: str = "heavy_le1",                         # one of: "heavy_le1", "total_le1", "both", "none"
    # how to combine built-in with external list
    combine: str = "auto",                              # "auto" (external if provided else built-in),
                                                        # "union", "builtin_only", "external_only", "intersection"
    # optional file with ligands (one ID per line); merged with/exclusive per `combine`
    excluded_ligands_file: Optional[str] = None,

    label_col: str = "apoholo_assignment",
    role_col: str = "role",
    query_ligs_col: str = "query_ligs",
    nonquery_ligs_col: str = "nonquery_ligs",

    overwrite: bool = True,
    new_col: str = "apoholo_assignment_relabeled",
    save_dir: Optional[str] = None,
    save_name: str = "global_results_relabeled.csv",
    verbose: bool = True,

    # behavior flags
    query_ion_disables_filter: bool = True,            # if any Q has excluded ligand, disable filtering globally
    protect_labels: Set[str] = frozenset({"U"}),       # labels to preserve unchanged
    candidate_role: str = "C",
    demote_only: bool = False,                         # RUN_TYPES holo-2.4: only H/H* -> A** (never promote)
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Re-label apo/holo assignment using an exclusion list of ligands (ions, etc.).

    Built-in vs external lists
    --------------------------
    - Provide your own `excluded_ligands` (iterable) and/or `excluded_ligands_file` (one ID per line).
    - `builtin` selects which built-in set to use when no external is given:
        "heavy_le1": EXCLUDED_LIGANDS_HEAVY_LE1
        "total_le1": EXCLUDED_LIGANDS_TOTAL_LE1
        "both": union of the two
        "none": no built-in
    - `combine` controls how to merge built-in and external:
        "auto": external if provided else built-in
        "union": union of both
        "builtin_only": use only built-in
        "external_only": use only external
        "intersection": intersection of built-in and external

    Protections
    -----------
    - Do NOT modify rows where role != candidate_role (default 'C').
    - Do NOT modify rows whose current label is in `protect_labels` (default {'U'}).
    - If `query_ion_disables_filter` and ANY query row (role=='Q') has a query
      ligand in the final exclusion set, no filtering is applied anywhere.
    - `demote_only` (docs/RUN_TYPES.md, holo-2.4): AHoJ 2.4 labels already carry the holo-overlap
      gate, and gate-rejected entities stay listed in nonquery_ligs. Rebuilding H/H*/A from the ligand
      columns would therefore undo the gate. With demote_only=True a candidate may only move from
      H/H* to A** (an ion-only pocket, the v2c demotion); A/A*/A** rows are never promoted.

    Relabeling (for eligible candidate rows)
    ----------------------------------------
      H   : if query_ligs has ≥1 ligand
      H*  : else if (filtered) nonquery_ligs has ≥1 ligand
      A   : else
    """
    # ---- choose built-in ----
    builtin = (builtin or "heavy_le1").lower()
    if builtin == "heavy_le1":
        builtin_set = set(EXCLUDED_LIGANDS_HEAVY_LE1)
    elif builtin == "total_le1":
        builtin_set = set(EXCLUDED_LIGANDS_TOTAL_LE1)
    elif builtin == "both":
        builtin_set = set(EXCLUDED_LIGANDS_HEAVY_LE1) | set(EXCLUDED_LIGANDS_TOTAL_LE1)
    elif builtin == "none":
        builtin_set = set()
    else:
        raise ValueError("builtin must be one of: 'heavy_le1','total_le1','both','none'")

    # ---- load external (iterable and/or file) ----
    external_set: Set[str] = set()
    if excluded_ligands is not None:
        external_set |= {str(x).strip().upper() for x in excluded_ligands if str(x).strip()}
    if excluded_ligands_file:
        if not os.path.isfile(excluded_ligands_file):
            raise FileNotFoundError(f"excluded_ligands_file not found: {excluded_ligands_file}")
        with open(excluded_ligands_file, "r", encoding="utf-8") as fh:
            for line in fh:
                tok = line.strip()
                if tok and tok not in {".","?","NaN"}:
                    external_set.add(tok.upper())

    # ---- combine strategy ----
    combine = (combine or "auto").lower()
    if combine == "auto":
        final_excluded = external_set if external_set else builtin_set
    elif combine == "union":
        final_excluded = builtin_set | external_set
    elif combine == "builtin_only":
        final_excluded = builtin_set
    elif combine == "external_only":
        final_excluded = external_set
    elif combine == "intersection":
        final_excluded = builtin_set & external_set
    else:
        raise ValueError("combine must be one of: 'auto','union','builtin_only','external_only','intersection'")

    # ---- basic checks ----
    required = {label_col, role_col, query_ligs_col, nonquery_ligs_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    #old_labels = out[label_col].astype(str)
    target_col = label_col if overwrite else new_col
    if not overwrite and target_col in out.columns and target_col != label_col:
        out = out.drop(columns=[target_col])

    # Keep original dtype and NaNs
    old_labels = out[label_col].copy()
    new_labels = out[label_col].copy()

    # helper to parse ligands
    _split_re = re.compile(r"[-\s;,]+")
    def _parse_ligs(val) -> list[str]:
        if pd.isna(val):
            return []
        s = str(val).strip().upper()
        if s in {"", "NAN", "NONE", "."}:
            return []
        return [p for p in _split_re.split(s) if p]

    # global disable if any Q has excluded ligand in query_ligs (unchanged)
    disable_filter = False
    if query_ion_disables_filter:
        qrows = out[out[role_col] == "Q"]
        if not qrows.empty:
            q_has_excl = qrows[query_ligs_col].apply(lambda v: any(l in final_excluded for l in _parse_ligs(v)))
            disable_filter = bool(q_has_excl.any())

    # relabel candidates ONLY
    # AHOJ-012: iterate positionally so the label written via .iat aligns with the row even if
    #   `out` ever carries a non-default index (iterrows yields the index *label*, .iat needs a
    #   *position*). Safe today on the fresh RangeIndex; this makes it robust to sliced input.
    for pos, (_idx, row) in enumerate(out.iterrows()):
        role = str(row.get(role_col, ""))
        raw_lbl = row.get(label_col)
        cur_label = (str(raw_lbl).upper() if pd.notna(raw_lbl) else "")  # normalize for logic only

        # protections
        if role != candidate_role:           # e.g., Q, M -> untouched
            continue
        if cur_label in protect_labels:      # e.g., 'U' -> untouched
            continue

        q_ligs = _parse_ligs(row.get(query_ligs_col, ""))
        n_ligs = _parse_ligs(row.get(nonquery_ligs_col, ""))

        if not disable_filter and final_excluded:
            n_ligs = [x for x in n_ligs if x not in final_excluded]

        if demote_only:
            # keep AHoJ's own label unless a holo call rests on excluded ligands only
            if cur_label in {"H", "H*"} and not q_ligs and not n_ligs:
                new_labels.iat[pos] = "A**"
            continue

        if q_ligs:
            new_labels.iat[pos] = "H"
        elif n_ligs:
            new_labels.iat[pos] = "H*"
        else:
            # If you added the A** marking, keep it here:
            new_labels.iat[pos] = "A**" if cur_label in {"H","H*"} else "A"

    # write target column
    out[target_col] = new_labels

    # NA-aware change mask (so NaN vs NaN is NOT counted as a change)
    _na = object()  # unique sentinel
    changed_mask = ((out[role_col] == candidate_role) &~(old_labels.fillna(_na) == out[target_col].fillna(_na)))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, save_name)
        out.to_csv(out_path, index=False)
        if verbose:
            print(f"[relabel_apoholo_by_ions] saved: {out_path}")

    if verbose:
        print(f"[relabel_apoholo_by_ions] rows changed: {int(changed_mask.sum())}/{len(out)} "
              f"(target_col='{target_col}', disable_filter={disable_filter}, demote_only={demote_only}, "
              f"builtin='{builtin}', combine='{combine}', final_excluded={len(final_excluded)})")

    return out, changed_mask

