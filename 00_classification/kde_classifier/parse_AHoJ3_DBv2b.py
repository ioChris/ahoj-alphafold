# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 15:28:34 2024

@author: Christos P. Feidakis
"""
""" v3b
    - add columns in query_summaries.csv for number of % observed apo pockets (>= 100%, 70%)
    - add ahoj query column in apo_summaries.csv and holo_summaries.csv
    - add optional hard filter for percentage of observed residues (out of total mapped) 
      for global results df (enforced through sort_dataframe_v2 function)
    v3c
    - add number of cryptic pockets according to argument (separately for apo and holo dfs)
    - add AHoJ-DB version being parsed as a manual argument
    - update save_job_categories to also save reason in Discarded jobs
    - change 'full_pockets_observed' argument to 'full_pockets_aligned' and integrate it 
      into sort_dataframe_v2b to check for fully aligned pockets instead (it was obsolete)
    v3c2
    - add ratio_apo_to_unobs30_plus (ratio_apo_to_unobs30+) to represent entries where (num_unobs30 + num_unobs100) make up 30% or more of the total apo results (num_apo_pockets + num_unobs100)
    v3c3
    - add deltas for select metrics (volumetric) into the create_summary_df function (get delta for each result row, then apply absolute, mean and take different between query)
    - detach calculate_averages_from_summaries from main multi-processing loop
    - create 2 different summaries for paired and unpaired entries so that there is no need to re-run parsing in 2 modes
    - remove union argument
    - make apo holo summaries optional through new argument (skip create_summary_df for apo and holo dataframes)
    v3c4
    - get cryptic full pockets in addition to default filtering cryptic pockets
    - add columns in query summaries (i.e., rows with 0 unobserved residues: num_apo_0unobs, num_holo_0unobs, num_holo_obs100, crypt_A_obs, crypt_H_obs)
    v3c5
    - start adding descriptive statistics to query sums (stanard deviation, conf. intervals, variation coefficient, min/max)
    - edit sort_dataframe v2c function (add option to filter by difference from query RoG)
    - add min/max per metric (v3c5b)
    - add pocket_distance filtering to sort_dataframe
    - compute and add deltas to query summaries (add synthetic data point representing zero variation from query (0))

    TODO: 
    Add computed metrics (composite)
        i) local conformational flexibility: total range of variability between known alternative conformations, 
        ii) local conf. dispersion: avg. distance between known alternative conformations

    Add binding site / ligand promiscuity: 
        i) ligands per H entry avg, 
        ii) num. H entries / num. ligands avg, 
        iii) ligands per H state avg

    Compute & save additional query summaries: all (done), apo-only, holo-only

"""

import os
import glob
import tarfile
import pandas as pd
import io
import time
import numpy as np
import multiprocessing
#import sys
import argparse
import json
import sys

from parsing_functions_DBv2 import create_summary_df, sort_dataframe_v2c, split_dataframe, \
    DualOutput, calculate_averages_from_summaries, parse_query_summaries, compute_metrics



def save_arguments(args, parsing_dir):
    # Convert the Namespace object to a dictionary
    args_dict = vars(args)
    # Define the file path for the parameters JSON file
    param_file = os.path.join(parsing_dir, 'parameters.json')
    # Save the dictionary to a JSON file
    with open(param_file, "w") as json_file:
        json.dump(args_dict, json_file, indent=1)


def save_list_to_file(list_items, filename):
    with open(filename, 'w') as file:
        for item in list_items:
            file.write("%s\n" % item)


def replace_zero_with_nan(df, columns, negative_columns = ['bfac_t', 'bfac_avg']):
    """ columns: columns_that_should_not_have_zero """
    #negative_columns = ['bfac_t', 'bfac_avg']
    df[columns] = df[columns].replace({0: np.nan, '': np.nan})
    for column in negative_columns:
        if column in df.columns:
            df[column] = df[column].apply(lambda x: np.nan if pd.to_numeric(x, errors='coerce') < 0 else x)
    return df


def get_2level_cwd():
    npath = os.path.normpath(os.getcwd())
    path0 = os.path.join(npath.split(os.sep)[0], os.sep, npath.split(os.sep)[1], npath.split(os.sep)[2])
    if os.path.exists(path0):
        print(f'Root path found >> {path0}')
    else:
        print(f'Error finding root path in working dir: {npath}')
    return path0


def safe_mean(series):
    return pd.to_numeric(series, errors='coerce').mean()


def get_all_batch_dirs(base_path):
    batch_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    return batch_dirs


def is_query_result_dir(dir_path):
    return os.path.isfile(os.path.join(dir_path, 'global_results.csv'))


def collect_job_directories(base_path, batch_dirs):
    """ From batch_dirs returns all .tgz files and extracted query result directories """
    job_dict = {}
    for batch_dir in batch_dirs:
        full_batch_path = os.path.join(base_path, batch_dir)
        if not os.path.exists(full_batch_path):
            print(f"Batch directory {full_batch_path} does not exist.")
            continue

        # Collect all .tgz files
        tgz_files = glob.glob(os.path.join(full_batch_path, "*.tgz"))
        for file_path in tgz_files:
            job_name = os.path.basename(file_path).replace('.tgz', '')
            job_dict[job_name] = file_path

        # Collect all extracted query result directories
        dirs = [d for d in os.listdir(full_batch_path) if os.path.isdir(os.path.join(full_batch_path, d))]
        for dir_name in dirs:
            dir_path = os.path.join(full_batch_path, dir_name)
            if is_query_result_dir(dir_path):
                job_name = dir_name
                job_dict[job_name] = dir_path

    return job_dict


def select_first_n_jobs(job_dict, n=None):
    """
    Returns a dictionary containing the first n items from job_dict.
    If n is 0 or None, returns the whole dictionary.
    """
    if n is None or n == 0:
        return job_dict
    # Use itertools.islice for efficiency
    from itertools import islice
    return dict(islice(job_dict.items(), n))


def get_number_of_full_pockets(df, args):
    query_resis3 = df.loc[df['role'] == 'Q', '#resis3'].values[0]  # Directly grab the #resis3 value from the query row
    df = df[(df['role'] != 'C') | (df['pocket_len'] == query_resis3)]


def get_query_row_info(df_q):
    query_mean_rog, query_structure, query_unps, query_resis3 = np.nan, '', '', np.nan  # Initialize with proper values

    # Remove checks and assume that query row is not empty #if not df_q.empty:
    query_mean_rog = safe_mean(df_q['RoG']) if 'RoG' in df_q.columns else np.nan
    query_structure = df_q['structure'].iloc[0] if 'structure' in df_q.columns and not df_q['structure'].empty else ''
    query_unps = df_q['UNPs'].iloc[0] if 'UNPs' in df_q.columns and not df_q['UNPs'].empty else ''
    query_resis3 = df_q.iloc[0]['#resis3']

    return query_mean_rog, query_structure, query_unps, query_resis3


def extract_unique_ligands(df):
    # Convert all values in 'query_ligs' and 'nonquery_ligs' to strings
    df['query_ligs'] = df['query_ligs'].astype(str)
    df['nonquery_ligs'] = df['nonquery_ligs'].astype(str)
    ligands = []
    # Iterate through the 'query_ligs' and 'nonquery_ligs' columns
    for column in ['query_ligs', 'nonquery_ligs']:
        for item in df[column]:
            ligands.extend(item.split('-'))
    # Remove 'nan' strings and create a set of unique ligand names
    unique_ligands = set(ligand for ligand in ligands if ligand != 'nan' and pd.notna(ligand))
    # Return the set of unique ligand names and their count
    return unique_ligands, len(unique_ligands)


def extract_query_info(df_q, num_apo_pockets, num_holo_pockets, num_apo_0unobs, num_holo_0unobs, num_apo_obs100, num_holo_obs100, num_apo_obs70, num_unobs_apo30, num_unobs_apo50, num_unobs_pockets, crypt_A, crypt_H, crypt_A_obs, crypt_H_obs, crypt_A_full, crypt_H_full, unique_ligands, num_unique_ligands):
    columns = ['ahoj_query', 'qstruct', 'qchains3', 'qlig', 'num_all_pocket_ligs','qUNPs', 'num_apo_pockets', 'num_holo_pockets', 'num_apo_0unobs', 'num_holo_0unobs', 
               'num_apo_obs100', 'num_holo_obs100', 'num_apo_obs70', 'num_unobs30', 'num_unobs50', 'num_unobs100', 'ratio_apo_to_unobs30', 'ratio_apo_to_unobs30+', 
               'crypt_A', 'crypt_H', 'crypt_Ao', 'crypt_Ho', 'crypt_Af', 'crypt_Hf', 'all_pocket_ligs']

    if df_q.empty or len(df_q) < 1:
        raise ValueError("Query DataFrame (df_q) is empty or does not have any rows.")

    query_structure = df_q.iloc[0]['structure']
    query_chains3 = df_q.iloc[0]['chains3']
    query_POI = df_q.iloc[0]['query_POI']
    query_UNPs = df_q.iloc[0]['UNPs']
    poi_parts = query_POI.split('_')
    qlig = poi_parts[1] if len(poi_parts) > 1 else ''
    ahoj_query = f"{query_structure} {query_POI.replace('_', ' ')}"
    unique_ligands_str = '-'.join(unique_ligands)

    # New computed columns
    ratio_apo_to_unobs30 = round(( num_unobs_apo30 / num_apo_pockets) * 100, 1) if num_apo_pockets > 0 else 0
    num_apo_plus_unobs100 = num_apo_pockets + num_unobs_pockets
    ratio_apo_to_unobs30_plus = round(( (num_unobs_apo30 + num_unobs_pockets) / num_apo_plus_unobs100) * 100, 1) if num_apo_plus_unobs100 > 0 else 0

    query_info_df = pd.DataFrame([[ahoj_query, query_structure, query_chains3, qlig, num_unique_ligands, query_UNPs, num_apo_pockets, num_holo_pockets, num_apo_0unobs, num_holo_0unobs, num_apo_obs100, num_holo_obs100, num_apo_obs70, num_unobs_apo30, num_unobs_apo50, num_unobs_pockets, ratio_apo_to_unobs30, ratio_apo_to_unobs30_plus, crypt_A, crypt_H, crypt_A_obs, crypt_H_obs, crypt_A_full, crypt_H_full, unique_ligands_str]], columns=columns)
    return query_info_df


def process_and_filter_job(file_path, job_name, args):
    try:
    #if args:
        with tarfile.open(file_path, 'r:gz') as tar:
            file_name = './global_results.csv'
            if file_name in tar.getnames():
                file = tar.extractfile(file_name)
                df = pd.read_csv(io.TextIOWrapper(file, newline=""))

                # Drop rows where all values are nan (Empty AF row)
                df = df.dropna(how="all")

                # Replace zero with np.nan in columns that should not have zero
                df = replace_zero_with_nan(df, args.columns_nozero)

                # Filter global results dataframe incl. query
                df = sort_dataframe_v2c(df, sorting=0, full_pockets_aligned=args.full_pockets_aligned, bndgrsds_prcnt_seqres=args.bndgrsds_prcnt_seqres, 
                                        bndgrsds_prcnt_obs=args.bndgrsds_prcnt_obs, resolution_soft=args.resolution_soft, rog_diff_threshold_deltaQ=args.rog_diff_thrs_query, 
                                        rog_diff_threshold_mean=args.rog_diff_thrs_mean, pocket_distance=args.pocket_dist)

                # Extract query row(s)
                df_q = df[df['role'] == 'Q']

                # Discard job if query df is empty or has more than 1 rows
                if df_q.empty: return ('Discarded', (job_name, 'Empty query df'))
                elif len(df_q) > 1: return ('Discarded', (job_name, f'Multiple rows ({len(df_q)}) in query df'))

                # Get query row information
                query_mean_rog, query_structure, query_unps, query_resis3 = get_query_row_info(df_q)

                # Calculate statistics A for candidate metrics
                df_c = df[df['role'] == 'C']
                df_c_metrics = compute_metrics(df_c, df_q, deltas=2)
                
                # Save statistics for query (e.g., plots)
                # Function, get df_c_metrics, plot, save to mirror dir
                #plot_flexibility(df_c, save_dir=args.)

                df.insert(2, 'query_struct', query_structure)
                df.insert(4, 'query_UNPs', query_unps)

                # Split global results to apo, holo & unobserved
                apo_df, holo_df, dis_df = split_dataframe(df)

                # Get the (number of) rows where '%unobserved_rsd' is 30% or more and 50% or more
                num_apo_unobs50 = (apo_df['%unobserved_rsd'] >= 50).sum()   #apo_unobs50 = apo_df[apo_df['%unobserved_rsd'] >= 50]
                num_apo_unobs30 = (apo_df['%unobserved_rsd'] >= 30).sum()   #apo_unobs30 = apo_df[apo_df['%unobserved_rsd'] >= 30]

                # Get number of apo & holo states with 0 unobserved residues
                num_apo_0unobs = (apo_df['unobserved_rsd'] == 0).sum()
                num_holo_0unobs = (holo_df['unobserved_rsd'] == 0).sum()

                # Get number of apos that are over certain % observed
                num_apo_obs100 = (apo_df['%mapped_obs'] == 100).sum()       #apo_obs100 = apo_df[apo_df['%mapped_obs'] >= 100]
                num_holo_obs100 = (holo_df['%mapped_obs'] == 100).sum()
                num_apo_obs70 = (apo_df['%mapped_obs'] >= 70).sum()         #apo_obs70 = apo_df[apo_df['%mapped_obs'] >= 70]

                # Get number of cryptic pockets in query according to pocket RMSD # Now also getting full pockets separately
                crypt_A, crypt_H, crypt_A_obs, crypt_H_obs, crypt_A_full, crypt_H_full = '','','','','',''
                if args.crypt_rms:
                    crypt_rms_filtered_apo = apo_df[apo_df['pocket_rms'] >= args.crypt_rms]
                    crypt_A = len(crypt_rms_filtered_apo)
                    crypt_A_obs = (crypt_rms_filtered_apo['unobserved_rsd'] == 0).sum()
                    crypt_A_full = (crypt_rms_filtered_apo['pocket_len'] == query_resis3).sum()

                    crypt_rms_filtered_holo = holo_df[holo_df['pocket_rms'] >= args.crypt_rms]
                    crypt_H = len(crypt_rms_filtered_holo)
                    crypt_H_obs = (crypt_rms_filtered_holo['unobserved_rsd'] == 0).sum()
                    crypt_H_full = (crypt_rms_filtered_holo['pocket_len'] == query_resis3).sum()

                apo_df['RoG_distance'] = abs(apo_df['RoG'] - query_mean_rog).round(2)
                holo_df['RoG_distance'] = abs(holo_df['RoG'] - query_mean_rog).round(2)

                # Get the unique ligands and their count
                unique_ligands, num_unique_ligands = extract_unique_ligands(df)

                # Extract query info and build data frame
                query_info_df = extract_query_info(df_q, len(apo_df), len(holo_df), num_apo_0unobs, num_holo_0unobs, num_apo_obs100, num_holo_obs100, num_apo_obs70, num_apo_unobs30, num_apo_unobs50, len(dis_df), crypt_A, crypt_H, crypt_A_obs, crypt_H_obs, crypt_A_full, crypt_H_full, unique_ligands, num_unique_ligands)

                # Merge the metrics row with query_info_df
                query_info_df = pd.concat([query_info_df.iloc[:, :-1], df_c_metrics, query_info_df.iloc[:, -1:]], axis=1)

                # Check if the entry is paired
                if not apo_df.empty and not holo_df.empty: paired_entry = 1
                else: paired_entry = 0

                # Initialize apo and holo summary dfs
                apo_summary, holo_summary = pd.DataFrame(), pd.DataFrame()

                if args.apo_holo_sums == 1:
                    if not apo_df.empty:
                        apo_summary = create_summary_df(df_q, apo_df, 'A', paired_entry, job_name)
                    if not holo_df.empty:
                        holo_summary = create_summary_df(df_q, holo_df, 'H', paired_entry, job_name)

                return ('Success', (job_name, len(df), apo_summary, holo_summary, query_info_df))

            else:
                return ('File Not Found', (job_name, 0))

    except Exception as e:
        print(f'Exception while processing job {job_name}: {e}')
        return ('Error', (job_name, str(e), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))


def worker(job_name, file_path, args):
    result = process_and_filter_job(file_path, job_name, args)
    return (job_name, result)


def process_job_file_concurrently(job_dict, num_processes, args):
    aggregate_results = {
        'Success': {'count': 0, 'total_rows': 0, 'jobs': []},
        'File Not Found': {'count': 0, 'jobs': []},
        'Error': {'count': 0, 'jobs': []},
        'Discarded': {'count': 0, 'jobs': []}
    }

    # Processing step (process_and_filter_job)
    apo_values_list, holo_values_list, query_info_list = [], [], []
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.starmap(worker, [(job_name, file_path, args) for job_name, file_path in job_dict.items()])

    print('All job results loaded, processing results')
    for job_name, result in results:

        status, detail = result

        if status == 'Success':
            aggregate_results['Success']['count'] += 1
            aggregate_results['Success']['total_rows'] += detail[1]
            aggregate_results['Success']['jobs'].append(job_name)
            apo_values_list.append(detail[2].iloc[1:2])
            holo_values_list.append(detail[3].iloc[1:2])
            query_info_list.append(detail[4])
        elif status == 'File Not Found':
            aggregate_results['File Not Found']['count'] += 1
            aggregate_results['File Not Found']['jobs'].append(job_name)
        elif status == 'Discarded':
            aggregate_results['Discarded']['count'] += 1
            aggregate_results['Discarded']['jobs'].append((job_name, detail[1]))  # Save job name and reason
        else:
            aggregate_results['Error']['count'] += 1
            aggregate_results['Error']['jobs'].append(job_name)

    apo_summaries = pd.concat(apo_values_list, ignore_index=True)
    holo_summaries = pd.concat(holo_values_list, ignore_index=True)
    query_info_summaries = pd.concat(query_info_list, ignore_index=True)
    print('\nDone collecting summaries')

    return aggregate_results, apo_summaries, holo_summaries, query_info_summaries


def print_aggregate_results(aggregate_results):
    print("Processing Summary:")
    for category, data in aggregate_results.items():
        if category == 'Success':
            print(f"{category}: Jobs processed - {len(data['jobs'])}, Total rows - {data['total_rows']}")
        else:
            print(f"{category}: Jobs processed - {len(data['jobs'])}")
    print('')


def save_job_categories(aggregate_results, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for category, data in aggregate_results.items():
        filename = os.path.join(directory, f"{category}_jobs.txt")
        with open(filename, 'w') as file:
            if category == 'Discarded':
                # Write job names with reasons for discarded jobs
                for job_name, reason in data['jobs']:
                    file.write(f"{job_name}: {reason}\n")
            else:
                # Write job names for other categories
                for job_name in data['jobs']:
                    file.write(job_name + '\n')
    print("Job categories saved successfully.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process zipped AHoJ-DB raw data.")

    # Entry filtering
    parser.add_argument('--bndgrsds_prcnt_seqres',  type=int,   default=100,    help='None or int. Threshold for all %mapped_sqr (hard filter). Condition for keeping is >= threshold')
    parser.add_argument('--bndgrsds_prcnt_obs',     type=int,   default=100,    help='None or int. Threshold for all %mapped_obs (hard filter). Condition for keeping is >= threshold')
    parser.add_argument('--full_pockets_aligned',   type=int,   default=1,      help='Reject candidate pockets with unobserved residues (also rejects results with poorly mapped pockets')
    parser.add_argument('--resolution_soft',        type=float, default=3.0,    help='None or float. Filter by resolution and discard entries w/o resolution')

    # States filtering (apo - holo results within entry)
    parser.add_argument('--crypt_rms',              type=float, default=2.0,    help='pocket RMSD threshold for considering cryptic or flexible pockets')
    parser.add_argument('--rog_diff_thrs_query',    type=float, default=0.5,    help='float or None, e.g., 0.3: candidates whose RoG differs more than 0.3*query_RoG (+/-) from the (mean) query RoG are filtered out (lower=stricter)')
    parser.add_argument('--rog_diff_thrs_mean',     type=float, default=0.5,    help='float or None, e.g., 0.3: candidates whose RoG differs more than 0.3*mean_RoG (+/-) from the average candidate RoG are filtered out (lower=stricter)')
    parser.add_argument('--pocket_dist',            type=float, default=5.0,    help='Filter out pockets that are farther than X after global chain(s) alignment')

    # Processing options
    parser.add_argument('--batch_limit',            type=int,   default=0,       help='0: no limit, integer: number of batches. Number of batch directories to process')
    parser.add_argument('--batch_selection',        type=list,  default=['103'], help='Select which batch directory/ies to process')
    parser.add_argument('--num_processes',          type=int,   default=None,    help='Number of processes to use for multiprocessing, i.e., None (=50% cores) / 12 / 28.')
    parser.add_argument('--apo_holo_sums',          type=int,   default=0,       help='Calculate numerical summaries for apo and holo data frames (slow)')
    parser.add_argument('--columns_nozero',         nargs='+',  default=['RoG', 'bfac_t', 'bfac_avg', 'resolution'], help='Columns that should not have zero values, replace with nan (0: np.nan, "": np.nan)')
    #parser.add_argument('--union',                 type=int,   default=0,      help='0/1/. 1=Get database statistics (apoholo_averages) for pockets from entries with at least one apo and one holo pocket.')

    # Directory options
    parser.add_argument('--parsing_dir_suffix',     type=str,   default='_v3c5b',help='Suffix for parsing results directory name, that represents script version')
    parser.add_argument('--db_version',             type=str,   default='v2b',  help='AHoJ-DB version being parsed')
    parser.add_argument('--local_base_dir',         type=str,   default=None,   help='Input directory with unzipped tar files')
    parser.add_argument('--save_base_dir',          type=str,   default=None,   help='Base directory to save parsing results')
    parser.add_argument('--save_meta_dir',          type=str,   default=None,   help='Base directory to save entry meta-processing results (multiple sub-directories')

    return parser.parse_args()


def main():

    args = parse_arguments()
    start_time = time.time()
    two_level_cwd = get_2level_cwd()

    ## Input
    # Directory with raw AHoJ-DB data in batches
    if not args.local_base_dir:
        raise SystemExit('--local_base_dir is required (directory with the raw AHoJ-DB batches); there is no default')
    base_path = os.path.join(two_level_cwd, args.local_base_dir)

    ## Output
    # Base directory for saving the parsing results
    if not args.save_base_dir:
        raise SystemExit('--save_base_dir is required (where the parsing results go); there is no default')
    save_base_dir = args.save_base_dir
    # Base directory for saving entry meta-processing (i.e., plots, statistics)
    if not args.save_meta_dir:
        raise SystemExit('--save_meta_dir is required (where the per-entry plots and statistics go); there is no default')
    save_meta_dir = args.save_meta_dir

    # Select which database batch directories will be processed
    all_batch_dirs = get_all_batch_dirs(base_path)
    if args.batch_limit == 0:
        batch_dirs = all_batch_dirs
    else:
        batch_dirs = all_batch_dirs[:args.batch_limit]
    if args.batch_selection:
        batch_dirs = args.batch_selection
    print(f'\nSelected [{len(batch_dirs)}] batch dir(s) for processing:\n{batch_dirs}')

    print(f'Processing [{len(batch_dirs)}] batch directories out of [{len(all_batch_dirs)}] batches found in output folder')
    print('\n = Arguments from argparse =')
    print(args, '\n')

    job_dict = collect_job_directories(base_path, batch_dirs)
    total_job_folders = len(job_dict)
    print(f'Collected [{total_job_folders}] zipped jobs in [{len(batch_dirs)}] batch directories.\n')
    #for key, value in job_dict.items(): print(key, value)
    sys.exit(1)

    num_cpus = multiprocessing.cpu_count()
    num_processes = args.num_processes if args.num_processes else int(num_cpus / 2)
    print(f'{num_cpus} cores detected')
    print(f'Processing {total_job_folders} jobs with {num_processes} out of {num_cpus} cores')

    # Process global results dataframes
    result_counts, apo_summaries, holo_summaries, query_info_summaries = process_job_file_concurrently(job_dict, num_processes, args) # new
    #result_counts, combined_averages_df, apo_summaries, holo_summaries, query_info_summaries = process_job_file_concurrently(job_dict, num_processes, args) # old

    # Sort query summaries
    print('Sorting query summaries by computed local_flexibility')
    query_info_summaries = query_info_summaries.sort_values(by='local_flexibility', ascending=False)

    # Test save
    #filename_string = f'parsing{args.parsing_dir_suffix}_DB{args.db_version}_{len(batch_dirs)}batches_{total_job_folders}jobs_union{args.union}_{args.bndgrsds_prcnt_seqres}seqres_{args.bndgrsds_prcnt_obs}obs_{args.resolution_soft}res_{args.full_pockets_aligned}fullpcktaln'
    #parsing_dir = os.path.join(save_base_dir, filename_string)
    #os.makedirs(parsing_dir, exist_ok=True)
    #apo_summaries.to_csv(os.path.join(parsing_dir, 'apo_summaries_test.csv'), index=False)
    #holo_summaries.to_csv(os.path.join(parsing_dir, 'holo_summaries_test.csv'), index=False)

    combined_averages_df_all, combined_averages_df_paired = pd.DataFrame(), pd.DataFrame()

    if args.apo_holo_sums == 1:
        print('')
        print('Calculating averages over all apo and holo summaries (all entries)') # Even for entries with 0 apo or 0 holo
        combined_averages_df_all = calculate_averages_from_summaries(apo_summaries, holo_summaries, paired=0) # union 0
        print('Calculating averages over all apo and holo summaries (paired entries)') # At least one apo and one holo result
        combined_averages_df_paired = calculate_averages_from_summaries(apo_summaries, holo_summaries, paired=1) # union 1
        #combined_averages_df_paired, apo_summaries_rounded, holo_summaries_rounded = calculate_averages_from_summaries(apo_summaries, holo_summaries)

        metrics_to_discard = ['Metric', 'class', 'exp_method_combined', 'unpseq_overlap']
        combined_averages_df_all.drop(metrics_to_discard, inplace=True)
        combined_averages_df_paired.drop(metrics_to_discard, inplace=True)

        print('\nCombined apo and holo grand averages (all)')
        print(combined_averages_df_all.to_string(index=True))
        print('\nCombined apo and holo grand averages (paired)')
        print(combined_averages_df_paired.to_string(index=True))

    # Parse query summaries dataframe
    ligand_info_ratio, len_total, len_paired, paired_query_ratio, unique_uniprots_in, uniprots_paired, ddf_dis, ddf_dis2 = parse_query_summaries(query_info_summaries)

    uniprots_unpaired = unique_uniprots_in - uniprots_paired
    uniprots_unpaired_percent = round((len(uniprots_unpaired) / len(unique_uniprots_in)) * 100, 2)
    uniprots_paired_percent = round((len(uniprots_paired) / len(unique_uniprots_in)) * 100, 2)

    end_time = time.time()
    total_time = end_time - start_time

    filename_string = f'parsing{args.parsing_dir_suffix}_DB{args.db_version}_{len(batch_dirs)}batches_{total_job_folders}jobs_{args.bndgrsds_prcnt_seqres}seqres_{args.bndgrsds_prcnt_obs}obs_{args.resolution_soft}res_{args.full_pockets_aligned}fullpcktaln_sums{args.apo_holo_sums}'
    #filename_string = f'parsing{args.parsing_dir_suffix}_DB{args.db_version}_{len(batch_dirs)}batches_{total_job_folders}jobs_union{args.union}_{args.bndgrsds_prcnt_seqres}mpdseqres_{args.bndgrsds_prcnt_obs}mpdobs_{args.resolution_soft}res_{args.full_pockets_aligned}fullpckts'
    parsing_dir = os.path.join(save_base_dir, filename_string)
    os.makedirs(parsing_dir, exist_ok=True)
    report_file = os.path.join(parsing_dir, 'parsing_report.txt')
    print('')

    with DualOutput(report_file):
        print(f'Processed [{len(batch_dirs)}] batch directories using [{num_processes}] out of [{num_cpus}] cores')
        print('\nThresholds:')
        print('\n'.join(f'{k}={v}' for k, v in vars(args).items()), '\n')

        print(f'{len_paired} queries with at least one apo out of {len_total} ({paired_query_ratio} %)')
        print(f'Total UniProt ACs in queries: {len(unique_uniprots_in)}')
        print(f'UniProts with apo (paired): {len(uniprots_paired)} ({uniprots_paired_percent} %)')
        print(f'UniProts without apo (unpaired): {len(uniprots_unpaired)} ({uniprots_unpaired_percent} %)')
        print_aggregate_results(result_counts)
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Total job folders processed: {total_job_folders}")

    # Save parameters
    save_arguments(args, parsing_dir)

    query_summaries_csv = os.path.join(parsing_dir, 'query_summaries_rog_DQ5_2.csv')

    query_info_summaries.to_csv(query_summaries_csv, index=False)
    ligand_info_csv = os.path.join(parsing_dir, 'ligands_summary.csv')
    ligand_info_ratio.to_csv(ligand_info_csv, index=False)

    apo_summaries.to_csv(os.path.join(parsing_dir, 'apo_summaries.csv'), index=False)
    holo_summaries.to_csv(os.path.join(parsing_dir, 'holo_summaries.csv'), index=False)

    combined_averages_df_all.to_csv(os.path.join(parsing_dir, 'apoholo_averages_all.csv'), index=True)
    combined_averages_df_paired.to_csv(os.path.join(parsing_dir, 'apoholo_averages_paired.csv'), index=True)
    #combined_averages_csv = os.path.join(parsing_dir, 'apoholo_averages.csv')

    save_list_to_file(unique_uniprots_in, os.path.join(parsing_dir, 'unps_all_input.txt'))
    save_list_to_file(uniprots_paired, os.path.join(parsing_dir, 'unps_paired.txt'))
    save_list_to_file(uniprots_unpaired, os.path.join(parsing_dir, 'unps_unpaired.txt'))
    save_job_categories(result_counts, parsing_dir)
    dis_1_csv = os.path.join(parsing_dir, 'disorder_df.csv')
    ddf_dis.to_csv(dis_1_csv, index=False)
    dis_2_csv = os.path.join(parsing_dir, 'disorder_only_df.csv')
    ddf_dis2.to_csv(dis_2_csv, index=False)

    print(f'\nParsing results saved to {parsing_dir}')


if __name__ == '__main__':
    main()
