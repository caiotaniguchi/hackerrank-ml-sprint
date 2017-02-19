# contest: HackerRank Machine Learning CodeSprint
# challenge: Challenge Recommendation
# date: September 4th, 2016
# username: caiotaniguchi
# name: Caio Taniguchi
# email: caiotaniguchi@gmail.com

# Importing packages
import pandas as pd
import numpy as np

print('Loading data files...')
challenges = pd.read_csv('challenges.csv')
submissions = pd.read_csv('submissions.csv')

print('Modifying datasets...')
# Add a flag to determine if a hacker solved a given challenge or not
tmp = submissions.groupby(['hacker_id', 'challenge_id'], as_index=False)['solved'].agg(np.sum)
tmp['solved'] = tmp['solved'] > 0
submissions = submissions.drop('solved', axis=1).merge(tmp, on=['hacker_id', 'challenge_id'], how='left')

# Add whether or not a challenge is part of the target contest
challenges['in_target_contest'] = challenges['contest_id'] == 'c8ff662c97d345d2'
challenges.drop('contest_id', axis=1, inplace=True)

# Add submission count and solved submission count per challenge
submission_count = challenges.groupby('challenge_id', as_index=False).agg({
        'solved_submission_count': np.sum,
        'total_submissions_count': np.sum
    })
challenges = pd.merge(challenges.drop(['solved_submission_count', 'total_submissions_count'], axis=1),
                      submission_count, on='challenge_id', how='left')

# Remove duplicate entries and challenges that are not part of submissions.csv
challenges = challenges.sort_values(by=['challenge_id', 'in_target_contest'], ascending=[True, False])
challenges = challenges.drop_duplicates('challenge_id')

# Prepare data to generate the models
hackers = np.unique(submissions['hacker_id'])
recommendations = challenges[challenges['in_target_contest']]\
                    .sort_values(by=['solved_submission_count', 'total_submissions_count'], ascending=False)

recommendations.fillna('Unknown', inplace=True)
submissions_full = submissions.merge(challenges, on='challenge_id', how='left')
submissions_full.fillna('Unknown', inplace=True)
final_submission = []

print('Generating recommendations... This should take a while...')
for hacker in hackers:
    to_remove = submissions.loc[submissions['hacker_id'] == hacker, ['challenge_id', 'solved']].drop_duplicates()
    to_retry = to_remove.loc[~to_remove['solved'], 'challenge_id']
    to_remove = to_remove.loc[to_remove['solved'], 'challenge_id']
    domain_filter = np.unique(submissions_full.loc[submissions_full['hacker_id'] == hacker, 'domain'])
    subdomain_filter = np.unique(submissions_full.loc[submissions_full['hacker_id'] == hacker, 'subdomain'])
    dom_recs = recommendations[recommendations['domain'].isin(domain_filter)]
    subdom_recs = dom_recs[dom_recs['subdomain'].isin(subdomain_filter)]
    dom_subdom_recs = pd.concat([to_retry, subdom_recs['challenge_id'], dom_recs['challenge_id']]).drop_duplicates()
    cur_recs = dom_subdom_recs[-dom_subdom_recs.isin(to_remove)]
        
    if cur_recs.shape[0] < 10:
        default_recs = recommendations.loc[-recommendations['challenge_id'].isin(to_remove), 'challenge_id']
        cur_recs = pd.concat([cur_recs, dom_subdom_recs, default_recs]).drop_duplicates()
        
    tmp = {
        'hacker_id': hacker,
        'c_1': cur_recs.iloc[0],
        'c_2': cur_recs.iloc[1],
        'c_3': cur_recs.iloc[2],
        'c_4': cur_recs.iloc[3],
        'c_5': cur_recs.iloc[4],
        'c_6': cur_recs.iloc[5],
        'c_7': cur_recs.iloc[6],
        'c_8': cur_recs.iloc[7],
        'c_9': cur_recs.iloc[8],
        'c_10': cur_recs.iloc[9]
    }
    final_submission.append(tmp)

final_submission = pd.DataFrame(final_submission)  
cols = final_submission.columns.tolist()[::-1]
final_submission = final_submission[cols]
final_submission.head()    

print('Writting recommendation.csv file...')
final_submission.to_csv('recommendation.csv', header=False, index=False)