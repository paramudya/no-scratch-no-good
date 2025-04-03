"""
    The technique or methods used may not be the most optimal or pythonic (likely it is what author can think up of the moment in time; exactly what is aimed to be improved).
    First iteration target is: 1. the class works 2. generates output exactly like its skleadn's counterpart; however certainly not as efficiently.
"""
import pandas as pd 

def if_numeric_col(serie: pd.Series): # belum dipakai
    if 0>0:
        return 0
    return 1


class EnjangDecisionTree:
    """
        Enjang Gaming, a renowned name in one of my FM24 saves, retired to write algorithms from scratch.
        Here is his first attempt, doing decision tree (coincidentally my work involves segmentation with decision tree currently)
    """
    def __init__(self, is_classification, min_samples_leaf_pct, max_leaves, max_depth = None,
                 eval_metric='mae', verbose=1):
        """ 
        Args:
            min_samples_leaf_pct: Minimum percentage of samples in each leaf.
            max_leaves: Maximum number of leaves (nodes with )
            max_depth: Number of depth (or iteration) maximum allowed before stopping.

            eval_metric: Technical metric to optimize for in training (options: mae)
        """

        self.params = {
            'min_samples_leaf_pct': min_samples_leaf_pct,
            'max_leaves': max_leaves,
            'max_depth': max_depth
        }

        self.is_classification = is_classification
        self.eval_metric = eval_metric
        self.iter_number = 1
        self.verbose = verbose
        
        # self.leaves = {}
        # self.nodes = {}

    # stopping params are going to be incorporated in these two funcs: is_leaf() and is_tree_stopping()
    def is_leaf(self, df):
        if len(df) == 1: # for now only this, later on there will be more params like min_samples_leaf_pct
            return 1
        elif df[self.target].nunique() == 1: # from logic that given all population contain the same target value, then it is a leaf objectively
            return 1
        else:
            return 0

    def is_tree_stopping(self):
        # for the meantime, stop when every single leaf contains 1 sample
        if all([item['active'] == 0 for item in self.temp_leaves]):
            print(f"no active nodes left (active nodes: {[item['active'] for item in self.temp_leaves]})")
            return 1
        
        # max depth constraint
        elif self.params['max_depth'] is not None and self.iter_number > self.params['max_depth']:
            print(f"max depth of {self.params['max_depth']} reached (iter #{self.iter_number})")
            return 1
        
        else: 
            print(f'\n\n\ncontinuing on the iteration to iter #{self.iter_number}')
            return 0

    def apply_rules(self, rules):
        temp_df = self.df.copy()

        for rule in rules:    
            temp_df = temp_df.query(rule) # later optimize to one str containing list of rules

        return temp_df

    def train(self, df, features, target):
        self.df = df
        self.features = features
        self.target = target
        self.temp_leaves = [{
            'rules': [],
            'len': len(self.df),
            'active': True,
            'is_confirmed_leaf': 0,
            }
        ]

        if not self.is_classification:
            print('defined as regression problem')
            # loop before any stopping criterion is achieved or each leaf contains no more than 1 sample
            while not self.is_tree_stopping():
                best_breakpoint = {
                    'score': 9999999999
                }

                # loop for active temp leaves (newly created nodes in the prev iter)
                active_temp_leaves = [l for l in self.temp_leaves if l['active']]
                print('number of active nodes left:', len(active_temp_leaves))

                for leaf in active_temp_leaves:
                    leaf['active'] = False # this is supposed to turn all the active in the original dict self.temp_leaves to False

                for i_active_leaf, temp_leaf in enumerate(active_temp_leaves):
                    temp_df = self.apply_rules(temp_leaf['rules'])
                    print(f'active leaf number {i_active_leaf+1} of {len(active_temp_leaves)}')
                    # loop each feature and check each possible data breakpoint
                    for feat in self.features:
                        # if is_numeric_col(temp_df[feat]): # assuming numeric col for now

                            ordered_feat =  list(set(temp_df[feat])) # set for order and distinct    

                            # loop all (n-1) rows of the feat        
                            for i in range(1,len(ordered_feat)):
                                breakpoint = sum(ordered_feat[i-1:i+1])/2 

                                # break into 2 groups separated by the breakpoint of the feature
                                group_1, group_2 = temp_df[temp_df[feat] >= breakpoint], temp_df[temp_df[feat] < breakpoint]
                                group_1['avg_target'], group_2['avg_target'] = sum(group_1[self.target])/len(group_1),sum(group_2[self.target])/len(group_2)
                                
                                if self.eval_metric == 'mae':
                                    sum_error = sum(abs(group_1['avg_target'] - group_1[self.target])) + sum(abs(group_2['avg_target'] - group_2[self.target]))
                                    metric_score = sum_error / len(temp_df)
                                # if self.eval_metric == '...'

                                if metric_score < best_breakpoint['score']: # assuming all metric score lesser better
                                    best_breakpoint = {
                                        'feature': feat,
                                        'value': breakpoint,
                                        'score': metric_score
                                        }

                                if self.verbose > 0:
                                    print(f'breaking {feat} on breakpoint val: {breakpoint}. {self.eval_metric}: {metric_score}')
                                    if metric_score < best_breakpoint['score']: # assuming all metric score lesser better
                                        print(f'new potential breakpoint: {best_breakpoint}')
                        
                    # loop active node level (after best breakpoint found)
                    print(f'Best breakpoint found on iter #{self.iter_number} active node ({i_active_leaf}): {best_breakpoint}\n')
                    # append group 1
                    rule_temp_group = temp_leaf['rules'] + [f'{best_breakpoint['feature']} >= {best_breakpoint['value']}']
                    df_temp_group = self.apply_rules(rule_temp_group)
                    confimed_leaf = self.is_leaf(df_temp_group)
                    self.temp_leaves.append(
                        { 
                        'rules': rule_temp_group,
                        'len': len(df_temp_group),
                        'active': not confimed_leaf,
                        'is_confirmed_leaf': confimed_leaf,
                        })
                    
                    # append group 2
                    rule_temp_group = temp_leaf['rules'] + [f'{best_breakpoint['feature']} < {best_breakpoint['value']}']
                    df_temp_group = self.apply_rules(rule_temp_group)
                    confimed_leaf = self.is_leaf(df_temp_group)
                    self.temp_leaves.append(
                        { 
                        'rules': rule_temp_group,
                        'len': len(df_temp_group),
                        'active': not confimed_leaf,
                        'is_confirmed_leaf':  confimed_leaf,
                        })
                
                # loop iter level
                print(f'tree nodes after iter #{self.iter_number}')
                print(*self.temp_leaves,sep='\n')

                self.iter_number += 1


                    
