"""
    Signature:
        - Instead of X and y as input to fit/train and predict/test; 
          df, features, and target are used, aside from distinction to the sklearn's ver, for a more straight forward understanding for amateurs like me
        - 

    Comment:
        The technique or methods used may not be the most optimal or pythonic (likely it is what author can think up of the moment in time; exactly what is aimed to be improved).
        First iteration target is: 1. the class works 2. generates output exactly like its skleadn's counterpart; however certainly not as efficiently.
"""
import pandas as pd 

def if_numeric_col(serie: pd.Series): # belum dipakai
    if 0>0:
        return 0
    return 1

def calc_metric_score(temp_df, feat, breakpoint, target, eval_metric='mse'):
    # break into 2 groups separated by the breakpoint of the feature

    group_1, group_2 = temp_df[temp_df[feat] >= breakpoint], temp_df[temp_df[feat] < breakpoint]
    group_1['avg_target'], group_2['avg_target'] = sum(group_1[target])/len(group_1),sum(group_2[target])/len(group_2)
    
    error_abs_list = list(abs(group_1['avg_target'] - group_1[target])) + list(abs(group_2['avg_target'] - group_2[target]))
    if eval_metric == 'mae':
        sum_error = sum(error_abs_list)
        return sum_error / len(temp_df)
    
    elif eval_metric == 'mse':
        sum_sq_error = sum([err**2 for err in error_abs_list])
        return sum_sq_error / len(temp_df)

class EnjangDecisionTree:
    """
        Enjang Gaming, a renowned name in one of my FM24 saves, retired to write algorithms from scratch.
        Here is his first attempt, doing decision tree (coincidentally my work involves segmentation with decision tree currently)
    """
    def __init__(self, is_classification, min_samples_leaf_pct=None, min_samples_leaf=None, max_leaves=None, max_depth=None,
                 eval_metric='mse', verbose=1):
        """ 
        Args:
            min_samples_leaf_pct: Minimum percentage of samples in each leaf.
            max_leaves: Maximum number of leaves (nodes with )
            max_depth: Number of depth (or iteration) maximum allowed before stopping.

            eval_metric: Technical metric to optimize for in training (options: mae)
        """

        self.params = {
            'min_samples_leaf_pct': min_samples_leaf_pct if min_samples_leaf_pct is not None else 0,
            'min_samples_leaf': min_samples_leaf if min_samples_leaf is not None else 0,
            'max_leaves': max_leaves if max_leaves is not None else 9999999,
            'max_depth': max_depth if max_depth is not None else 9999999
        }
        print(self.params)

        self.is_classification = is_classification
        self.eval_metric = eval_metric
        self.iter_number = 1
        self.verbose = verbose
        self.rules_raw = ''

        # self.leaves = {}
        # self.nodes = {}

    # stopping params are going to be incorporated in these 3 funcs: _is_gonna_break_rule(), _is_leaf(), and _is_tree_stopping() 
    def _is_gonna_break_rule(self): # for min samples leaf (pct and not) and current leaf is >=2 times the minimum
        self.temp_min_samples_node = min(self.len_df_temp_group)
        print('self.temp_min_samples_node',self.temp_min_samples_node,'\n',"self.params['min_samples_leaf']",self.params['min_samples_leaf'])
        if self.temp_min_samples_node < self.params['min_samples_leaf']:
            return True
        else:
            return False
    
    def _is_leaf(self, df):
        if len(df) == 1: # for now only this, later on there will be more params like min_samples_leaf_pct or min_samples_leaf (not pct)
            return True
        elif len(df) < 2*self.params['min_samples_leaf']: # when <2 times the min, in 0 way would the next iter have both groups contain at least the min samples in a leaf
            return True
        elif df[self.target].nunique() == 1: # from logic that given all population contain the same target value, then it is a leaf objectively
            return True
        else:
            return False

    def _is_tree_stopping(self):
        # for the meantime, stop when every single leaf contains 1 sample
        if all([item['active'] == 0 for item in self.temp_leaves]):
            print(f"no active nodes left (active nodes: {[item['active'] for item in self.temp_leaves]})")
            return True
        
        # max depth constraint
        elif self.params['max_depth'] is not None and self.iter_number > self.params['max_depth']:
            # all active nodes are then deemed as 'leaves'
            self.temp_leaves = [
                {**leaf, "is_confirmed_leaf": True, "active": False} if leaf["active"] == True else leaf
                for leaf in self.temp_leaves
            ]
            print(f"max depth of {self.params['max_depth']} reached (iter #{self.iter_number})")
            return True
        
        else: 
            print(f'\n\n\ncontinuing on the iteration to iter #{self.iter_number}')
            return False

    def _apply_rules(self, rules):
        temp_df = self.df.copy()

        for rule in rules:    
            temp_df = temp_df.query(rule) # later optimize to one str containing list of rules

        return temp_df
    def test(self, df_test): # wow hadnt thought of it before but train input being this way (df feats target) allow a single var input for test
        pass # finish tomoz, now zzz first

    def train(self, df, features, target):
        self.df = df
        self.features = features
        self.target = target
        self.temp_leaves = [{
            'rules': [],
            'len': len(self.df),
            'active': True,
            'is_confirmed_leaf': True,
            }
        ]

        if not self.is_classification:
            print('defined as regression problem')
            # loop before any stopping criterion is achieved or each leaf contains no more than 1 sample
            while not self._is_tree_stopping():
                best_breakpoint = {
                    'score': 9999999999
                }

                # loop for active temp leaves (newly created nodes in the prev iter)
                active_temp_leaves = [l for l in self.temp_leaves if l['active']]
                print('number of active nodes left:', len(active_temp_leaves))

                for leaf in active_temp_leaves:
                    leaf['active'] = False # this is supposed to turn all the active in the original dict self.temp_leaves to False

                for i_active_leaf, temp_leaf in enumerate(active_temp_leaves):
                    temp_df = self._apply_rules(temp_leaf['rules'])
                    print(f'active leaf number {i_active_leaf+1} of {len(active_temp_leaves)}')
                    # loop each feature and check each possible data breakpoint
                    for feat in self.features:
                        # if is_numeric_col(temp_df[feat]): # assuming numeric col for now

                            ordered_feat =  list(set(temp_df[feat])) # set for order and distinct  
                            ordered_feat.sort()
                            # loop all (n-1) rows of the feat        
                            for i in range(1,len(ordered_feat)):
                                breakpoint = sum(ordered_feat[i-1:i+1])/2 

                                metric_score = calc_metric_score(temp_df, feat, breakpoint, self.target,
                                                                 eval_metric=self.eval_metric)
                                
                                if self.verbose > 0:
                                    print(f'breaking {feat} on breakpoint val (from {ordered_feat[i-1:i+1]}): {breakpoint}. {self.eval_metric}: {metric_score}')
                 
                                if metric_score < best_breakpoint['score']: # potential winning breakpoint: assuming for now all metric score lesser better
                                    self.rule_temp_group = [temp_leaf['rules'] + [f'{feat} > {breakpoint}'], 
                                                    temp_leaf['rules'] + [f'{feat} <= {breakpoint}']]
                                    self.df_temp_group = [self._apply_rules(self.rule_temp_group[0]),
                                                          self._apply_rules(self.rule_temp_group[1])]
                                    self.len_df_temp_group = [len(self.df_temp_group[0]),
                                                         len(self.df_temp_group[1])]
                                    
                                    if self.verbose > 0: print(f'\tpotential new breakpoint')

                                    if not self._is_gonna_break_rule(): # eventual winning breakpoint
                                        best_breakpoint = {
                                            'feature': feat,
                                            'value': breakpoint,
                                            'score': metric_score
                                            }
                                        if self.verbose > 0: print(f'\t\teventual new breakpoint')
                                    else: # at least one of the new groups have samples smaller than required
                                        if self.verbose > 0:
                                            print(f'\t\tbreakpoint ineligible due to potential node (smallest: {self.temp_min_samples_node}) smaller than params')


                    # loop active node level (after best breakpoint found)
                    print(f'Best breakpoint found on iter #{self.iter_number} active node ({i_active_leaf}): {best_breakpoint}\n')
                    # append group 1
                    confimed_leaf = self._is_leaf(self.df_temp_group[0])
                    self.temp_leaves.append(
                        { 
                        'rules': self.rule_temp_group[0],
                        'len': self.len_df_temp_group[0],
                        'active': not confimed_leaf,
                        'is_confirmed_leaf': confimed_leaf,
                        })
                    
                    # append group 2
                    confimed_leaf = self._is_leaf(self.df_temp_group[1])
                    self.temp_leaves.append(
                        { 
                        'rules': self.rule_temp_group[1],
                        'len': self.len_df_temp_group[1],
                        'active': not confimed_leaf,
                        'is_confirmed_leaf':  confimed_leaf,
                        })
                
                # loop iter level
                print(f'tree nodes after iter #{self.iter_number}')
                print(*self.temp_leaves,sep='\n')

                self.iter_number += 1


                    
