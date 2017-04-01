#!/usr/bin/env python
# encoding: utf-8
#
# Copyright SAS Institute
#
#  Licensed under the Apache License, Version 2.0 (the License);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

'''
Base class for estimators

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import uuid
import re
# from pdb import set_trace as bp

DTREE_MAP = {
    'max_branches': 'proc_option_maxbranch',
    'max_depth'   : 'proc_option_maxdepth',
    'leaf_size'   : 'minleafsize',
    'inputs'      : 'input',
    'n_bins'      : 'proc_option_intervalbins',
    'prune'       : 'prune',
    'alpha'       : 'alpha',
    'nominals'    : 'nominals'
}

FOREST_MAP = {
    'max_depth': 'proc_option_maxdepth',
    'leaf_size': 'proc_option_leafsize',
    'inputs'   : 'input',
    'nominals' : 'nominals',
    'alpha'    :'proc_option_alpha',
    'bootstrap': 'proc_option_inbagfraction',
    'n_bins'   :'proc_option_intervalbins',
    'seed'     :'proc_option_seed',
    'n_trees'  :'proc_option_maxtrees'
}

GB_MAP = {
    'leaf_size'     : 'proc_option_leafsize',
    'learning_rate' : 'proc_option_shrinkage',
    'max_branches'  : 'proc_option_maxbranches',
    'max_depth'     : 'proc_option_maxdepth',
    'n_bins'        : 'proc_option_TB_nbins', # this is handled special
    'n_trees'       : 'proc_option_iterations',
    'seed'          : 'proc_option_seed',
    'subsample_rate': 'proc_option_trainproportion',
    'var_importance': 'importance',
    'inputs'        : 'input',
    'nominals'      : 'nominals'
}

NN_MAP = {
    'hiddens'    : 'hidden',
    'inputs'     : 'input',
    'num_tries'  : 'numtries',
    'target_act' :'target_act',
    'nominals'   : 'nominals'
}

REG_MAP = {
    'selection'  : 'selection',
    'intercept'  : 'intercept',
    'sl'         : 'sl',
    'criterion'  : 'criterion',
    'max_effects':'maxeffects'

}

SCORE_MAP = {
    'NObs': 'NObsUsed'
}


class EstimatorMixIn(object):
    ''' Additional estimator methods '''
    def remap_params(self, est, params):
        '''
        Remap key names and values

        Returns
        -------
        dict

        '''
        out = {}
        procoptsDict = {}

        if est == 'RF':
            MAP = FOREST_MAP
        elif est == 'DT':
            MAP = DTREE_MAP
        elif est == 'GB':
            MAP = GB_MAP
        elif est == 'NN':
            MAP = NN_MAP
        elif est == 'REG':
            MAP = REG_MAP
        else:
            print('ERROR in remap_parms')
            return None

        for key, value in params.items():
            if hasattr(value, 'replace'):
                value = value.replace('_', '')
            k = MAP.get(key, None)
            if (k):
                if k.startswith('proc_option_'):
                    k2 = k[12:]
                    if k2.casefold() == 'seed':
                        value = int(value)
                    if k2.casefold() ==  'alpha':
                        if value == 0:
                            value = 1
                    procoptsDict[k2] = value
                else:
                    out[k] = value

        if procoptsDict:
            if est == 'GB':
                n_bins = procoptsDict.pop('TB_nbins', None)
                if n_bins:
                    procoptsDict['INTERVALBINS'] = n_bins
                    procoptsDict['CATEGORICALBINS'] = n_bins    
            procopts = ' '.join('{}={}'.format(key, val) for key, val in procoptsDict.items())
            out['procopts'] = procopts

        return out

    def create_model_table(self, sas):
        '''
        Create an output table definition

        Parameters
        ----------
        sas : SASsession object
            The connection to use to create the output table

        Raises
        ------
        ValueError
            If the connection object is None

        Returns
        -------
        Dict object (path and filename)

        '''
        if sas is None:
            raise ValueError('There is no connection object to create the output table.')

        ll = sas.submit('libname work list')

        path1 = ll['LOG'].partition('Physical Name=')[2].partition('\n')[0].strip().rstrip() + '/'
        # name = ('%s%s' % (prefix, uuid.uuid4())).replace('-', '_')
        name1 = 'A' + uuid.uuid4().hex[1:32]

        return dict(path=path1, name=name1)


class ModelMixIn(object):
    ''' Additional model methods '''

    def _format_score_info(self, value, nominal=True):
        ''' Normalize the score output '''
        value = value.iloc[:, 0]
        value.index.name = None
        value.name = None
        if nominal:
            value.MisClassificationRate *= 100
        for i, x in value.iteritems():
            if isinstance(x, str):
                value[i] = x.strip()
        value.index = [SCORE_MAP.get(x, x) for x in value.index]
        return value

    def _checkLogForError(self, log):
        lines = re.split(r'[\n]\s*', log)
        for line in lines:
            if line.startswith('ERROR'):
                return (False, line)
        return (True, '')

    def commonScore(self, table, algo=''):
        ''' Common Score function for estimators '''
        self._check_backend(table)
        self.ml = table.sas.sasml()
        self.stat = table.sas.sasstat()
        data = self.data
        results = self.diagnostics
        params = self.get_params()
        if 'ERROR_LOG' in results._names:
            raise ValueError("Model fitting Failed. See the Log for Details ")

        target = params.pop('target')
        name     = data.get('name')
        codename = data.get('path') + name
        sortOrder = 'DESC'
        if algo.casefold() in ['decisionforest', 'gbtree']:
            table.sas.set_batch(True)
            tmp = table.sas.sasdata("scoredata", libref=results._name)
            table.sas.set_batch(False)
            if algo.casefold() == 'gbtree':
                res = self.ml.treeboost(data=table, score=tmp, procopts="inmodel="+results._name+".model")
                # does the scoredata exist
                exists = table.sas.exist('scoredata', libref=results._name)
                if not exists:
                    raise ValueError("Score File does not exist after scoring")
                # does the scoring log have errors
                check, errorMsg = self._checkLogForError(res._log)
                if not check:
                    raise ValueError("Scoring Failed: "+ errorMsg)

            elif algo.casefold() == 'decisionforest':
                res = self.ml.hp4score(data=table, score=dict(file=codename, out=tmp))
                if 'ERROR_LOG' in res._names:
                    raise ValueError("Scoring Failed. See the Log for Details ")
        else:
            if algo.casefold() in ['decisiontree', 'logisticregression']:
                sortOrder='ACS'
            ll = table.sas.submit("data "+results._name+".scoredata;\nset "+table.libref+"."+table.table+ ";\n%include '"+codename+"';\nrun;")
            # does the scoredata exist
            exists = table.sas.exist('scoredata', libref=results._name)
            if not exists:
                raise ValueError("Score File does not exist after scoring")
            check, errorMsg = self._checkLogForError(ll['LOG'])
            if not check:
                raise ValueError("Scoring Failed: "+ errorMsg)
            tmp = table.sas.sasdata("scoredata", libref=results._name)

        results._names = list(set(['SCOREDATA']) | set(results._names))

        nomTarget = False
        params['nominals'] = [x.casefold() for x in params['nominals']]
        if target.casefold() in params['nominals']:
            nomTarget = True
        event = self.assessPrep(tmp, target, nominal=nomTarget, sort=sortOrder)
        assessResults = tmp.assessModel(target=target, prediction=str("P_" + target + event ), nominal=nomTarget, event=event)

        # copy assessResults to model results
        copy_string = """
                proc copy in={0} out={1};
                    select ASSESSMENTBINSTATISTICS ASSESSMENTSTATISTICS;
                run;"""
        table.sas.submit(copy_string.format(assessResults._name, results._name))
        if not table.sas.exist("ASSESSMENTBINSTATISTICS", libref=results._name):
            raise ValueError('Model Assessment Failed. ')

        results._names = list(set(['ASSESSMENTSTATISTICS', 'ASSESSMENTBINSTATISTICS']) | set(results._names))
        tr = table.sas.sasdata("ASSESSMENTSTATISTICS", libref = results._name)
        if target.casefold() in params['nominals']:
            return self._format_score_info(tr.to_df().transpose())
        else:
            return self._format_score_info(tr.to_df().transpose(), nominal=False)

    def assessPrep(self, table=None, target='', nominal=True, sort='DESC'):
        """

        Parameters
        ----------
        table
        sort

        Returns
        -------

        """
        if nominal:
            s = str('(' + sort + ')')
            code_string = """
            proc delete data=work._DMDBTARGET; run;
            proc hpdmdb data={0}.{1}{2} classout=work._DMDBTARGET(keep=name nraw craw level frequency nmisspercent);
                class {3} {4};
            run;
            data _null_;
                file LOG;
                set work._DMDBTARGET(where=(level ^= ''));
                put 'STARTLIST';
                if _n_=1 then put LEVEL;
                put 'STARTLISTend';
                stop;
            run;
            """
        else:
            s = ''
            code_string = """
            proc delete data=work._DMDBTARGET; run;
            proc hpdmdb data={0}.{1}{2} varout=work._DMDBTARGET;
                var {3};
            run;
            data _null_;
                file LOG;
                set work._DMDBTARGET(where=(name ^= ''));
                put 'STARTLIST';
                if _n_=1 then put NAME;
                put 'STARTLISTend';
                stop;
            run;
            """            


        # ignore teach_me_SAS mode to run contents
        nosub = table.sas.nosub
        table.sas.nosub = False
        ll = table.sas.submit(code_string.format(table.libref, table.table, table._dsopts(), target, s))
        check, errorMsg = self._checkLogForError(ll['LOG'])
        if not check:
            raise ValueError("Failed to process event level: "+ errorMsg)
        table.sas.nosub = nosub
        l2 = ll['LOG'].partition("STARTLIST\n")
        l2 = l2[2].rpartition("STARTLISTend\n")
        dlist1 = l2[0].split("\n")
        del dlist1[len(dlist1) - 1]
        dlist1 = [x.casefold() for x in dlist1]
        if nominal:
            return dlist1[0]
        else:
            if dlist1[0].casefold() == target.casefold():
                return ''  
