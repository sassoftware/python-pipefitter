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
SAS Imputer Implementations

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import pandas as pd
import uuid
from .... import transformer


class Imputer(transformer.Imputer):
    ''' SAS Imputer Implementation '''

    def transform(self, table, value=transformer.Imputer.MEAN):
        '''
        Fill data missing values with specified values

        Parameters
        ----------
        table : SASdata
            The table to impute
        value : ImputerMethod or scalar or dict or Series or DataFrame, optional
            Specifies the value to use in place of missing values.
                * If an ImputerMethod is specified, that method is used for all
                  missing values.
                * If a scalar is specified, that value is used to substitute for
                  all missings.
                * If a dict is specified, the keys correspond to the columns and
                  the values are the substitution values (which may also be
                  ImputerMethod instances).
                * If a Series is specified, the index corresponds to the columns
                  and the values are the substitution values.

        Returns
        -------
        SASdata

        '''
        if (table.table.startswith('_imp_')): 
           tname = table.table
        else:
           tname = "_imp_"+table.sas._io._logcnt()+table.table[0:18] 
        sql = "proc sql;\n  select\n"
        ds1 = "data "+table.libref+"."+tname+"; set "+table.libref+"."+table.table+";\n"

        sqlsel  = '    %s(%s),\n'
        dsmiss  = '  if missing(%s) then do;\n    %s = %s;\n  end;\n'
        sqlinto = '  into\n'

        modesql = ''
        modeq   = "proc sql outobs=1;\n  select %s, count(*) as freq into :imp_mode_%s, :imp_mode_freq\n"
        modeq  += "  from %s where %s is not null group by %s order by freq desc, %s;\nquit;\n"

        # get list of variables and types
        code  = "data _null_; d = open('"+table.libref+"."+table.table+"');\n"
        code += "nvars = attrn(d, 'NVARS');\n"
        code += "vn='VARNUMS='; vl='VARLIST='; vt='VARTYPE=';\n"
        code += "put vn nvars; put vl;\n"
        code += "do i = 1 to nvars; var = varname(d, i); put var; end;\n"
        code += "put vt;\n"
        code += "do i = 1 to nvars; var = vartype(d, i); put var; end;\n"
        code += "run;"
        ll = table.sas.submit(code, "text")
        l2 = ll['LOG'].rpartition("VARNUMS= ")
        l2 = l2[2].partition("\n")
        nvars = int(l2[0])
        l2 = l2[2].partition("\n")
        varlist = l2[2].upper().split("\n", nvars)
        del varlist[nvars]
        l2 = l2[2].partition("VARTYPE=")
        l2 = l2[2].partition("\n")
        vartype = l2[2].split("\n", nvars)
        del vartype[nvars]

        vars = dict(zip(varlist, vartype))
        # DataFrame
        if isinstance(value, (pd.DataFrame, type(table))):
            raise TypeError('DataFrame-like replacements are not supported')

        # Convert Series / etc. to a dictionary
        if hasattr(value, 'to_dict'):
            value = value.to_dict()
            for k, v in value.items():
                if isinstance(v, (list, tuple)):
                    value[k] = v[0]

        globalval = False
        # Replace the missing values
        if not isinstance(value, dict):
           globalval = True
           method    = [value] * nvars
           value     = dict(zip(varlist, method))

        if isinstance(value, dict):
            for col, val in value.items():
               if not isinstance(val, transformer.ImputerMethod):
                  if type(val) == str:
                     if vars.get(col.upper()) != 'N':
                        ds1     += dsmiss % (col, col, '"'+str(val)+'"')
                     else:
                        if not globalval:
                           raise TypeError("Column '%s' is numeric, but substitution value is character." % col)
                  else:
                     if vars.get(col.upper()) == 'N':
                        ds1     += dsmiss % (col, col, val)
                     else:
                        if not globalval:
                           raise TypeError("Column '%s' is character, but substitution is numeric." % col)

               elif val == transformer.Imputer.MAX:
                  sql     += sqlsel %('max', col)
                  sqlinto += '    :imp_max_'+col+',\n'
                  if vars.get(col.upper()) == 'N':
                     ds1     += dsmiss % (col, col, '&imp_max_'+col+'.')
                  else:
                     ds1     += dsmiss % (col, col, '"&imp_max_'+col+'."')

               elif val == transformer.Imputer.MIN:
                  sql     += sqlsel %('min', col)
                  sqlinto += '    :imp_min_'+col+',\n'
                  if vars.get(col.upper()) == 'N':
                     ds1     += dsmiss % (col, col, '&imp_min_'+col+'.')
                  else:
                     ds1     += dsmiss % (col, col, '"&imp_min_'+col+'."')

               elif val == transformer.Imputer.MODE:
                  modesql    += modeq %(col, col, table.libref+"."+table.table, col, col, col)
                  if vars.get(col.upper()) == 'N':
                     ds1     += dsmiss % (col, col, '&imp_mode_'+col+'.')
                  else:
                     ds1     += dsmiss % (col, col, '"&imp_mode_'+col+'."')

               elif vars.get(col.upper()) != 'N':
                  continue 
     
               elif val == transformer.Imputer.MEAN:
                  sql     += sqlsel %('mean', col)
                  sqlinto += '    :imp_mean_'+col+',\n'
                  ds1     += dsmiss % (col, col, '&imp_mean_'+col+'.')

               elif val == transformer.Imputer.MEDIAN:
                  sql     += sqlsel %('median', col)
                  sqlinto += '    :imp_median_'+col+',\n'
                  ds1     += dsmiss % (col, col, '&imp_median_'+col+'.')

               elif val == transformer.Imputer.MIDRANGE:
                  sql     += sqlsel %('max', col)
                  sqlinto += '    :imp_max_'+col+',\n'
                  sql     += sqlsel %('min', col)
                  sqlinto += '    :imp_min_'+col+',\n'
                  ds1     += dsmiss % (col, col, '(&imp_min_'+col+'.'+' + '+'&imp_max_'+col+'.'+') / 2')

               elif val == transformer.Imputer.RANDOM:
                  sql     += sqlsel %('max', col)
                  sqlinto += '    :imp_max_'+col+',\n'
                  sql     += sqlsel %('min', col)
                  sqlinto += '    :imp_min_'+col+',\n'
                  ds1     += dsmiss % (col, col, '&imp_min_'+col+'.'+' + (&imp_max_'+col+'.'+' - &imp_min_'+col+'.'+') * ranuni(0)')
               else:
                  print('HOW DID I GET to HERE?????')

            if len(sql) > 20:
               sql = sql.rstrip(', \n')+'\n'+sqlinto.rstrip(', \n')+'\n  from '+table.libref+'.'+table.table+';\nquit;\n' 
            else:
               sql = ''
            ds1 += 'run;\n'


        ll = table.sas.submit(modesql+sql+ds1)
        outtable = table.sas.sasdata(tname, libref=table.libref, results=table.results, dsopts=table.dsopts)
        return outtable


