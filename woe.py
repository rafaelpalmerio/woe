df = pd.DataFrame([['0-39', 1244, 9514],
                   ['30-40', 2390, 21949],
                   ['40-50', 2893, 32144],
                   ['50-60', 2149, 32657],
                   ['60-70', 952, 26472],
                   ['70-80', 298, 12402],
                   ['80-90', 89, 4358],
                   ['90-130', 11, 478]],
                 columns=['age', 'bad', 'good'])


def get_woe(data, bad_col = 'bad', good_col = 'good'):
    df = data.copy()
    df['all'] = df[bad_col] + df[good_col]
    df['total_distribution'] = (df['all']) / df['all'].sum()
    df['bad_rate'] = df[bad_col] / df['all']
    df['distribution_good'] = df[good_col] / df[good_col].sum()
    df['distribution_bad'] = df[bad_col] / df[bad_col].sum()
    df['woe'] = np.log(df['distribution_good'] / df['distribution_bad'])*100

    iv = (
        (df['distribution_good'] - df['distribution_bad'])*
        np.log(df['distribution_good']/df['distribution_bad'])
    ).sum()
    
    return df, iv

class Woe:
    
    def fit(self, data, data_col, target_col, data_col_is_numeric, nbins=20):
        df = data.copy()
        self.data_col = data_col
        self.data_col_is_numeric = data_col_is_numeric
        self.target_col = target_col
        self.new_data_col = data_col
        if data_col_is_numeric:
            df['bin'], self.bins = pd.qcut(df[data_col], q=20, retbins=True)
            df['bin_code'] = df['bin'].cat.codes
            self.new_data_col = 'bin_code'
        res = get_woe(pd.crosstab(df[self.new_data_col], df[self.target_col]), 0, 1)[0].reset_index()
        self.d = {int(x[self.new_data_col]):x['woe'] for index, x in res.iterrows()}
        
    def transform(self, data):
        new_df = data.copy()
        if self.data_col_is_numeric:
            new_df['bin_code'] = pd.cut(new_df[self.data_col], bins=self.bins).cat.codes
        new_df['woe'] = new_df[self.new_data_col].map(self.d)
        
        return new_df['woe']
    
