
Trying to understand the bug.

1. in the `rdt.transformers.categorical.OneHotEncodingTransformer`, (though method is defined in `.base`) the output columns are not defined:
```bash
(Pdb) self.get_output_columns()
['None.value0', 'None.value1', 'None.value2', 'None.value3', 'None.value4', 'None.value5', 'None.value6']
(Pdb) data.columns
Index(['None.value0', 'None.value1', 'None.value2', 'None.value3',
       'None.value4', 'None.value5', 'None.value6'],
      dtype='object')
(Pdb) data.head()
    None.value0   None.value1   None.value2   None.value3   None.value4   None.value5   None.value6
0  8.544042e-01  2.655355e-17  7.015797e-10  1.271636e-01  1.522267e-02  3.209622e-03  4.481079e-38
1  5.825917e-25  8.926271e-43  1.741067e-35  3.992792e-23  2.501956e-29  5.874925e-33  1.000000e+00
2  1.000000e+00  2.567566e-31  9.168337e-29  3.365345e-13  1.293156e-11  3.809055e-27  0.000000e+00
3  1.000000e+00  3.928383e-17  1.207312e-22  4.372707e-18  5.076192e-15  3.916351e-18  3.940451e-42
4  1.000000e+00  1.901732e-15  3.121897e-34  8.550410e-18  1.568426e-16  2.234726e-16  4.063766e-43
(Pdb) self.output_columns is None
True

```

2. in dp_cgans.data_transformer._inverse_transform_discrete()
- `ohe.get_input_columns() is None ` is True 
- inherits from `self._column_transform_info_list` in line 189 of data_transformer.py