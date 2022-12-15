import pandas as pd

df = pd.DataFrame({'score': ['low', 'high', 'medium', 'medium', 'low']})
print(df)

mapping = {'low': 0, 'medium': 1, 'high': 2}

df['score'] = df['score'].replace(mapping)
print(df)