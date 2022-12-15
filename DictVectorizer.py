from sklearn.feature_extraction import DictVectorizer

X_dict = [{'interest': 'tech', 'occupation': 'professional'},
{'interest': 'fashion', 'occupation': 'student'},
{'interest': 'fashion','occupation':'professional'},
{'interest': 'sports', 'occupation': 'student'},
{'interest': 'tech', 'occupation': 'student'},
{'interest': 'tech', 'occupation': 'retired'},
{'interest': 'sports','occupation': 'professional'}]

enc = DictVectorizer(sparse=False)
X_dict_enc = enc.fit_transform(X_dict)

print(X_dict_enc)
print(enc.vocabulary_)

