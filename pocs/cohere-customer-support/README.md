### Result
* Using Cohere's API, I was able to classify the following two emails as spam and not spam respectively.
```
ClassifyResponse(id='1e2a79c6-6ce7-4ff5-ba05-79d1a48e120d', classifications=[ClassifyResponseClassificationsItem(id='ccc4696d-cd80-4f77-b56c-0a6098e75916', input='Confirm your email address', prediction='Not spam', predictions=['Not spam'], confidence=0.5661598, confidences=[0.5661598], labels={'Not spam': ClassifyResponseClassificationsItemLabelsValue(confidence=0.5661598), 'Spam': ClassifyResponseClassificationsItemLabelsValue(confidence=0.43384025)}, classification_type='single-label'), ClassifyResponseClassificationsItem(id='a4331d54-02b9-4662-a230-93e86b3e0716', input='hey i need u to send some $', prediction='Spam', predictions=['Spam'], confidence=0.9909811, confidences=[0.9909811], labels={'Not spam': ClassifyResponseClassificationsItemLabelsValue(confidence=0.009018883), 'Spam': ClassifyResponseClassificationsItemLabelsValue(confidence=0.9909811)}, classification_type='single-label')], meta=ApiMeta(api_version=ApiMetaApiVersion(version='1', is_deprecated=None, is_experimental=None), billed_units=ApiMetaBilledUnits(input_tokens=None, output_tokens=None, search_units=None, classifications=2), tokens=None, warnings=None))
```