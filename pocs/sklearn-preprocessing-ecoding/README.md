### Result
* sklearn
* Preprocessing
* Encoding
* Encoding works by converting categorical data into numerical data.
* It is generally useful to encode the input data before feeding it to the model.

Result:
```
Original Data:
   class     sex  embark_town
0  Third    male  Southampton
1  First  female    Cherbourg
2  Third  female  Southampton
3  First  female  Southampton
4  Third    male  Southampton

Encoded Data:
   class_First  class_Second  class_Third  sex_female  sex_male  embark_town_Cherbourg  embark_town_Queenstown  embark_town_Southampton  embark_town_nan
0          0.0           0.0          1.0         0.0       1.0                    0.0                     0.0                      1.0              0.0
1          1.0           0.0          0.0         1.0       0.0                    1.0                     0.0                      0.0              0.0
2          0.0           0.0          1.0         1.0       0.0                    0.0                     0.0                      1.0              0.0
3          1.0           0.0          0.0         1.0       0.0                    0.0                     0.0                      1.0              0.0
4          0.0           0.0          1.0         0.0       1.0                    0.0                     0.0                      1.0              0.0

```

