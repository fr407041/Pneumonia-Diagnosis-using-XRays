Pneumonia Diagnosis using XRays from Kaggle Data Sets
===============
<h3 id="Introduction"> Data Introduction </h3>
The data is from Kaggle : https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
<br>( The original data is from https://data.mendeley.com/datasets/rscbjbr9sj/2 )
<br>Here we just use kaggle's data to analysis.

There are two category in kaggle's data sets : Normal and PNEUMONIA (PEUMONIA can split into virus and bacteria, but currently we just consider Normal & PNEUMONIA to study)
![image](https://github.com/fr407041/Pneumonia-Diagnosis-using-XRays/blob/master/image/2category.png)

<h3> Data explore </h3>
The kaggle's data have split data set into 3 folders : train、val and test.
<br>The train folder totally have 5216 jpg files (Normal:1341，PNEUMONIA:3875).
<br>The val folder totally have 16 jpg files (Normal:8，PNEUMONIA:8).
<br>The test folder totally have 624 jpg files (Normal:234，PNEUMONIA:390).

**Remark\! The train folder is an imbalance data sets for Normal & PNEUMONIA (about 1:3)**
<h3> Model Building </h3>
