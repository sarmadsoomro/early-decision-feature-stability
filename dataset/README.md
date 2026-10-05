# Datasets

Neither dataset is redistributed here. Both are public: download them and
unpack the CSVs directly into this directory (no subfolders). The scripts find
this directory through `code/scripts/paths.py`. To keep the data elsewhere, set
`FS_DATA_DIR` to that directory.

## Expected contents

All eight files are needed. The scripts read six of the seven OULAD tables
(`courses.csv` is not read) and `xapi_edu_data.csv`, and every run records the
SHA-256 of every `.csv` file in the data directory. `stability_bootstrap.py`
refuses to summarise if that record differs from the one written by the CV
run, so keep exactly these files here, unchanged, and no other `.csv` files.
The stored outputs were produced with files matching these checksums:

| File | Source | Rows | SHA-256 |
|---|---|---|---|
| `assessments.csv` | OULAD | 206 | `9ce92381e4a0ac457f8e251b2eb2c179ed51e023ab30d1d671a0268bd62316ba` |
| `courses.csv` | OULAD | 22 | `4f16eee7454b15e109b0a21a0e43be820e6846ed6f9301bb7feb5ab5ad737a75` |
| `studentAssessment.csv` | OULAD | 173,912 | `b1839e17618a5de36b6d189a242d3123f7939a56c8c9ccef0c17b9199c614a6b` |
| `studentInfo.csv` | OULAD | 32,593 | `815fbc3e2de29f79900bc63f0345a35db62c9146afaa1d86e7107b8b3beffc60` |
| `studentRegistration.csv` | OULAD | 32,593 | `bbc87a0de1fe2a9ec6decce1ca6c1a02cf17b54ad3bdcc673fc6413b41c61d47` |
| `studentVle.csv` | OULAD | 10,655,280 | `52668253d876c5becbcb72185977152700cecab2942aca807fecc3dd54b937f0` |
| `vle.csv` | OULAD | 6,364 | `d1b28303dea802ad87b4484e1196e878e06824850b9a4fe8aa34693439fe87e9` |
| `xapi_edu_data.csv` | xAPI-Edu-Data | 480 | `7ec40f0d14e616233d8ceff1ccbaa6d31947d537a1c5befb29e9fb00b453cee3` |

Rows exclude the header line. The files take about 465 MB, of which
`studentVle.csv` is 454 MB; it exceeds GitHub's 100 MB per-file limit, which is
one reason the CSVs are not committed.

## Open University Learning Analytics Dataset (OULAD)

<https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset>

Licence: CC BY 4.0. Cite Kuzilek, Hlosta and Zdrahal (2017), *Scientific Data*
4:170171. The download is one zip archive of the seven tables above.

What the scripts read:

| Table | Used for |
|---|---|
| `studentInfo.csv` | Registrations, demographics, final result |
| `studentRegistration.csv` | Registration and withdrawal dates |
| `studentVle.csv`, `vle.csv` | Daily clicks per VLE resource, and each resource's activity type |
| `studentAssessment.csv`, `assessments.csv` | Submission dates and scores, and each assessment's deadline and type |

## xAPI-Edu-Data

<https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data>

Cite Amrieh, Hamtini and Aljarah (2016). A Kaggle account is needed to
download it. Rename the file to `xapi_edu_data.csv` if it arrives under another
name; the loader looks for that exact name.

## Analysis samples

The 32,593 rows of `studentInfo.csv` are registrations, not the analysis
sample. At each decision point the scripts drop registrations made after the
cutoff and registrations withdrawn on or before it, then split off the 2014J
holdout and remove students who also appear in 2014J from the development
data. The resulting sizes are in Table 1 of the manuscript and in `n_dev` and
`n_holdout` of each `oulad_dayN_holdout.json`. No rows are filtered on feature
values.
