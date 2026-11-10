# TG3K within-block split

Run `python make_split.py` to regenerate the CSV and JSON manifests from
`E:\Thyroid Dataset\tg3k`. Use `--data` for another location. Paths in the
manifest are relative to the dataset root; the source JPG files are untouched.

The 15 candidate groups are contiguous runs of identical image dimensions.
They are not verified patient or original-video IDs. A single validation
window comprising about 15% of each group is placed at 30%, 50%, or 70% of
the run in a rotating pattern. At least 25 frames (or 10% of the group) on
each side are marked `unused`, leaving separated train and validation frames.

This split makes both train and validation contain all 15 observable groups
and reduces adjacent-frame leakage. It does not measure generalization to new
patients or devices. Use independent, patient-separated robotic sweep data for
the final test. Do not report the within-block validation score as a
patient-independent test score.

`split_manifest.csv` lists every image and mask. `group_summary.csv` gives
the group boundaries and counts. `split_ids.json` provides plain ID lists.
